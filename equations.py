from __future__ import annotations

from functools import cache
from typing import Callable, Union, cast

import numpy as np
import scipy.sparse.linalg as spla  # type: ignore
from numpy.typing import NDArray
from scipy import sparse  # type: ignore

from finite import (
    Difference,
    DifferenceNonUniformGrid,
    DifferenceUniformGrid,
    Domain,
    NonUniformPeriodicGrid,
    UniformNonPeriodicGrid,
    UniformPeriodicGrid,
)
from timesteppers import RK22, CrankNicolson, EquationSet, StateVector


class DiffusionBC(EquationSet):
    @cache
    def _crank_nicolson(self, dt: float) -> Callable[[], None]:
        c = self.X.variables[0]
        M, _ = c.shape
        Mmat = sparse.lil_array(sparse.eye(M + 2))
        Mmat[0, -2] = 1
        Mmat[-3, -1] = 1
        Lmat = sparse.lil_array((M + 2, M + 2))
        Lmat[:M, :M] = -self.D * sparse.csc_array(self.d2x.matrix)
        LHS = (Mmat + dt / 2 * Lmat).tolil()
        RHS = (Mmat - dt / 2 * Lmat).tolil()
        LHS[M:, :] = 0
        RHS[M:, :] = 0
        LHS[-2, 0] = 1
        LHS[-1, :M] = self.dx.matrix[-1, :]
        LU = spla.splu(LHS.tocsc())
        RHS = RHS[:, :-2].tocsc()
        return lambda: np.copyto(c, LU.solve(RHS @ c)[:-2, :])  # type: ignore

    def __init__(
        self,
        c: NDArray[np.float64],
        D: float,
        spatial_order: int,
        domain: Domain,
    ) -> None:

        self.dx = _diff_grid(1, spatial_order, domain.grids[0], 0)
        self.d2x = _diff_grid(2, spatial_order, domain.grids[0], 0)
        d2y = _diff_grid(2, spatial_order, domain.grids[1], 1)

        self.t = 0.0
        self.iter = 0
        M, N = c.shape
        self.D = D

        self.X = StateVector([c])
        # M*(C1-C0)/dt + L*(C1+C0)/2 == 0
        # M*(C1-C0)/dt == -L*(C1+C0)/2
        # M*(C1-C0)/dt == -L*(C1+C0)/2+d
        # M*(C1-C0) == -dt*L/2*(C1+C0)+dt*d
        # (M+dt*L/2)*C1-dt*d = (M-dt*L/2)*C0

        # self.M = sparse.eye(M)
        # self.L = -D * sparse.csc_array(d2x.matrix)
        # self.xstep = CrankNicolson(self, 0)

        self.M = sparse.eye(N)
        self.L = -D * sparse.csc_array(d2y.matrix)
        self.ystep = CrankNicolson(self, 1)

    def step(self, dt: float) -> None:
        self.ystep.step(dt / 2)
        self._crank_nicolson(dt)()
        self.ystep.step(dt / 2)
        self.t += dt
        self.iter += 1


class Wave2DBC(EquationSet):
    def __init__(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        p: NDArray[np.float64],
        spatial_order: int,
        domain: Domain,
    ) -> None:
        dx = _diff_grid(1, spatial_order, domain.grids[0], 0)
        dy = _diff_grid(1, spatial_order, domain.grids[1], 1)
        self.X = StateVector([u, v, p])

        def f(X: StateVector) -> NDArray[np.float64]:
            X.scatter()
            u, v, p = X.variables
            du = dx @ p
            dv = dy @ p
            dp: NDArray[np.float64] = dx @ u + dy @ v
            return -cast(
                NDArray[np.float64],
                np.concatenate((du, dv, dp), axis=0),  # type: ignore
            )

        self.F = f

        def bc(X: StateVector) -> None:
            X.scatter()
            u, v, p = X.variables
            u[0, :] = 0
            u[-1, :] = 0
            X.gather()

        self.BC = bc


class ReactionDiffusion2D(EquationSet):
    def __init__(
        self,
        c: NDArray[np.float64],
        D: float,
        dx2: DifferenceUniformGrid,
        dy2: DifferenceUniformGrid,
    ):
        self.t = 0.0
        self.iter = 0
        M, N = c.shape

        self.X = StateVector([c])
        self.F = lambda X: X.data * (1 - X.data)
        self.tstep = RK22(self)

        self.M = sparse.eye(M)
        self.L = -D * sparse.csc_array(dx2.matrix)
        self.xstep = CrankNicolson(self, 0)

        self.M = sparse.eye(N)
        self.L = -D * sparse.csc_array(dy2.matrix)
        self.ystep = CrankNicolson(self, 1)

    def step(self, dt: float) -> None:
        self.xstep.step(dt / 2)
        self.ystep.step(dt / 2)
        self.tstep.step(dt)
        self.ystep.step(dt / 2)
        self.xstep.step(dt / 2)
        self.t += dt
        self.iter += 1


def _diff_grid(
    derivative_order: int,
    convergence_order: int,
    grid: Union[UniformPeriodicGrid, UniformNonPeriodicGrid, NonUniformPeriodicGrid],
    axis: int,
) -> Difference:
    def as_odd(x: int) -> int:
        return x + (1 - x % 2)

    def as_even(x: int) -> int:
        return x + x % 2

    if isinstance(grid, NonUniformPeriodicGrid):
        if derivative_order % 2 == 1:  # 奇数阶导数的时候凑偶数代数精度
            return DifferenceNonUniformGrid(
                derivative_order, as_even(convergence_order), grid, axis
            )
        else:  # 偶数阶导数的时候凑奇数代数精度
            return DifferenceNonUniformGrid(
                derivative_order, as_odd(convergence_order), grid, axis
            )
    else:
        return DifferenceUniformGrid(
            derivative_order, as_even(convergence_order), grid, axis
        )


class ViscousBurgers2D(EquationSet):
    def __init__(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        nu: float,
        spatial_order: int,
        domain: Domain,
    ):

        dx = _diff_grid(1, spatial_order, domain.grids[0], 0)
        d2x = _diff_grid(2, spatial_order, domain.grids[0], 0)
        dy = _diff_grid(1, spatial_order, domain.grids[1], 1)
        d2y = _diff_grid(2, spatial_order, domain.grids[1], 1)

        self.t = 0.0
        self.iter = 0
        M, N = u.shape
        self.X = StateVector([u, v], 0)

        def f(X: StateVector) -> NDArray[np.float64]:
            X.scatter()
            u, v = X.variables
            return -cast(
                NDArray[np.float64],
                np.concatenate(  # type: ignore
                    (
                        np.multiply(u, dx @ u) + np.multiply(v, dy @ u),
                        np.multiply(u, dx @ v) + np.multiply(v, dy @ v),
                    ),
                    axis=0,
                ),
            )

        self.F = f
        self.tstep = RK22(self)

        self.M = sparse.eye(2 * M)
        self.L = sparse.bmat(
            [
                [-nu * sparse.csc_array(d2x.matrix), sparse.csr_matrix((M, M))],
                [sparse.csr_matrix((M, M)), -nu * sparse.csc_array(d2x.matrix)],
            ]
        )
        self.xstep = CrankNicolson(self, 0)

        self.X = StateVector([u, v], 1)
        self.M = sparse.eye(2 * N)
        self.L = sparse.bmat(
            [
                [-nu * sparse.csc_array(d2y.matrix), sparse.csr_matrix((N, N))],
                [sparse.csr_matrix((N, N)), -nu * sparse.csc_array(d2y.matrix)],
            ]
        )
        self.ystep = CrankNicolson(self, 1)

    def step(self, dt: float) -> None:
        self.xstep.step(dt / 2)
        self.ystep.step(dt / 2)
        self.tstep.step(dt)
        self.ystep.step(dt / 2)
        self.xstep.step(dt / 2)
        self.t += dt
        self.iter += 1


class ViscousBurgers(EquationSet):
    def __init__(
        self,
        u: NDArray[np.float64],
        nu: float,
        d: Union[DifferenceUniformGrid, DifferenceNonUniformGrid],
        d2: Union[DifferenceUniformGrid, DifferenceNonUniformGrid],
    ):
        self.u = u
        self.X = StateVector([u])

        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu * d2.matrix

        def f(X: StateVector) -> NDArray[np.float64]:
            return -X.data * (d @ X.data)

        self.F = f


class Wave(EquationSet):
    def __init__(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        d2: Union[DifferenceUniformGrid, DifferenceNonUniformGrid],
    ):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)  # noqa: E741
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01], [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01], [L10, L11]])

        self.F = lambda X: 0 * X.data


class SoundWave(EquationSet):
    def __init__(
        self,
        u: NDArray[np.float64],
        p: NDArray[np.float64],
        d: Union[DifferenceUniformGrid, DifferenceNonUniformGrid],
        rho0: Union[float, NDArray[np.float64]],
        gammap0: Union[float, NDArray[np.float64]],
    ):
        (N,) = u.shape
        I = sparse.eye(N, N)  # noqa: E741
        Z = sparse.csr_matrix((N, N))
        self.X = StateVector([u, p])
        self.M = sparse.bmat([[rho0 * sparse.csc_array(I), Z], [Z, I]])
        self.L = sparse.bmat([[Z, d.matrix], [gammap0 * sparse.csc_array(d.matrix), Z]])
        self.F = lambda X: np.zeros(X.data.shape)


class ReactionDiffusion(EquationSet):
    def __init__(
        self,
        c: NDArray[np.float64],
        d2: Union[DifferenceUniformGrid, DifferenceNonUniformGrid],
        c_target: Union[float, NDArray[np.float64]],
        D: float,
    ):
        (N,) = c.shape
        self.X = StateVector([c])
        self.M = sparse.eye(N, N)
        self.L = -D * d2.matrix
        self.F = lambda X: X.data * (c_target - X.data)
