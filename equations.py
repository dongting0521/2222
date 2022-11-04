from __future__ import annotations

from typing import Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse  # type: ignore

from finite import (
    Difference,
    DifferenceNonUniformGrid,
    DifferenceUniformGrid,
    Domain,
    UniformPeriodicGrid,
)
from timesteppers import RK22, CrankNicolson, EquationSet, StateVector


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


class ViscousBurgers2D(EquationSet):
    def __init__(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        nu: float,
        spatial_order: int,
        domain: Domain,
    ):
        def odd(x: int) -> int:
            return x // 2 * 2 + 1

        def even(x: int) -> int:
            return (x + 1) // 2 * 2

        dx: Difference
        d2x: Difference
        dy: Difference
        d2y: Difference
        if isinstance(domain.grids[0], UniformPeriodicGrid):
            dx = DifferenceUniformGrid(1, even(spatial_order), domain.grids[0], 0)
            d2x = DifferenceUniformGrid(2, even(spatial_order), domain.grids[0], 0)
        else:
            dx = DifferenceNonUniformGrid(1, even(spatial_order), domain.grids[0], 0)
            d2x = DifferenceNonUniformGrid(2, odd(spatial_order), domain.grids[0], 0)

        if isinstance(domain.grids[1], UniformPeriodicGrid):
            dy = DifferenceUniformGrid(1, even(spatial_order), domain.grids[1], 1)
            d2y = DifferenceUniformGrid(2, even(spatial_order), domain.grids[1], 1)
        else:
            dy = DifferenceNonUniformGrid(1, even(spatial_order), domain.grids[1], 1)
            d2y = DifferenceNonUniformGrid(2, odd(spatial_order), domain.grids[1], 1)

        self.t = 0.0
        self.iter = 0
        M, N = u.shape
        self.X = StateVector([u, v], 0)

        def f(X: StateVector) -> NDArray[np.float64]:
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
                [nu * d2x.matrix, sparse.csr_matrix((M, M))],
                [sparse.csr_matrix((M, M)), nu * d2x.matrix],
            ]
        )
        self.xstep = CrankNicolson(self, 0)

        self.X = StateVector([u, v], 1)
        self.M = sparse.eye(2 * N)
        self.L = sparse.bmat(
            [
                [nu * d2y.matrix, sparse.csr_matrix((N, N))],
                [sparse.csr_matrix((N, N)), nu * d2y.matrix],
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
