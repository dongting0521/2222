from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import deque
from functools import cache
from typing import Any, Callable, Optional, cast

import numpy as np
import scipy.sparse.linalg as spla  # type: ignore
from numpy.typing import NDArray
from scipy import sparse  # type: ignore
from scipy.special import factorial  # type: ignore

from farray import apply_matrix, axslice
from finite import Difference


class StateVector:
    def __init__(self, variables: list[NDArray[np.float64]], axis: int = 0):
        self.axis = axis
        var0 = variables[0]
        shape = list(var0.shape)
        self.N = shape[axis]
        shape[axis] *= len(variables)
        self.shape = tuple(shape)
        self.data = np.zeros(shape)
        self.variables = variables
        self.gather()

    def gather(self) -> None:
        for i, var in enumerate(self.variables):
            np.copyto(  # type: ignore
                self.data[axslice(self.axis, i * self.N, (i + 1) * self.N)], var
            )

    def scatter(self) -> None:
        for i, var in enumerate(self.variables):
            np.copyto(  # type: ignore
                var, self.data[axslice(self.axis, i * self.N, (i + 1) * self.N)]
            )


class EquationSet(metaclass=ABCMeta):
    X: StateVector
    M: NDArray[np.float64]
    L: NDArray[np.float64]
    F: Optional[Callable[[StateVector], NDArray[np.float64]]]


class Timestepper(metaclass=ABCMeta):
    t: float
    iter: int
    X: StateVector
    dt: Optional[float]

    @abstractmethod
    def _step(self, dt: float) -> NDArray[np.float64]:
        pass

    def __init__(self) -> None:
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt: float) -> None:
        self.X.gather()
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1

    def evolve(self, dt: float, time: float) -> None:
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper, metaclass=ABCMeta):
    X: StateVector
    BC: Optional[Callable[[StateVector], None]]

    def __init__(self, eq_set: EquationSet):
        assert eq_set.F is not None
        super().__init__()
        self.X = eq_set.X
        self.F = eq_set.F
        if hasattr(eq_set, "BC"):
            self.BC = eq_set.BC  # type: ignore
        else:
            self.BC = None

    def step(self, dt: float) -> None:
        super().step(dt)
        if self.BC:
            self.BC(self.X)


class ForwardEuler(ExplicitTimestepper):
    def _step(self, dt: float) -> NDArray[np.float64]:
        return self.X.data + dt * self.F(self.X)


class LaxFriedrichs(ExplicitTimestepper):
    def __init__(self, eq_set: EquationSet):
        super().__init__(eq_set)
        N = len(self.X.data)
        A = sparse.diags([1 / 2, 1 / 2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1 / 2
        A[-1, 0] = 1 / 2
        self.A = A

    def _step(self, dt: float) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self.A @ self.X.data + dt * self.F(self.X))


class Leapfrog(ExplicitTimestepper):
    def _step(self, dt: float) -> NDArray[np.float64]:
        if self.iter == 0:
            self.X_old = self.X.data.copy()
            return self.X.data + dt * self.F(self.X)
        else:
            X_temp: NDArray[np.float64] = self.X_old + 2 * dt * self.F(self.X)
            self.X_old = self.X.data.copy()
            return X_temp


class LaxWendroff(ExplicitTimestepper):
    def __init__(
        self,
        X: StateVector,
        F1: Callable[[StateVector], NDArray[np.float64]],
        F2: Callable[[StateVector], NDArray[np.float64]],
    ):
        self.t = 0
        self.iter = 0
        self.X = X
        self.F1 = F1
        self.F2 = F2

    def _step(self, dt: float) -> NDArray[np.float64]:
        return (
            self.X.data
            + dt * self.F1(self.X)
            + cast(NDArray[np.float64], dt**2 / 2 * self.F2(self.X))
        )


class Multistage(ExplicitTimestepper):
    def __init__(
        self,
        eq_set: EquationSet,
        stages: int,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
    ):
        super().__init__(eq_set)
        self.stages = stages
        self.a = a
        self.b = b

        self.X_list = []
        self.K_list = []
        for i in range(self.stages):
            self.X_list.append(StateVector([var.copy() for var in self.X.variables]))
            self.K_list.append(self.X.data.copy())

    def _step(self, dt: float) -> NDArray[np.float64]:
        X = self.X
        X_list = self.X_list
        K_list = self.K_list
        stages = self.stages

        np.copyto(X_list[0].data, X.data)  # type: ignore
        for i in range(1, stages):
            K_list[i - 1] = self.F(X_list[i - 1])

            np.copyto(X_list[i].data, X.data)  # type: ignore
            # this loop is slow -- should make K_list a 2D array
            for j in range(i):
                X_list[i].data += self.a[i, j] * dt * K_list[j]
            if self.BC:
                self.BC(X_list[i])

        K_list[-1] = self.F(X_list[-1])

        # this loop is slow -- should make K_list a 2D array
        for i in range(stages):
            X.data += self.b[i] * dt * K_list[i]

        return X.data


def RK22(eq_set: EquationSet) -> Multistage:
    a = np.array([[0, 0], [1 / 2, 0]])
    b = np.array([0, 1])
    return Multistage(eq_set, 2, a, b)


class AdamsBashforth(ExplicitTimestepper):
    f_list: deque[NDArray[np.float64]]

    def __init__(self, eq_set: EquationSet, steps: int, dt: float):
        super().__init__(eq_set)
        self.steps = steps
        self.dt = dt
        self.f_list = deque()
        for i in range(self.steps):
            self.f_list.append(self.X.data.copy())

    def _step(self, dt: float) -> NDArray[np.float64]:
        f_list = self.f_list
        f_list.rotate()
        f_list[0] = self.F(self.X)
        if self.iter < self.steps:
            coeffs = self._coeffs(self.iter + 1)
        else:
            coeffs = self._coeffs(self.steps)

        for i, coeff in enumerate(coeffs):
            self.X.data += self.dt * coeff * self.f_list[i].data

        return self.X.data

    def _coeffs(self, num: int) -> NDArray[np.float64]:

        i = (1 + np.arange(num))[None, :]
        j = (1 + np.arange(num))[:, None]
        S = (-i) ** (j - 1) / factorial(j - 1)

        b = (-1) ** (j + 1) / factorial(j)

        a: NDArray[np.float64] = np.linalg.solve(S, b)  # type: ignore
        return a


class ImplicitTimestepper(Timestepper):
    LU: spla.SuperLU

    def __init__(self, eq_set: EquationSet, axis: int):
        super().__init__()
        self.axis = axis
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L

    def _LUsolve(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.axis == 0:
            return cast(NDArray[np.float64], self.LU.solve(data))
        elif self.axis == len(data.shape) - 1:
            return cast(NDArray[np.float64], self.LU.solve(data.T).T)
        else:
            raise ValueError("Can only do implicit timestepping on first or last axis")


class BackwardEuler(ImplicitTimestepper):
    def _step(self, dt: float) -> NDArray[np.float64]:
        if dt != self.dt:
            self.LHS = cast(Any, self.M + dt * self.L)
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec="NATURAL")
        self.dt = dt
        return self._LUsolve(self.X.data)


class CrankNicolson(ImplicitTimestepper):
    def _step(self, dt: float) -> NDArray[np.float64]:
        if dt != self.dt:
            self.LHS = cast(Any, self.M + dt / 2 * self.L)
            self.RHS = cast(Any, self.M - dt / 2 * self.L)
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec="NATURAL")
        self.dt = dt
        return self._LUsolve(apply_matrix(self.RHS, self.X.data, self.axis))


class BackwardDifferentiationFormula(Timestepper):
    steps: int
    thist: list[float]
    uhist: list[NDArray[np.float64]]

    def __init__(
        self,
        u: NDArray[np.float64],
        L_op: Difference,
        steps: int,
    ):
        super().__init__()
        self.u = u
        self.func = L_op
        self.steps = steps
        self.thist = []
        self.uhist = []

    def _step(self, dt: float) -> NDArray[np.float64]:
        self.thist.append(dt)
        self.uhist.append(self.u)
        steps = min(self.steps, len(self.uhist))
        solve = self._coeff(tuple(self.thist[-steps:]))
        return solve(np.stack(self.uhist[-steps:], axis=1))

    @cache
    def _coeff(
        self, thist: tuple[float, ...]
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        (N,) = self.u.shape
        steps = len(thist)
        x = np.cumsum(np.array((0,) + thist))
        xx = x[-1]
        x /= xx
        coeff = np.zeros((steps + 1,))
        for i in range(steps + 1):
            poly = np.array([1.0])
            for j in range(steps + 1):
                if i != j:
                    poly = np.convolve(poly, np.array([1.0, -x[j]]))
                    poly /= x[i] - x[j]
            poly = poly[:-1] * np.arange(steps, 0, -1)
            coeff[i] = poly @ (x[-1] ** np.arange(steps - 1, -1, -1))
        coeff /= xx
        lu = spla.splu(self.func.matrix - coeff[-1] * sparse.eye(N, N))
        return lambda u: cast(NDArray[np.float64], lu.solve(u @ coeff[:-1]))


class IMEXTimestepper(metaclass=ABCMeta):
    t: float
    iter: int
    X: StateVector
    M: NDArray[np.float64]
    L: NDArray[np.float64]
    dt: Optional[float]

    @abstractmethod
    def _step(self, dt: float) -> NDArray[np.float64]:
        pass

    def __init__(self, eq_set: EquationSet):
        assert eq_set.F is not None
        self.t = 0
        self.iter = 0
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.dt = None

    def evolve(self, dt: float, time: float) -> None:
        while self.t < time - 1e-8:
            self.step(dt)

    def step(self, dt: float) -> None:
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):
    def _step(self, dt: float) -> NDArray[np.float64]:
        if dt != self.dt:
            LHS: Any = self.M + dt * self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec="NATURAL")
        self.dt = dt

        RHS: NDArray[np.float64] = cast(
            NDArray[np.float64], self.M @ self.X.data
        ) + dt * self.F(self.X)
        return cast(NDArray[np.float64], self.LU.solve(RHS))


class CNAB(IMEXTimestepper):
    def _step(self, dt: float) -> NDArray[np.float64]:
        LHS: Any
        RHS: NDArray[np.float64]
        if self.iter == 0:
            # Euler
            LHS = self.M + dt * self.L
            LU = spla.splu(LHS.tocsc(), permc_spec="NATURAL")

            self.FX = self.F(self.X)
            RHS = cast(NDArray[np.float64], self.M @ self.X.data) + dt * self.FX
            self.FX_old = self.FX
            return cast(NDArray[np.float64], LU.solve(RHS))
        else:
            if dt != self.dt:
                LHS = self.M + dt / 2 * self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec="NATURAL")
            self.dt = dt

            self.FX = self.F(self.X)
            RHS = (
                cast(NDArray[np.float64], self.M @ self.X.data)
                - cast(NDArray[np.float64], 0.5 * dt * self.L @ self.X.data)
                + cast(NDArray[np.float64], 3 / 2 * dt * self.FX)
                - cast(NDArray[np.float64], 1 / 2 * dt * self.FX_old)
            )
            self.FX_old = self.FX
            return cast(NDArray[np.float64], self.LU.solve(RHS))


class BDFExtrapolate(IMEXTimestepper):
    coeffs: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    xhist: list[NDArray[np.float64]]
    fhist: list[NDArray[np.float64]]

    def __init__(self, eq_set: EquationSet, steps: int):
        super().__init__(eq_set)
        self.steps = steps
        self.xhist = []
        self.fhist = []
        for s in range(1, steps + 1):
            if len(self.coeffs) < s:
                a = np.zeros((s + 1,))
                b = np.zeros((s,))
                for i in range(s + 1):
                    poly = np.array([1.0])
                    x1 = i / s
                    for j in range(s + 1):
                        if i != j:
                            x2 = j / s
                            poly = np.convolve(poly, np.array([1.0, -x2]))
                            poly /= x1 - x2
                        if i < s and j == s - 1:
                            b[i] = poly.sum()
                    a[i] = poly[:-1] @ np.arange(s, 0, -1)
                self.coeffs.append((a, b))

    def _step(self, dt: float) -> NDArray[np.float64]:
        self.xhist.append(self.X.data)
        self.fhist.append(self.F(self.X))
        steps = min(self.steps, len(self.xhist))
        solve = self._coeff(dt, steps)
        return solve(
            np.stack(self.xhist[-steps:], axis=1), np.stack(self.fhist[-steps:], axis=1)
        )

    @cache
    def _coeff(
        self, dt: float, steps: int
    ) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
        a, b = self.coeffs[steps - 1]
        a = cast(NDArray[np.float64], a / (steps * dt))
        lu = spla.splu(self.L + a[-1] * self.M)
        return lambda x, f: cast(
            NDArray[np.float64], lu.solve(f @ b - self.M @ (x @ a[:-1]))
        )
