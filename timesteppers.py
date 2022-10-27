from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import cache
from typing import Any, Callable, Generic, Optional, TypeVar, Union, cast

import numpy as np
import scipy.sparse.linalg as spla  # type: ignore
from numpy.typing import NDArray
from scipy import sparse  # type: ignore

from finite import DifferenceNonUniformGrid, DifferenceUniformGrid

T = TypeVar("T")


class Timestepper(Generic[T], metaclass=ABCMeta):
    t: float
    iter: int
    u: NDArray[np.float64]
    func: T
    dt: Optional[float]

    @abstractmethod
    def _step(self, dt: float) -> NDArray[np.float64]:
        pass

    def __init__(self, u: NDArray[np.float64], f: T):
        self.t = 0
        self.iter = 0
        self.u = u
        self.func = f
        self.dt = None

    def step(self, dt: float) -> None:
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1

    def evolve(self, dt: float, time: float) -> None:
        while self.t < time - 1e-8:
            self.step(dt)


class ForwardEuler(Timestepper[Callable[[NDArray[np.float64]], NDArray[np.float64]]]):
    def _step(self, dt: float) -> NDArray[np.float64]:
        return self.u + dt * self.func(self.u)


class LaxFriedrichs(Timestepper[Callable[[NDArray[np.float64]], NDArray[np.float64]]]):
    def __init__(
        self,
        u: NDArray[np.float64],
        f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1 / 2, 1 / 2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1 / 2
        A[-1, 0] = 1 / 2
        self.A = A

    def _step(self, dt: float) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self.A @ self.u + dt * self.func(self.u))


class Leapfrog(Timestepper[Callable[[NDArray[np.float64]], NDArray[np.float64]]]):
    def _step(self, dt: float) -> NDArray[np.float64]:
        if self.iter == 0:
            self.u_old = np.copy(self.u)  # type: ignore
            return self.u + dt * self.func(self.u)
        else:
            u_temp: NDArray[np.float64] = self.u_old + 2 * dt * self.func(self.u)
            self.u_old = np.copy(self.u)  # type: ignore
            return u_temp


class LaxWendroff(Timestepper[Callable[[NDArray[np.float64]], NDArray[np.float64]]]):
    def __init__(
        self,
        u: NDArray[np.float64],
        func1: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        func2: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ):
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = func1
        self.f2 = func2

    def _step(self, dt: float) -> NDArray[np.float64]:
        return cast(
            NDArray[np.float64],
            cast(NDArray[np.float64], self.u + dt * self.f1(self.u))
            + cast(NDArray[np.float64], dt**2 / 2 * self.f2(self.u)),
        )


class Multistage(Timestepper[Callable[[NDArray[np.float64]], NDArray[np.float64]]]):
    stages: int
    a: NDArray[np.float64]
    b: NDArray[np.float64]

    def __init__(
        self,
        u: NDArray[np.float64],
        f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        stages: int,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
    ):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b

    def _step(self, dt: float) -> NDArray[np.float64]:
        k = np.zeros((self.u.size, self.stages))
        for i in range(self.stages):
            k[:, i] = self.func(self.u + dt * (k @ self.a[i, :]))
        return self.u + dt * (k @ self.b)


class AdamsBashforth(Timestepper[Callable[[NDArray[np.float64]], NDArray[np.float64]]]):
    coeffs: list[NDArray[np.float64]] = []
    steps: int
    uhist: list[NDArray[np.float64]]
    fhist: list[NDArray[np.float64]]

    def __init__(
        self,
        u: NDArray[np.float64],
        f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        steps: int,
        _: Any,
    ):
        super().__init__(u, f)
        self.steps = steps
        self.uhist = []
        self.fhist = []
        for s in range(1, steps + 1):
            if len(self.coeffs) < s:
                coeff = np.zeros((s,))
                for i in range(s):
                    poly = np.array([1.0])
                    x1 = i / s
                    for j in range(s):
                        if i != j:
                            x2 = j / s
                            poly = np.convolve(poly, np.array([1.0, -x2]))
                            poly /= x1 - x2
                    poly /= np.arange(s, 0, -1)
                    coeff[i] = poly @ (1 - ((s - 1) / s) ** np.arange(s, 0, -1)) * s
                self.coeffs.append(coeff)

    def _step(self, dt: float) -> NDArray[np.float64]:
        self.uhist.append(self.u)
        self.fhist.append(self.func(self.u))
        steps = min(self.steps, len(self.uhist))
        return self.uhist[-1] + (
            dt
            * cast(
                NDArray[np.float64],
                np.stack(self.fhist[-steps:], axis=1) @ self.coeffs[steps - 1],
            )
        )


class BackwardEuler(
    Timestepper[Union[DifferenceUniformGrid, DifferenceNonUniformGrid]]
):
    def __init__(
        self,
        u: NDArray[np.float64],
        L: Union[DifferenceUniformGrid, DifferenceNonUniformGrid],
    ):
        super().__init__(u, L)
        N = len(u)
        self.I = sparse.eye(N, N)  # noqa: E741

    def _step(self, dt: float) -> NDArray[np.float64]:
        if dt != self.dt:
            self.LHS = self.I - dt * self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec="NATURAL")
        self.dt = dt
        return cast(NDArray[np.float64], self.LU.solve(self.u))


class CrankNicolson(
    Timestepper[Union[DifferenceUniformGrid, DifferenceNonUniformGrid]]
):
    def __init__(
        self,
        u: NDArray[np.float64],
        L_op: Union[DifferenceUniformGrid, DifferenceNonUniformGrid],
    ):
        super().__init__(u, L_op)
        N = len(u)
        self.I = sparse.eye(N, N)  # noqa: E741

    def _step(self, dt: float) -> NDArray[np.float64]:
        if dt != self.dt:
            self.LHS = self.I - dt / 2 * self.func.matrix
            self.RHS = self.I + dt / 2 * self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec="NATURAL")
        self.dt = dt
        return cast(NDArray[np.float64], self.LU.solve(self.RHS @ self.u))


class BackwardDifferentiationFormula(
    Timestepper[Union[DifferenceUniformGrid, DifferenceNonUniformGrid]]
):
    steps: int
    thist: list[float]
    uhist: list[NDArray[np.float64]]

    def __init__(
        self,
        u: NDArray[np.float64],
        L_op: Union[DifferenceUniformGrid, DifferenceNonUniformGrid],
        steps: int,
    ):
        super().__init__(u, L_op)
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


class StateVector:
    N: int
    data: NDArray[np.float64]
    variables: list[NDArray[np.float64]]

    def __init__(self, variables: list[NDArray[np.float64]]):
        var0 = variables[0]
        self.N = len(var0)
        size = self.N * len(variables)
        self.data = np.zeros(size)
        self.variables = variables
        self.gather()

    def gather(self) -> None:
        for i, var in enumerate(self.variables):
            np.copyto(self.data[i * self.N : (i + 1) * self.N], var)  # type: ignore

    def scatter(self) -> None:
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[i * self.N : (i + 1) * self.N])  # type: ignore


class EquationSet(metaclass=ABCMeta):
    X: StateVector
    M: NDArray[np.float64]
    L: NDArray[np.float64]
    F: Optional[Callable[[StateVector], NDArray[np.float64]]]


class IMEXTimestepper:
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
