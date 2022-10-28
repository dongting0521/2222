from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy import sparse  # type: ignore

from finite import DifferenceNonUniformGrid, DifferenceUniformGrid
from timesteppers import EquationSet, StateVector


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
        rho0: float,
        gammap0: float,
    ):
        (N,) = u.shape
        I = sparse.eye(N, N)  # noqa: E741
        Z = sparse.csr_matrix((N, N))
        self.X = StateVector([u, p])
        self.M = sparse.bmat([[rho0 * I, Z], [Z, I]])
        self.L = sparse.bmat([[Z, d.matrix], [gammap0 * d.matrix, Z]])
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
