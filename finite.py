from __future__ import annotations

import typing
from typing import Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse  # type: ignore
from scipy.special import factorial  # type: ignore


class UniformPeriodicGrid:
    def __init__(self, N: int, length: float):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:
    def __init__(self, values: NDArray[np.float64], length: float):
        self.values = values
        self.length = length
        self.N = len(values)


class DifferenceUniformGrid:
    def __init__(
        self,
        derivative_order: Union[str, float],
        convergence_order: Union[str, float],
        grid: UniformPeriodicGrid,
        stencil_type: str = "centered",
    ):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.matrix = DifferenceNonUniformGrid(
            derivative_order,
            convergence_order,
            NonUniformPeriodicGrid(grid.values, grid.length),
            stencil_type,
        ).matrix

    def __matmul__(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        return typing.cast(NDArray[np.float64], self.matrix @ other)


class DifferenceNonUniformGrid:
    def __init__(
        self,
        _derivative_order: Union[str, float],
        _convergence_order: Union[str, float],
        grid: NonUniformPeriodicGrid,
        stencil_type: str = "centered",
    ):
        derivative_order = int(float(str(_derivative_order).replace(" ", "")))
        convergence_order = int(float(str(_convergence_order).replace(" ", "")))
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        num_points = derivative_order + convergence_order - 1
        matrix = sparse.dok_matrix((grid.N, grid.N))
        print(num_points)
        for i in range(*grid.values.shape):  # 以这个点为中心
            xi = grid.values[i]
            rng = range(i - num_points // 2, i + num_points // 2 + 1)
            for j in rng:  # 遍历所有的插值点
                poly = np.zeros((derivative_order + 1,))
                poly[0] = 1
                xj = (j // grid.N) * grid.length + grid.values[j % grid.N] - xi
                for k in rng:  # 遍历所有的Lagrange单项式
                    if k != j:
                        xk = (k // grid.N) * grid.length + grid.values[k % grid.N] - xi
                        poly[1:] = (xk * poly[1:] - poly[:-1]) / (xk - xj)
                        poly[0] *= xk / (xk - xj)
                matrix[i, j % grid.N] += poly[-1]
        matrix = matrix.tocsr()
        matrix *= factorial(derivative_order)
        self.matrix = matrix

    def __matmul__(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        return typing.cast(NDArray[np.float64], self.matrix @ other)


class Difference:
    matrix: NDArray[np.float64]

    def __matmul__(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self.matrix @ other)


class CenteredFiniteDifference(Difference):
    def __init__(self, grid: UniformPeriodicGrid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([-1 / (2 * h), 0, 1 / (2 * h)])
        matrix = sparse.diags(diags, offsets=j, shape=[N, N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1 / (2 * h)
        matrix[0, -1] = -1 / (2 * h)
        self.matrix = matrix


class CenteredFiniteDifference4(Difference):
    def __init__(self, grid: UniformPeriodicGrid):
        h = grid.dx
        N = grid.N
        j = [-2, -1, 0, 1, 2]
        diags = np.array([1, -8, 0, 8, -1]) / (12 * h)
        matrix = sparse.diags(diags, offsets=j, shape=[N, N])
        matrix = matrix.tocsr()
        matrix[-2, 0] = -1 / (12 * h)
        matrix[-1, 0] = 8 / (12 * h)
        matrix[-1, 1] = -1 / (12 * h)

        matrix[0, -2] = 1 / (12 * h)
        matrix[0, -1] = -8 / (12 * h)
        matrix[1, -1] = 1 / (12 * h)
        self.matrix = matrix
