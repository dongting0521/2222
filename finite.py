from typing import Literal, Sequence, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse  # type: ignore
from scipy.special import factorial  # type: ignore

from farray import apply_matrix, reshape_vector


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

    def dx_array(self, j: NDArray[np.int64]) -> NDArray[np.float64]:
        shape = (self.N, len(j))
        dx = np.zeros(shape)

        jmin = -j.min()
        jmax = j.max()

        values_padded = np.zeros(self.N + jmin + jmax)
        if jmin > 0:
            values_padded[:jmin] = self.values[-jmin:] - self.length
        if jmax > 0:
            values_padded[jmin:-jmax] = self.values
            values_padded[-jmax:] = self.length + self.values[:jmax]
        else:
            values_padded[jmin:] = self.values

        for i in range(self.N):
            dx[i, :] = values_padded[jmin + i + j] - values_padded[jmin + i]

        return dx


class UniformNonPeriodicGrid:
    def __init__(self, N: int, interval: tuple[float, float]):
        """Non-uniform grid; no grid points at the endpoints of the interval"""
        self.start = interval[0]
        self.end = interval[1]
        self.dx = (self.end - self.start) / (N - 1)
        self.N = N
        self.values = np.linspace(self.start, self.end, N, endpoint=True)


class Domain:
    def __init__(
        self,
        grids: Sequence[
            Union[UniformPeriodicGrid, NonUniformPeriodicGrid, UniformNonPeriodicGrid]
        ],
    ):
        self.dimension = len(grids)
        self.grids = grids
        shape = []
        for grid in self.grids:
            shape.append(grid.N)
        self.shape = shape

    def values(self) -> list[NDArray[np.float64]]:
        v = []
        for i, grid in enumerate(self.grids):
            grid_v = grid.values
            grid_v = reshape_vector(grid_v, self.dimension, i)
            v.append(grid_v)
        return v

    def plotting_arrays(self) -> list[NDArray[np.int64]]:
        v: list[NDArray[np.int64]] = []
        expanded_shape = np.array(self.shape, dtype=np.int64)
        expanded_shape += 1
        for i, grid in enumerate(self.grids):
            grid_v = grid.values
            grid_v = np.concatenate((grid_v, [grid.length]))  # type: ignore
            grid_v = reshape_vector(grid_v, self.dimension, i)
            grid_v = np.broadcast_to(grid_v, expanded_shape)  # type: ignore
            v.append(cast(NDArray[np.int64], grid_v))
        return v


class Difference:
    axis: int
    matrix: NDArray[np.float64]

    def __matmul__(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        return apply_matrix(self.matrix, other, axis=self.axis)


class DifferenceUniformGrid(Difference):
    def __init__(
        self,
        derivative_order: int,
        convergence_order: int,
        grid: Union[UniformPeriodicGrid, UniformNonPeriodicGrid],
        axis: int = 0,
        stencil_type: Literal["centered"] = "centered",
    ):
        if stencil_type == "centered" and convergence_order % 2 != 0:
            raise ValueError("Centered finite difference has even convergence order")

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self._stencil_shape(stencil_type)
        self._make_stencil(grid)
        self._build_matrix(grid)

    def _stencil_shape(self, stencil_type: Literal["centered"]) -> None:
        dof = self.derivative_order + self.convergence_order  # 插值点的个数

        if stencil_type == "centered":
            # cancellation if derivative order is even
            dof = dof - (1 - dof % 2)
            j = np.arange(dof) - dof // 2

        self.dof = dof
        self.j = j

    def _make_stencil(
        self, grid: Union[UniformPeriodicGrid, UniformNonPeriodicGrid], shift: int = 0
    ) -> None:

        # assume constant grid spacing
        self.dx = grid.dx
        i = np.arange(self.dof)[:, None]
        j = self.j[None, :] + shift
        S = 1 / factorial(i) * (j * self.dx) ** i

        b = np.zeros(self.dof)
        b[self.derivative_order] = 1.0

        self.stencil = np.linalg.solve(S, b)  # type: ignore

    def _build_matrix(
        self, grid: Union[UniformPeriodicGrid, UniformNonPeriodicGrid]
    ) -> None:
        shape = [grid.N] * 2
        matrix = sparse.diags(self.stencil, self.j, shape=shape)
        matrix = matrix.tolil()
        jmin = -self.j.min()
        if jmin > 0:
            for i in range(jmin):
                if isinstance(grid, UniformNonPeriodicGrid):
                    self._make_stencil(grid, jmin - i)
                    matrix[i, : self.j.size] = self.stencil
                else:
                    matrix[i, -jmin + i :] = self.stencil[: jmin - i]

        jmax = self.j.max()
        if jmax > 0:
            for i in range(jmax):
                if isinstance(grid, UniformNonPeriodicGrid):
                    self._make_stencil(grid, -i - 1)
                    matrix[-jmax + i, -self.j.size :] = self.stencil
                else:
                    matrix[-jmax + i, : i + 1] = self.stencil[-i - 1 :]
        self.matrix = matrix.tocsc()


class DifferenceNonUniformGrid(Difference):
    def __init__(
        self,
        derivative_order: int,
        convergence_order: int,
        grid: NonUniformPeriodicGrid,
        axis: int = 0,
        stencil_type: Literal["centered"] = "centered",
    ):
        if (derivative_order + convergence_order) % 2 == 0:
            raise ValueError(
                "The derivative plus convergence order must be odd"
                " for centered finite difference"
            )

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self._stencil_shape(stencil_type)
        self._make_stencil(grid)
        self._build_matrix(grid)

    def _stencil_shape(self, stencil_type: Literal["centered"]) -> None:
        dof = self.derivative_order + self.convergence_order
        j = np.arange(dof) - dof // 2
        self.dof = dof
        self.j = j

    def _make_stencil(self, grid: NonUniformPeriodicGrid) -> None:
        self.dx = grid.dx_array(self.j)

        i = np.arange(self.dof)[None, :, None]
        dx = self.dx[:, None, :]
        S = 1 / factorial(i) * (dx) ** i

        b = np.zeros((grid.N, self.dof))
        b[:, self.derivative_order] = 1.0

        self.stencil = np.linalg.solve(S, b)  # type: ignore

    def _build_matrix(self, grid: NonUniformPeriodicGrid) -> None:
        shape = [grid.N] * 2
        diags = []
        for i, jj in enumerate(self.j):
            if jj < 0:
                s = slice(-jj, None, None)
            else:
                s = slice(None, None, None)
            diags.append(self.stencil[s, i])
        matrix = sparse.diags(diags, self.j, shape=shape)

        matrix = matrix.tolil()
        jmin = -self.j.min()
        if jmin > 0:
            for i in range(jmin):
                matrix[i, -jmin + i :] = self.stencil[i, : jmin - i]

        jmax = self.j.max()
        if jmax > 0:
            for i in range(jmax):
                matrix[-jmax + i, : i + 1] = self.stencil[-jmax + i, -i - 1 :]

        self.matrix = matrix.tocsc()


class ForwardFiniteDifference(Difference):
    def __init__(self, grid: UniformPeriodicGrid, axis: int = 0):
        self.axis = axis
        h = grid.dx
        N = grid.N
        j = [0, 1]
        diags = np.array([-1 / h, 1 / h])
        matrix = sparse.diags(diags, offsets=j, shape=[N, N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1 / h
        self.matrix = matrix


class CenteredFiniteDifference(Difference):
    def __init__(self, grid: UniformPeriodicGrid, axis: int = 0):
        self.axis = axis
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([-1 / (2 * h), 0, 1 / (2 * h)])
        matrix = sparse.diags(diags, offsets=j, shape=[N, N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1 / (2 * h)
        matrix[0, -1] = -1 / (2 * h)
        self.matrix = matrix


class CenteredFiniteSecondDifference(Difference):
    def __init__(self, grid: UniformPeriodicGrid, axis: int = 0):
        self.axis = axis
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([1 / h**2, -2 / h**2, 1 / h**2])
        matrix = sparse.diags(diags, offsets=j, shape=[N, N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1 / h**2
        matrix[0, -1] = 1 / h**2
        self.matrix = matrix


class CenteredFiniteDifference4(Difference):
    def __init__(self, grid: UniformPeriodicGrid, axis: int = 0):
        self.axis = axis
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
