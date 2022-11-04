# Helper functions from K. J. Burns
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse  # type: ignore


def apply_matrix(
    matrix: NDArray[np.float64], array: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    """Contract any direction of a multidimensional array with a matrix."""
    dim = len(array.shape)
    # Build Einstein signatures
    mat_sig = [dim, axis]
    arr_sig = list(range(dim))
    out_sig = list(range(dim))
    out_sig[axis] = dim
    # Handle sparse matrices
    if sparse.isspmatrix(matrix):
        matrix = matrix.toarray()  # type: ignore
    return cast(
        NDArray[np.float64],
        np.einsum(matrix, mat_sig, array, arr_sig, out_sig),  # type: ignore
    )


def reshape_vector(
    data: NDArray[np.float64], dim: int = 2, axis: int = -1
) -> NDArray[np.float64]:
    """Reshape 1-dim array as a multidimensional vector."""
    # Build multidimensional shape
    shape = [1] * dim
    shape[axis] = data.size
    return data.reshape(shape)


def axindex(axis: int, index: slice) -> tuple[slice, ...]:
    """Index array along specified axis."""
    if axis < 0:
        raise ValueError("`axis` must be positive")
    # Add empty slices for leading axes
    return (slice(None),) * axis + (index,)


def axslice(
    axis: int, start: int, stop: int, step: Optional[int] = None
) -> tuple[slice, ...]:
    """Slice array along a specified axis."""
    return axindex(axis, slice(start, stop, step))
