import numpy as np
from ._core import matmul  # <-- relative import from C++ core


def matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute y = A @ x using the fast C++ matmul."""
    M, N = A.shape
    assert x.shape == (N,), 'x must have shape (N,)'

    # reshape x into 2D for C++ (N x 1)
    x2 = x.reshape(N, 1).astype(A.dtype, copy=False)

    # allocate output (M x 1)
    y2 = np.zeros((M, 1), dtype=A.dtype)

    # call your C++ kernel
    matmul(A, x2, y2)

    # return as 1D vector
    return y2.ravel()
