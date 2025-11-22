import numpy as np
import package_example as pe


def test_matmul_matches_numpy():
    M, N, K = 32, 64, 32

    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(N, K).astype(np.float32)
    C = np.zeros((M, K), dtype=np.float32)

    pe.matmul(A, B, C)

    C_np = A @ B

    assert np.allclose(C, C_np), 'C and C_np are not equal'


def test_matvec_random():
    rng = np.random.default_rng(42)

    # Dimensions
    M, N = 10, 7

    # Random matrix A and vector x
    A = rng.standard_normal((M, N), dtype=np.float32)
    x = rng.standard_normal(N).astype(np.float32)

    # C++ version
    y = pe.matvec(A, x)

    # NumPy ground truth
    y_np = A @ x

    # Shape checks
    assert y.shape == (M,)
    assert y_np.shape == (M,)

    # Numerical correctness
    assert np.allclose(y, y_np, rtol=1e-5, atol=1e-6)
