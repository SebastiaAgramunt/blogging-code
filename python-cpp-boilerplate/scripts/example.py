import numpy as np
import package_example as pe  # this uses your __init__.py exports


def main():
    # Create a random matrix A of shape (M, N)
    M, N = 4, 3
    A = np.random.randn(M, N).astype(np.float32)

    print('A:')
    print(A)

    B = np.random.randn(N, M).astype(np.float32)

    print('\nB:')
    print(B)

    # Prepare output matrix C
    C = np.zeros((M, M), dtype=np.float32)

    # multiply A and B using the C++ extension
    pe.matmul(A, B, C)

    print('\nA * B:')
    print(C)

    # Verify correctness using pure NumPy
    C_np = A @ B

    print('\nNumPy result:')
    print(C_np)

    print('\nDifference (should be near zero):')
    print(C - C_np)


if __name__ == '__main__':
    main()
