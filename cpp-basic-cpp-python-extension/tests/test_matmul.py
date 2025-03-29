import numpy as np
import sys
from pathlib import Path

# we can't really import the library (shared object) from a script
# unless it's in the sys.path
SHARED_LIBRARY_DIR = Path(__file__).parents[1] / "build" / "lib"
sys.path.insert(0, str(SHARED_LIBRARY_DIR))

# now we can import our compiled library
import matrix_mul

# Define matrix dimensions
M, N, K = 32, 64, 32

# Create random matrices A[MxN] and B[NxK]
A = np.random.rand(M, N).astype(np.float32)
B = np.random.rand(N, K).astype(np.float32)
C = np.zeros((M, K), dtype=np.float32)  # Initialize C with zeros

# Call the compiled function
matrix_mul.matmul(A, B, C)

# Verify with NumPy
C_np = np.dot(A, B)

# Check if the results match
assert np.allclose(C, C_np), f"something went wrong, C and C_np are not equal"

print(f"Tests passed!")
