#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "matmul.h"

namespace py = pybind11;

void matmul_py(py::array_t<float> A, py::array_t<float> B, py::array_t<float> C) {
    auto bufA = A.request(), bufB = B.request(), bufC = C.request();

    if (bufA.ndim != 2 || bufB.ndim != 2 || bufC.ndim != 2) {
        throw std::runtime_error("All matrices must be 2D");
    }

    size_t M = bufA.shape[0];
    size_t N = bufA.shape[1];
    size_t K = bufB.shape[1];

    if (bufB.shape[0] != N || bufC.shape[0] != M || bufC.shape[1] != K) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication");
    }

    float* ptrA = static_cast<float*>(bufA.ptr);
    float* ptrB = static_cast<float*>(bufB.ptr);
    float* ptrC = static_cast<float*>(bufC.ptr);

    matmul(ptrA, ptrB, ptrC, M, N, K);  // same call as before
}


PYBIND11_MODULE(matrix_mul, m) {
    m.def("matmul", &matmul_py, "Matrix multiplication function");
}
