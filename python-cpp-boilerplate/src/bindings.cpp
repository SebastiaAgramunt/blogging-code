#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "matmul.h"

namespace py = pybind11;

void matmul_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B,
    py::array_t<float, py::array::c_style | py::array::forcecast> C
) {
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    if (bufA.ndim != 2 || bufB.ndim != 2 || bufC.ndim != 2) {
        throw std::runtime_error("All matrices must be 2D");
    }

    std::size_t M = static_cast<std::size_t>(bufA.shape[0]);
    std::size_t N = static_cast<std::size_t>(bufA.shape[1]);
    std::size_t N2 = static_cast<std::size_t>(bufB.shape[0]);
    std::size_t K = static_cast<std::size_t>(bufB.shape[1]);

    if (N2 != N) {
        throw std::runtime_error("Inner dimensions of A and B must match (A[MxN], B[NxK])");
    }
    if (static_cast<std::size_t>(bufC.shape[0]) != M ||
        static_cast<std::size_t>(bufC.shape[1]) != K) {
        throw std::runtime_error("C must have shape (M, K)");
    }

    auto* ptrA = static_cast<float*>(bufA.ptr);
    auto* ptrB = static_cast<float*>(bufB.ptr);
    auto* ptrC = static_cast<float*>(bufC.ptr);

    matmul(ptrA, ptrB, ptrC, M, N, K);
}

PYBIND11_MODULE(_core, m) {
    m.def("matmul", &matmul_py, "Matrix multiplication function");
}