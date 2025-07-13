cmake_minimum_required(VERSION 3.10)
project(GPUTools LANGUAGES CXX CUDA)

# default exec names
set(GPU_INFO_OUT_NAME    "gpu_info"
    CACHE STRING "Name of the gpu_info executable")
set(GPU_ALLOC_OUT_NAME   "gpu_allocate"
    CACHE STRING "Name of the gpu_allocate executable")

# Restore old FindCUDA behavior if needed
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# Language standards
set(CMAKE_CXX_STANDARD      14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75 86 CACHE STRING
    "List of CUDA architectures to build for (e.g. 61;70;75;86)")

# Find CUDA (for older CMake) or you can use find_package(CUDAToolkit) in 3.17+
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(gpu_info
  src/gpu_info.cu
)
target_link_libraries(gpu_info
  PRIVATE ${CUDA_CUDART_LIBRARY}
)
set_target_properties(gpu_info
  PROPERTIES OUTPUT_NAME ${GPU_INFO_OUT_NAME}
)

add_executable(gpu_allocate
  src/gpu_allocate.cu
)
target_link_libraries(gpu_allocate
  PRIVATE ${CUDA_CUDART_LIBRARY}
)
set_target_properties(gpu_allocate
  PROPERTIES OUTPUT_NAME ${GPU_ALLOC_OUT_NAME}
)

# (Optional) If you want to give a different on-disk name:
# set(EXE_NAME alloc_mem)
# set_target_properties(allocate_gpu_memory PROPERTIES OUTPUT_NAME ${EXE_NAME})

# Installation
install(TARGETS
  gpu_info
  gpu_allocate
  RUNTIME DESTINATION bin
)