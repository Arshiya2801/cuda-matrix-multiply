cmake_minimum_required(VERSION 3.18)
project(cuda-matrix-multiply LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(matrix_mul
  src/main.cpp
  src/matrix.cpp
  src/matrix_mul.cu
  tests/test_matrix_mul.cpp
)

set_target_properties(matrix_mul PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(matrix_mul PRIVATE CUDA::cudart CUDA::cublas)

option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
  enable_testing()
  add_test(NAME MatrixMulTest COMMAND matrix_mul --test)
endif()
