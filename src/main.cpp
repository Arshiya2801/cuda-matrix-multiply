#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include "../include/matrix.hpp"

using namespace std::chrono;

void test_matrix_mul();

int main(int argc, char *argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--test") == 0) {
    test_matrix_mul();
    return 0;
  }

  constexpr int M = 1024;
  constexpr int K = 1024;
  constexpr int N = 1024;

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C(M * N);

  for (int i = 0; i < M * K; ++i)
    A[i] = static_cast<float>(i % 100) / 100.0f;
  for (int i = 0; i < K * N; ++i)
    B[i] = static_cast<float>(i % 100) / 100.0f;

  cuda_warmup();

  auto bench = [&](const char *label, auto fn) {
    auto start = high_resolution_clock::now();
    fn(A.data(), B.data(), C.data(), M, K, N);
    auto end = high_resolution_clock::now();
    auto ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << label << ms << " ms\n";
  };

  bench("GPU naive            ", matrix_mul_gpu_naive);
  bench("GPU tiled (shared)   ", matrix_mul_gpu_tiled);
  bench("GPU vec tiled        ", matrix_mul_gpu_vec_tiled);
  bench("GPU cuBLAS           ", matrix_mul_cublas);
  bench("CPU                  ", matrix_mul_cpu);

  std::cout << "Matrix multiplication finished successfully.\n";
  return 0;
}
