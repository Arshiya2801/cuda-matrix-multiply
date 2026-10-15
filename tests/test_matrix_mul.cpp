#include <cmath>
#include <iostream>
#include <vector>

#include "../include/matrix.hpp"

namespace {

constexpr float EPSILON = 1e-2f;

bool almost_equal(float a, float b) {
  return std::fabs(a - b) < EPSILON * (1.0f + std::fabs(a));
}
} // namespace

static bool verify(const char *label, const std::vector<float> &expected,
                   const std::vector<float> &actual) {
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!almost_equal(expected[i], actual[i])) {
      std::cerr << label << " failed at index " << i
                << ": expected=" << expected[i] << ", got=" << actual[i]
                << "\n";
      return false;
    }
  }
  std::cout << label << " passed\n";
  return true;
}

void test_matrix_mul() {
  constexpr int M = 64;
  constexpr int K = 128;
  constexpr int N = 96;

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C_cpu(M * N);
  std::vector<float> C_gpu(M * N);

  for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(i % 37) / 37.0f;
  for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(i % 41) / 41.0f;

  matrix_mul_cpu(A.data(), B.data(), C_cpu.data(), M, K, N);

  matrix_mul_gpu_naive(A.data(), B.data(), C_gpu.data(), M, K, N);
  verify("gpu_naive", C_cpu, C_gpu);

  matrix_mul_gpu_tiled(A.data(), B.data(), C_gpu.data(), M, K, N);
  verify("gpu_tiled", C_cpu, C_gpu);

  matrix_mul_gpu_vec_tiled(A.data(), B.data(), C_gpu.data(), M, K, N);
  verify("gpu_vec_tiled", C_cpu, C_gpu);

  matrix_mul_cublas(A.data(), B.data(), C_gpu.data(), M, K, N);
  verify("gpu_cublas", C_cpu, C_gpu);
}
