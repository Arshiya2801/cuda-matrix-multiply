#include "../include/matrix.hpp"

void matrix_mul_cpu(const float *A, const float *B, float *C, int M, int K,
                    int N) {
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
      }
      C[row * N + col] = sum;
    }
  }
}
