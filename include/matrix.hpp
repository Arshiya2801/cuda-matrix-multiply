#pragma once

void cuda_warmup();

// C = A * B  (A is MxK, B is KxN, C is MxN, row-major)
void matrix_mul_cpu(const float *A, const float *B, float *C, int M, int K,
                    int N);

void matrix_mul_gpu_naive(const float *A, const float *B, float *C, int M,
                          int K, int N);

void matrix_mul_gpu_tiled(const float *A, const float *B, float *C, int M,
                          int K, int N);

void matrix_mul_gpu_vec_tiled(const float *A, const float *B, float *C, int M,
                              int K, int N);

void matrix_mul_cublas(const float *A, const float *B, float *C, int M, int K,
                       int N);
