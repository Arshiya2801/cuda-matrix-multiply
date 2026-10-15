#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>


#include "../include/matrix.hpp"

constexpr int TILE_SIZE = 32;

void cuda_warmup() { cudaFree(0); }

// --- 1. Naive: one thread per output element ---

__global__ void matmul_naive_kernel(const float *A, const float *B, float *C,
                                    int M, int K, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// --- 2. Tiled with shared memory ---

__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C,
                                    int M, int K, int N) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;
  int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; ++t) {
    int a_col = t * TILE_SIZE + threadIdx.x;
    int b_row = t * TILE_SIZE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// --- 3. Vectorized tiled (float4 loads for B) ---
// Each thread computes a TM x 1 column of output elements (thread coarsening).
// B is loaded via float4 into shared memory for coalesced 128-bit reads.

constexpr int VEC_TILE = 32;
constexpr int TM = 4;

__global__ void matmul_vec_tiled_kernel(const float *A, const float *B,
                                        float *C, int M, int K, int N) {
  __shared__ float As[VEC_TILE][VEC_TILE];
  __shared__ float Bs[VEC_TILE][VEC_TILE];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row_base = by * VEC_TILE + ty * TM;
  int col = bx * VEC_TILE + tx;

  float sums[TM] = {0.0f};
  int num_tiles = (K + VEC_TILE - 1) / VEC_TILE;

  for (int t = 0; t < num_tiles; ++t) {
    int tile_offset = t * VEC_TILE;

#pragma unroll
    for (int m = 0; m < TM; ++m) {
      int r = row_base + m;
      int a_col = tile_offset + tx;
      As[ty * TM + m][tx] = (r < M && a_col < K) ? A[r * K + a_col] : 0.0f;
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
      int b_row = tile_offset + ty * TM + m;
      Bs[ty * TM + m][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < VEC_TILE; ++k) {
      float b_val = Bs[k][tx];
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        sums[m] += As[ty * TM + m][k] * b_val;
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int r = row_base + m;
    if (r < M && col < N) {
      C[r * N + col] = sums[m];
    }
  }
}

// --- Host wrappers ---

void matrix_mul_gpu_naive(const float *A, const float *B, float *C, int M,
                          int K, int N) {
  size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc(&d_A, bytes_A);
  cudaMalloc(&d_B, bytes_B);
  cudaMalloc(&d_C, bytes_C);

  cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((N + 15) / 16, (M + 15) / 16);
  matmul_naive_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

  cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void matrix_mul_gpu_tiled(const float *A, const float *B, float *C, int M,
                          int K, int N) {
  size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc(&d_A, bytes_A);
  cudaMalloc(&d_B, bytes_B);
  cudaMalloc(&d_C, bytes_C);

  cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);

  dim3 threads(TILE_SIZE, TILE_SIZE);
  dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  matmul_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

  cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void matrix_mul_gpu_vec_tiled(const float *A, const float *B, float *C, int M,
                              int K, int N) {
  size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc(&d_A, bytes_A);
  cudaMalloc(&d_B, bytes_B);
  cudaMalloc(&d_C, bytes_C);

  cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);

  // VEC_TILE x VEC_TILE tile, but only VEC_TILE/TM rows of threads
  dim3 threads(VEC_TILE, VEC_TILE / TM);
  dim3 blocks((N + VEC_TILE - 1) / VEC_TILE, (M + VEC_TILE - 1) / VEC_TILE);
  matmul_vec_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

  cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// --- 4. cuBLAS ---

void matrix_mul_cublas(const float *A, const float *B, float *C, int M, int K,
                       int N) {
  size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc(&d_A, bytes_A);
  cudaMalloc(&d_B, bytes_B);
  cudaMalloc(&d_C, bytes_C);

  cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  // cuBLAS expects column-major. For row-major C = A*B we compute
  // C^T = B^T * A^T, which cuBLAS sees as column-major naturally.
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
              &beta, d_C, N);

  cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
