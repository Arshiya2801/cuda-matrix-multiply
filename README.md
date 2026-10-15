# CUDA Matrix Multiply

A simple C++ and CUDA project that demonstrates how to optimize matrix multiplication ($C = A \times B$) on NVIDIA GPUs step-by-step.

It compares a basic CPU version against three progressively faster CUDA kernels and NVIDIA's official **cuBLAS** library.

---

## What's Inside

1. **CPU Reference**: Simple 3-loop matrix multiplication on a single CPU core.
2. **GPU Naive**: Basic CUDA kernel where each thread calculates one output number.
3. **GPU Tiled (Shared Memory)**: Loads $32 \times 32$ matrix blocks into fast on-chip shared memory to avoid slow memory reads.
4. **GPU Thread-Coarsened**: Each thread calculates 4 output values at once using local registers for extra speed.
5. **GPU cuBLAS**: NVIDIA's official hand-tuned library used as the gold standard.

---

## Performance Comparison (1024 x 1024 Matrix)

| Version | Time (approx.) | How it works |
| :--- | :--- | :--- |
| **CPU** | ~1400 ms | Baseline (slowest, single thread) |
| **GPU Naive** | ~3.1 ms | Fast parallel start, but waits on memory |
| **GPU Tiled** | ~1.0 ms | **~3x faster** using fast Shared Memory |
| **GPU Coarsened** | ~0.7 ms | **~4.5x faster** reusing values in Registers |
| **cuBLAS** | ~0.1 ms | Hardware-level peak speed |

---

## How to Build and Run

### Requirements
- Computer with an NVIDIA GPU and CUDA Toolkit
- CMake (3.18 or newer)
- C++ compiler

### 1. Build
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### 2. Run Benchmark
```bash
./matrix_mul
```

### 3. Run Tests
Verify that all GPU kernels calculate the correct results:
```bash
./matrix_mul --test
```

---

## Profile with NVIDIA Nsight Systems

To view a timeline of kernel execution and memory transfers:
```bash
./scripts/nsys_easy.sh ./build/matrix_mul
```

---

## Running on Google Colab (Free GPU)

If your laptop does not have an NVIDIA GPU, you can run this for free in Google Colab:

1. Open [Google Colab](https://colab.research.google.com/) and set runtime to **T4 GPU** (*Runtime $\to$ Change runtime type $\to$ T4 GPU*).
2. Run this in a code cell:
```bash
!git clone https://github.com/Arshiya2801/cuda-matrix-multiply.git
%cd cuda-matrix-multiply
!mkdir -p build && cd build && cmake .. && cmake --build .
!./build/matrix_mul --test
!./build/matrix_mul
```

---

## Project Structure

- `src/matrix_mul.cu` - All CUDA kernels (Naive, Tiled, Coarsened, cuBLAS)
- `src/matrix.cpp` - CPU reference implementation
- `src/main.cpp` - Timing and benchmark runner
- `tests/test_matrix_mul.cpp` - Correctness testing
- `include/matrix.hpp` - Function declarations
- `scripts/` - Easy run and profiling scripts