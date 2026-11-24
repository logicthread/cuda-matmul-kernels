#pragma once

#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t st_ = (call);                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__,   \
                    __LINE__, (int)st_);                                      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Signature shared by every kernel launcher.
// Row-major: C (M x N) = alpha * A (M x K) * B (K x N) + beta * C
using SgemmLauncher = void (*)(int M, int N, int K, float alpha,
                               const float *A, const float *B, float beta,
                               float *C);
