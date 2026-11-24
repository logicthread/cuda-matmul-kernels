#pragma once

#include "common.cuh"

// Kernel 1: naive SGEMM.
//
// One thread computes one element of C by walking a row of A and a column of B
// straight from global memory. No shared memory, no tiling, no data reuse.
//
// Note on thread mapping: `row` follows threadIdx.x, so the 32 threads of a warp
// work on 32 *different rows* of A and C. Loads of A (stride K) and stores to C
// (stride N) are therefore uncoalesced. That is deliberate: it is the baseline,
// and kernel 2 fixes it by remapping threads so a warp spans consecutive columns.
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
            acc += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

inline void launch_sgemm_naive(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
}
