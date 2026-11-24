# cuda-matmul-kernels

Implementing different MatMul kernels on CUDA with profiling.

## Goal

Implement matrix multiplication (`C = A × B`, SGEMM) on CUDA in a series of
progressively optimized kernels, then profile each one to see *why* it is faster
or slower than the last. cuBLAS is the reference for both correctness and
performance.

## Planned kernels

| # | Kernel | Idea |
|---|--------|------|
| 1 | Naive | One thread per output element, all reads from global memory |
| 2 | Global memory coalescing | Remap thread indices so warps access contiguous memory |
| 3 | Shared-memory tiling | Stage tiles of A and B in shared memory |
| 4 | 1D block tiling | Each thread computes multiple outputs along one axis |
| 5 | 2D block tiling | Each thread computes a small register tile |
| 6 | Vectorized loads | `float4` global/shared accesses |
| 7 | Warp tiling | Organize work at warp level |
| 8 | Tensor cores (optional) | WMMA / MMA on supported GPUs |
| — | cuBLAS | Reference baseline |

## Profiling

- **Nsight Compute** (`ncu`): per-kernel metrics (achieved occupancy, memory
  throughput, bank conflicts, stall reasons, roofline).
- **Nsight Systems** (`nsys`): timeline view of kernel launches and memory transfers.
- **CUDA events**: wall-clock kernel timing inside the benchmark harness.
- Reported metrics: GFLOP/s, % of cuBLAS, achieved memory bandwidth.

## Layout

```
src/         CUDA kernels and host code
benchmarks/  benchmark driver, sizes sweep, result output
docs/        notes and profiling write-ups
```

## Requirements

- NVIDIA GPU with a recent driver
- CUDA Toolkit (nvcc, cuBLAS, Nsight tools)
- CMake ≥ 3.18 (build system to be added)

## Status

Project scaffolding only. No kernels or results yet; results will be added
only from logged runs.

## License

TBD
