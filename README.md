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
- CUDA Toolkit ≥ 11.5 (nvcc, cuBLAS; Nsight tools for profiling)

## Build and run

```bash
make                                   # builds build/bench (-arch=native)
./build/bench                          # all kernels, sizes 512..4096
./build/bench --kernel 01_naive --sizes 1024,2048 --reps 20
```

Override the target arch if `native` is unsupported: `make ARCH=sm_75` (T4).

The benchmark checks each kernel against cuBLAS (max abs error) and reports
ms, GFLOP/s and % of cuBLAS for square FP32 matrices.

### Google Colab

Open [`colab.ipynb`](colab.ipynb) in Colab, select a GPU runtime, and run all
cells. It clones this repo, builds, runs the benchmark, and has an optional
Nsight Compute cell.

## Status

| Kernel | Implemented | Verified on GPU |
|--------|-------------|-----------------|
| 1 Naive | yes | not yet |
| 2–8 | no | no |

No results yet; numbers will be added only from logged runs.

## License

TBD
