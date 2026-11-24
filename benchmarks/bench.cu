// Benchmark driver: runs registered SGEMM kernels against cuBLAS.
//
// Usage: ./build/bench [--kernel <name|all>] [--sizes 512,1024,2048] [--reps 10]
//
// For each square size N (M = N = K): checks correctness against cuBLAS
// (max abs error), times with CUDA events, reports ms, GFLOP/s, % of cuBLAS.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../src/common.cuh"
#include "../src/01_naive.cuh"

struct KernelEntry {
    const char *name;
    SgemmLauncher launch;
};

// Register new kernels here.
static const KernelEntry KERNELS[] = {
    {"01_naive", launch_sgemm_naive},
};

template <typename F>
static float time_ms(F &&fn, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

// Row-major C = alpha*A*B + beta*C via column-major cuBLAS: C^T = B^T * A^T.
static void cublas_sgemm(cublasHandle_t h, int M, int N, int K, float alpha,
                         const float *A, const float *B, float beta, float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                             A, K, &beta, C, N));
}

static std::vector<int> parse_sizes(const char *s) {
    std::vector<int> out;
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t comma = str.find(',', pos);
        if (comma == std::string::npos) comma = str.size();
        out.push_back(std::stoi(str.substr(pos, comma - pos)));
        pos = comma + 1;
    }
    return out;
}

int main(int argc, char **argv) {
    std::string kernel_sel = "all";
    std::vector<int> sizes = {512, 1024, 2048, 4096};
    int reps = 10;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--kernel") && i + 1 < argc) kernel_sel = argv[++i];
        else if (!strcmp(argv[i], "--sizes") && i + 1 < argc) sizes = parse_sizes(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--kernel <name|all>] [--sizes a,b,c] [--reps n]\n", argv[0]);
            return 1;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d), %d SMs\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("reps=%d, FP32, row-major, M=N=K\n\n", reps);
    printf("%-12s %6s %10s %10s %9s %12s\n", "kernel", "size", "ms", "GFLOP/s",
           "%cuBLAS", "max_abs_err");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int n : sizes) {
        const size_t elems = (size_t)n * n;
        const double flops = 2.0 * n * (double)n * n;

        std::vector<float> hA(elems), hB(elems), hC(elems), hRef(elems);
        for (auto &v : hA) v = dist(rng);
        for (auto &v : hB) v = dist(rng);

        float *dA, *dB, *dC, *dRef;
        CUDA_CHECK(cudaMalloc(&dA, elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dRef, elems * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
        // beta = 0 still computes beta*C, so C must not hold NaNs.
        CUDA_CHECK(cudaMemset(dC, 0, elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(dRef, 0, elems * sizeof(float)));

        // Reference result + cuBLAS timing.
        cublas_sgemm(handle, n, n, n, alpha, dA, dB, beta, dRef);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, elems * sizeof(float), cudaMemcpyDeviceToHost));
        const float cublas_ms = time_ms(
            [&] { cublas_sgemm(handle, n, n, n, alpha, dA, dB, beta, dRef); }, 2, reps);
        const double cublas_gflops = flops / (cublas_ms * 1e6);
        printf("%-12s %6d %10.3f %10.1f %8.1f%% %12s\n", "cublas", n, cublas_ms,
               cublas_gflops, 100.0, "-");

        for (const auto &k : KERNELS) {
            if (kernel_sel != "all" && kernel_sel != k.name) continue;

            CUDA_CHECK(cudaMemset(dC, 0, elems * sizeof(float)));
            k.launch(n, n, n, alpha, dA, dB, beta, dC);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, elems * sizeof(float), cudaMemcpyDeviceToHost));

            float max_err = 0.0f;
            for (size_t i = 0; i < elems; ++i)
                max_err = std::max(max_err, std::fabs(hC[i] - hRef[i]));

            const float ms = time_ms(
                [&] { k.launch(n, n, n, alpha, dA, dB, beta, dC); }, 1, reps);
            const double gflops = flops / (ms * 1e6);
            printf("%-12s %6d %10.3f %10.1f %8.1f%% %12.2e%s\n", k.name, n, ms, gflops,
                   100.0 * gflops / cublas_gflops, max_err,
                   max_err > 1e-2f * std::sqrt((float)n) ? "  <-- MISMATCH" : "");
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dRef));
        printf("\n");
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
