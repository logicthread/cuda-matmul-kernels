NVCC  ?= nvcc
# "native" targets the GPU in this machine (needs CUDA >= 11.5). Override e.g. ARCH=sm_75 for a T4.
ARCH  ?= native
FLAGS ?= -O3 -std=c++17 -lineinfo

SRCS := $(wildcard src/*.cuh)

build/bench: benchmarks/bench.cu $(SRCS)
	@mkdir -p build
	$(NVCC) $(FLAGS) -arch=$(ARCH) -o $@ $< -lcublas

run: build/bench
	./build/bench

clean:
	rm -rf build

.PHONY: run clean
