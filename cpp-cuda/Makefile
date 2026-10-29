CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 90
NVCC := $(CUDA_HOME)/bin/nvcc
TARGET := build/pdsdp-cuda
SOURCE := src/pdsdp_cuda.cu

NVCCFLAGS := -O3 -DNDEBUG -std=c++17 -lineinfo \
	-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
	-Xcompiler=-Wall,-Wextra,-fopenmp
LDLIBS := -lcusparse -lcusolver -lcublas -lcudart -Xcompiler=-fopenmp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET)
