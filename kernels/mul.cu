#include "tensor.h"
#include "utils.cuh"
#include <cuda_runtime.h>

__global__ void mul_kernel(float* out, float* a, float* b, int size) {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (i < size) out[i] = a[i] * b[i];
}

__global__ void mul_accum_kernel(float* out, float* a, float* b, int size) {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (i < size) out[i] += a[i] * b[i];
}

Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* out = zeros(a->shape, a->ndim);
    int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    mul_kernel<<<blocks, THREADS_PER_BLOCK>>>(out->data, a->data, b->data, a->size);

    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->parents = {a, b};
        out->backward_fn = [a, b, out]() {
            int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            if (a->requires_grad) {
                if (!a->grad) {
                    cudaMalloc(&a->grad, a->size * sizeof(float));
                    cudaMemset(a->grad, 0, a->size * sizeof(float));
                }
                mul_accum_kernel<<<blocks, THREADS_PER_BLOCK>>>(a->grad, out->grad, b->data, a->size);
            }
            if (b->requires_grad) {
                if (!b->grad) {
                    cudaMalloc(&b->grad, b->size * sizeof(float));
                    cudaMemset(b->grad, 0, b->size * sizeof(float));
                }
                mul_accum_kernel<<<blocks, THREADS_PER_BLOCK>>>(b->grad, out->grad, a->data, b->size);
            }
        };
    }

    return out;
}
