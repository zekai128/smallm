#include "tensor.h"
#include <cuda_runtime.h>

__global__ void masked_fill_kernel(float* out, float* in, float* mask, float val, int size) {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (i >= size) return;
    out[i] = mask[i] ? val : in[i];
}

__global__ void masked_fill_backward_kernel(float* a_grad, float* out_grad, float* mask, int size) {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (i >= size) return;
    if (!mask[i]) a_grad[i] += out_grad[i];
}

Tensor* masked_fill(Tensor* a, Tensor* mask, float val) {
    Tensor* out = zeros(a->shape, a->ndim);
    int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    masked_fill_kernel<<<blocks, THREADS_PER_BLOCK>>>(out->data, a->data, mask->data, val, a->size);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->parents = {a};
        out->backward_fn = [a, out, mask]() {
            if (!a->grad) {
                cudaMalloc(&a->grad, a->size * sizeof(float));
                cudaMemset(a->grad, 0, a->size * sizeof(float));
            }
            int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            masked_fill_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(a->grad, out->grad, mask->data, a->size);
        };
    }

    return out;
}
