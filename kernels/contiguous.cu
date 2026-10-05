#include "tensor.h"
#include <cuda_runtime.h>
#include <cstring>

__global__ void contiguous_kernel(float* out, float* in, int* shape, int* strides, int ndim, int size) {
    int flat = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (flat >= size) return;

    int tmp = flat;
    int idx[8];
    for (int i = ndim - 1; i >= 0; i--) {
        idx[i] = tmp % shape[i];
        tmp /= shape[i];
    }

    int src = 0;
    for (int i = 0; i < ndim; i++)
        src += idx[i] * strides[i];

    out[flat] = in[src];
}

__global__ void contiguous_backward_kernel(float* a_grad, float* out_grad, int* shape, int* strides, int ndim, int size) {
    int flat = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (flat >= size) return;

    int tmp = flat;
    int idx[8];
    for (int i = ndim - 1; i >= 0; i--) {
        idx[i] = tmp % shape[i];
        tmp /= shape[i];
    }

    int src = 0;
    for (int i = 0; i < ndim; i++)
        src += idx[i] * strides[i];

    a_grad[src] += out_grad[flat];
}

Tensor* contiguous(Tensor* a) {
    Tensor* out = zeros(a->shape, a->ndim);

    int* d_shape, *d_strides;
    cudaMalloc(&d_shape,   a->ndim * sizeof(int));
    cudaMalloc(&d_strides, a->ndim * sizeof(int));
    cudaMemcpy(d_shape,   a->shape,   a->ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, a->strides, a->ndim * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    contiguous_kernel<<<blocks, THREADS_PER_BLOCK>>>(out->data, a->data, d_shape, d_strides, a->ndim, a->size);

    cudaFree(d_shape);
    cudaFree(d_strides);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->parents = {a};
        out->backward_fn = [a, out]() {
            if (!a->grad) {
                cudaMalloc(&a->grad, a->size * sizeof(float));
                cudaMemset(a->grad, 0, a->size * sizeof(float));
            }

            int* d_shape, *d_strides;
            cudaMalloc(&d_shape,   a->ndim * sizeof(int));
            cudaMalloc(&d_strides, a->ndim * sizeof(int));
            cudaMemcpy(d_shape,   a->shape,   a->ndim * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_strides, a->strides, a->ndim * sizeof(int), cudaMemcpyHostToDevice);

            int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            contiguous_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(a->grad, out->grad, d_shape, d_strides, a->ndim, a->size);

            cudaFree(d_shape);
            cudaFree(d_strides);
        };
    }

    return out;
}
