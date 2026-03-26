#include <cuda_runtime.h>
#include "tensor.h"

__global__ void relu_kernel(float* a, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) c[i] = fmaxf(0.0f, a[i]);
}

Tensor* relu(Tensor* a) {
    Tensor* c = zeros(a->shape, a->ndim);

    int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    relu_kernel<<<blocks, THREADS_PER_BLOCK>>>(a->data, c->data, a->size);

    return c;
}
