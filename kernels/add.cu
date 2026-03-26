#include <cassert>
#include <cuda_runtime.h>
#include "tensor.h"

__global__ void add_kernel(float* a, float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) c[i] = a[i] + b[i];
}

Tensor* add(Tensor* a, Tensor* b) {
    assert(a->ndim == b->ndim);
    for (int i = 0; i < a->ndim; i++) assert(a->shape[i] == b->shape[i]);

    Tensor* c = zeros(a->shape, a->ndim);

    int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_kernel<<<blocks, THREADS_PER_BLOCK>>>(a->data, b->data, c->data, a->size);

    return c;
}
