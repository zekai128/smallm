#include <cuda_runtime.h>
#include "tensor.h"

__global__ void relu_kernel(float* a, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) c[i] = fmaxf(0.0f, a[i]);
}

__global__ void relu_backwards_kernel(float* a_grad, float* c_grad, int size, float* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && a[i] > 0) a_grad[i] += c_grad[i];
}

Tensor* relu(Tensor* a) {
    Tensor* c = zeros(a->shape, a->ndim);

    int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    relu_kernel<<<blocks, THREADS_PER_BLOCK>>>(a->data, c->data, a->size);

    if (a->requires_grad) {
        c->requires_grad = true;
        c->parents = {a};
        c->backward_fn = [a, c, blocks]() {
            if (!a->grad) {
                cudaMalloc(&a->grad, a->size * sizeof(float));
                cudaMemset(a->grad, 0, a->size * sizeof(float));
            }
            relu_backwards_kernel<<<blocks, THREADS_PER_BLOCK>>>(a->grad, c->grad, c->size, a->data);
        };
    }

    return c;
}
