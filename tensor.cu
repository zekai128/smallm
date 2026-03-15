#include <stdio.h>
#include <cuda_runtime.h>
#include "tensor.h"

Tensor* zeros(int* shape, int ndim) {
    Tensor* t = new Tensor();

    t->ndim = ndim;
    t->shape = new int[ndim];
    memcpy(t->shape, shape, sizeof(int)*ndim);

    t->strides = new int[ndim];
    t->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->strides[i] = t->size;
        t->size *= shape[i];
    }

    cudaMalloc(&t->data, t->size * sizeof(float));
    cudaMemset(t->data, 0, t->size * sizeof(float));
    t->grad = nullptr;

    t->requires_grad = false;

    return t;
}

Tensor* from_host(float* data, int* shape, int ndim) {
    Tensor* t = zeros(shape, ndim);
    cudaMemcpy(t->data, data, t->size * sizeof(float), cudaMemcpyHostToDevice);
    return t;
}

void to_host(Tensor* t, float* out) {
    cudaMemcpy(out, t->data, t->size * sizeof(float), cudaMemcpyDeviceToHost);
}

void free_tensor(Tensor* t) {
    cudaFree(t->data);
    if (t->grad) cudaFree(t->grad);
    delete[] t->shape;
    delete[] t->strides;
    delete t;
}
