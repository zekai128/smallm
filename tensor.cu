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

void print_tensor(Tensor* t) {
    float* host = new float[t->size];
    to_host(t, host);

    printf("Tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("])\n");

    int total = t->size;
    for (int i = 0; i < total; i++) {
        int stride = t->size;
        for (int d = 0; d < t->ndim; d++) {
            if (i % stride == 0) printf("[");
            stride /= t->shape[d];
        }

        printf("%.4f", host[i]);

        stride = 1;
        for (int d = t->ndim - 1; d >= 0; d--) {
            stride *= t->shape[d];
            if ((i + 1) % stride == 0) {
                printf("]");
                if (d > 0 && (i + 1) % (stride * t->shape[d-1]) != 0) printf("\n");
            } else {
                printf(", ");
                break;
            }
        }
    }
    printf("\n");

    delete[] host;
}
