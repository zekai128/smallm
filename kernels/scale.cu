#include "tensor.h"
#include "utils.cuh"
#include <cuda_runtime.h>

struct ScaleOp {
    float s;
    __device__ float operator()(float x) const { return x * s; }
};

Tensor* scale(Tensor* a, float s) {
    Tensor* out = zeros(a->shape, a->ndim);
    int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    elementwise_kernel<<<blocks, THREADS_PER_BLOCK>>>(a->data, out->data, a->size, ScaleOp{s});

    if (a->requires_grad) {
        out->requires_grad = true;
        out->parents = {a};
        out->backward_fn = [a, out, s]() {
            if (!a->grad) {
                cudaMalloc(&a->grad, a->size * sizeof(float));
                cudaMemset(a->grad, 0, a->size * sizeof(float));
            }
            int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            elementwise_kernel<<<blocks, THREADS_PER_BLOCK>>>(out->grad, a->grad, a->size, ScaleOp{s});
        };
    }

    return out;
}
