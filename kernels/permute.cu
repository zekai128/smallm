#include "tensor.h"
#include <cuda_runtime.h>
#include <cstring>

__global__ void permute_backwards_kernel(int size, float* a_grad, float* out_grad, int* order, int ndim, int* p_shape, int* p_strides, int* a_strides) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int flat = bid*THREADS_PER_BLOCK + tid;
    if (flat >= size) return;

    int tmp = flat;
    int p_index[8];
    for (int i = ndim-1; i >= 0; i--) {
        p_index[i] = tmp % p_shape[i];
        tmp /= p_shape[i];
    }

    int p_flat_index = 0;
    for (int i = 0; i < ndim; i++)
        p_flat_index += p_index[i] * p_strides[i];

    int a_flat_index = 0;
    for (int i = 0; i < ndim; i++)
        a_flat_index += p_index[i] * a_strides[order[i]];

    a_grad[a_flat_index] += out_grad[p_flat_index];
}

Tensor* permute(Tensor* a, int* order) {
    int ndim = a->ndim;

    int* new_shape   = new int[ndim];
    int* new_strides = new int[ndim];
    for (int i = 0; i < ndim; i++) {
        new_shape[i]   = a->shape[order[i]];
        new_strides[i] = a->strides[order[i]];
    }

    Tensor* out = new Tensor();
    out->data          = a->data;
    out->grad          = nullptr;
    out->size          = a->size;
    out->ndim          = ndim;
    out->shape         = new_shape;
    out->strides       = new_strides;
    out->requires_grad = false;
    out->is_view = true;

    if (a->requires_grad) {
        // copy order to heap so the lambda owns it (caller's array may be stack-allocated)
        int* order_copy = new int[ndim];
        memcpy(order_copy, order, ndim * sizeof(int));

        out->requires_grad = true;
        out->parents = {a};
        out->backward_fn = [a, out, ndim, order_copy]() {
            if (!a->grad) {
                cudaMalloc(&a->grad, a->size * sizeof(float));
                cudaMemset(a->grad, 0, a->size * sizeof(float));
            }

            // Copy host metadata arrays to device
            int *d_order, *d_p_shape, *d_p_strides, *d_a_strides;
            cudaMalloc(&d_order,     ndim * sizeof(int));
            cudaMalloc(&d_p_shape,   ndim * sizeof(int));
            cudaMalloc(&d_p_strides, ndim * sizeof(int));
            cudaMalloc(&d_a_strides, ndim * sizeof(int));
            cudaMemcpy(d_order,     order_copy,   ndim * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_p_shape,   out->shape,   ndim * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_p_strides, out->strides, ndim * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_a_strides, a->strides,   ndim * sizeof(int), cudaMemcpyHostToDevice);

            int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            permute_backwards_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                a->size, a->grad, out->grad, d_order, ndim, d_p_shape, d_p_strides, d_a_strides
            );

            cudaFree(d_order);
            cudaFree(d_p_shape);
            cudaFree(d_p_strides);
            cudaFree(d_a_strides);
            delete[] order_copy;
        };
    }

    return out;
}
