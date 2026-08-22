#include <cassert>
#include <cuda_runtime.h>
#include <vector>
#include "tensor.h"

#define MAX_DIMS 8

// Each thread handles one output gradient element and atomically accumulates
// it into the corresponding input gradient, applying the same broadcast clamping
// logic as the forward kernel.
__global__ void add_backward_kernel(
    float* grad_out,
    float* grad_a, float* grad_b,
    int* a_shape, int* b_shape, int* out_shape,
    int a_ndim, int b_ndim, int out_ndim,
    int out_size,
    bool do_a, bool do_b
) {
    int out_flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_flat >= out_size) return;

    int out_idx[MAX_DIMS];
    int remaining = out_flat;
    for (int d = out_ndim - 1; d >= 0; d--) {
        out_idx[d] = remaining % out_shape[d];
        remaining /= out_shape[d];
    }

    int a_offset = out_ndim - a_ndim;
    int a_flat = 0, a_stride = 1;
    for (int d = out_ndim - 1; d >= 0; d--) {
        int a_d = d - a_offset;
        if (a_d >= 0) {
            int idx = (a_shape[a_d] == 1) ? 0 : out_idx[d];
            a_flat += idx * a_stride;
            a_stride *= a_shape[a_d];
        }
    }

    int b_offset = out_ndim - b_ndim;
    int b_flat = 0, b_stride = 1;
    for (int d = out_ndim - 1; d >= 0; d--) {
        int b_d = d - b_offset;
        if (b_d >= 0) {
            int idx = (b_shape[b_d] == 1) ? 0 : out_idx[d];
            b_flat += idx * b_stride;
            b_stride *= b_shape[b_d];
        }
    }

    if (do_a) atomicAdd(&grad_a[a_flat], grad_out[out_flat]);
    if (do_b) atomicAdd(&grad_b[b_flat], grad_out[out_flat]);
}

__global__ void add_kernel(
    float* a, float* b, float* c,
    int* a_shape, int* b_shape, int* out_shape,
    int a_ndim, int b_ndim, int out_ndim,
    int out_size
) {
    int out_flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_flat >= out_size) return;

    // convert flat output index to multi-index
    int out_idx[MAX_DIMS];
    int remaining = out_flat;
    for (int d = out_ndim - 1; d >= 0; d--) {
        out_idx[d] = remaining % out_shape[d];
        remaining /= out_shape[d];
    }

    // compute flat index into a, clamping broadcast dims to 0
    int a_flat = 0;
    int a_stride = 1;
    int a_offset = out_ndim - a_ndim;  // how many leading dims a is missing
    for (int d = out_ndim - 1; d >= 0; d--) {
        int a_d = d - a_offset;
        if (a_d >= 0) {
            int idx = (a_shape[a_d] == 1) ? 0 : out_idx[d];
            a_flat += idx * a_stride;
            a_stride *= a_shape[a_d];
        }
    }

    // compute flat index into b, clamping broadcast dims to 0
    int b_flat = 0;
    int b_stride = 1;
    int b_offset = out_ndim - b_ndim;
    for (int d = out_ndim - 1; d >= 0; d--) {
        int b_d = d - b_offset;
        if (b_d >= 0) {
            int idx = (b_shape[b_d] == 1) ? 0 : out_idx[d];
            b_flat += idx * b_stride;
            b_stride *= b_shape[b_d];
        }
    }

    c[out_flat] = a[a_flat] + b[b_flat];
}

Tensor* add(Tensor* a, Tensor* b) {
    int out_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;

    // right-align shapes and compute output shape
    std::vector<int> out_shape(out_ndim);
    int a_offset = out_ndim - a->ndim;
    int b_offset = out_ndim - b->ndim;
    for (int d = 0; d < out_ndim; d++) {
        int a_dim = (d >= a_offset) ? a->shape[d - a_offset] : 1;
        int b_dim = (d >= b_offset) ? b->shape[d - b_offset] : 1;
        assert(a_dim == b_dim || a_dim == 1 || b_dim == 1);
        out_shape[d] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    Tensor* c = zeros(out_shape.data(), out_ndim);

    // copy shapes to device for kernel
    int *d_a_shape, *d_b_shape, *d_out_shape;
    cudaMalloc(&d_a_shape,   a->ndim   * sizeof(int));
    cudaMalloc(&d_b_shape,   b->ndim   * sizeof(int));
    cudaMalloc(&d_out_shape, out_ndim  * sizeof(int));
    cudaMemcpy(d_a_shape,   a->shape,        a->ndim   * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape,   b->shape,        b->ndim   * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape.data(), out_ndim * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (c->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        a->data, b->data, c->data,
        d_a_shape, d_b_shape, d_out_shape,
        a->ndim, b->ndim, out_ndim,
        c->size
    );

    cudaFree(d_a_shape);
    cudaFree(d_b_shape);
    cudaFree(d_out_shape);

    if (a->requires_grad || b->requires_grad) {
        c->requires_grad = true;
        c->parents = {a, b};
        c->backward_fn = [a, b, c, out_ndim, out_shape_vec = std::vector<int>(out_shape.data(), out_shape.data() + out_ndim)]() {
            if (a->requires_grad && !a->grad) {
                cudaMalloc(&a->grad, a->size * sizeof(float));
                cudaMemset(a->grad, 0, a->size * sizeof(float));
            }
            if (b->requires_grad && !b->grad) {
                cudaMalloc(&b->grad, b->size * sizeof(float));
                cudaMemset(b->grad, 0, b->size * sizeof(float));
            }

            int *d_a_shape, *d_b_shape, *d_out_shape;
            cudaMalloc(&d_a_shape,   a->ndim  * sizeof(int));
            cudaMalloc(&d_b_shape,   b->ndim  * sizeof(int));
            cudaMalloc(&d_out_shape, out_ndim * sizeof(int));
            cudaMemcpy(d_a_shape,    a->shape,             a->ndim  * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b_shape,    b->shape,             b->ndim  * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_shape,  out_shape_vec.data(), out_ndim * sizeof(int), cudaMemcpyHostToDevice);

            int blocks = (c->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            add_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                c->grad,
                a->grad, b->grad,
                d_a_shape, d_b_shape, d_out_shape,
                a->ndim, b->ndim, out_ndim,
                c->size,
                a->requires_grad, b->requires_grad
            );

            cudaFree(d_a_shape);
            cudaFree(d_b_shape);
            cudaFree(d_out_shape);
        };
    }

    return c;
}
