#include <cuda_runtime.h>
#include <cfloat>
#include "tensor.h"
#include "utils.cuh"


// One block per row, each thread handles multiple elements via grid-stride loop.
__global__ void elementwise_shifted_exp_kernel(float* out, float* in, float* rowmaxes, int dim) {
    int row = blockIdx.x;
    for (int col = threadIdx.x; col < dim; col += THREADS_PER_BLOCK)
        out[row * dim + col] = expf(in[row * dim + col] - rowmaxes[row]);
}

void elementwise_shifted_exp(float* out, float* in, float* rowmaxes, int rows, int dim) {
    elementwise_shifted_exp_kernel<<<rows, THREADS_PER_BLOCK>>>(out, in, rowmaxes, dim);
}

__global__ void elementwise_divide_by_sum_kernel(float* out, float* in, float* rowsums, int dim) {
    int row = blockIdx.x;
    for (int col = threadIdx.x; col < dim; col += THREADS_PER_BLOCK)
        out[row * dim + col] = in[row * dim + col] / rowsums[row];
}

void elementwise_divide_by_sum(float* out, float* in, float* rowsums, int rows, int dim) {
    elementwise_divide_by_sum_kernel<<<rows, THREADS_PER_BLOCK>>>(out, in, rowsums, dim);
}

__global__ void elementwise_mul_kernel(float* out, float* a, float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = a[i] * b[i];
}

// grad_in[row,col] += s[row,col] * (g[row,col] - dot(g,s)[row])
__global__ void softmax_backward_kernel(float* grad_in, float* s, float* g, float* row_dots, int dim) {
    int row = blockIdx.x;
    for (int col = threadIdx.x; col < dim; col += THREADS_PER_BLOCK) {
        int idx = row * dim + col;
        grad_in[idx] += s[idx] * (g[idx] - row_dots[row]);
    }
}

Tensor* softmax(Tensor* a) {
    int last_dim = a->shape[a->ndim - 1];
    int rows     = a->size / last_dim;

    float* rowmaxes = row_reduce(a->data, rows, last_dim, MaxReduceOp{});

    Tensor* shifted_exp = zeros(a->shape, a->ndim);
    elementwise_shifted_exp(shifted_exp->data, a->data, rowmaxes, rows, last_dim);

    float* rowsums = row_reduce(shifted_exp->data, rows, last_dim, SumReduceOp{});

    Tensor* output = zeros(a->shape, a->ndim);
    elementwise_divide_by_sum(output->data, shifted_exp->data, rowsums, rows, last_dim);

    cudaFree(rowmaxes);
    cudaFree(rowsums);
    free_tensor(shifted_exp);

    if (a->requires_grad) {
        output->requires_grad = true;
        output->parents = {a};
        output->backward_fn = [a, output, rows, last_dim]() {
            if (!a->grad) {
                cudaMalloc(&a->grad, a->size * sizeof(float));
                cudaMemset(a->grad, 0, a->size * sizeof(float));
            }

            float* gs;
            cudaMalloc(&gs, a->size * sizeof(float));
            int flat_blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            elementwise_mul_kernel<<<flat_blocks, THREADS_PER_BLOCK>>>(gs, output->grad, output->data, a->size);

            float* row_dots = row_reduce(gs, rows, last_dim, SumReduceOp{});
            cudaFree(gs);

            softmax_backward_kernel<<<rows, THREADS_PER_BLOCK>>>(a->grad, output->data, output->grad, row_dots, last_dim);

            cudaFree(row_dots);
        };
    }

    return output;
}
