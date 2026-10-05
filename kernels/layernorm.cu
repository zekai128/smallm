#include <cuda_runtime.h>
#include <cfloat>
#include "tensor.h"
#include "utils.cuh"


__global__ void mean_diffs_kernel(float* in, float* out, float* rowmeans, int row_dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    for (int row_index = tid; row_index < row_dim; row_index += THREADS_PER_BLOCK) {
        out[row*row_dim + row_index] = in[row*row_dim + row_index] - rowmeans[row];
    }
}

float* mean_diffs(float* in, float* rowmeans, int size, int row_dim) {
    float* out;
    cudaMalloc(&out, size*sizeof(float));
    int blocks = size / row_dim;
    mean_diffs_kernel<<<blocks, THREADS_PER_BLOCK>>>(in, out, rowmeans, row_dim);
    return out;
}

__global__ void z_scores_kernel(float* in, float* out, float* rowmeans, float* row_stddevs, int row_dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    for (int row_index = tid; row_index < row_dim; row_index += THREADS_PER_BLOCK) {
        out[row*row_dim + row_index] = (in[row*row_dim + row_index] - rowmeans[row]) / row_stddevs[row];
    }
}

float* z_scores(float* in, float* rowmeans, float* row_stddevs, int size, int row_dim) {
    float* out;
    cudaMalloc(&out, size*sizeof(float));
    int blocks = size / row_dim;
    z_scores_kernel<<<blocks, THREADS_PER_BLOCK>>>(in, out, rowmeans, row_stddevs, row_dim);
    return out;
}

__global__ void scale_and_bias_kernel(float* data, float* out, float* gamma, float* beta, int dim_size) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    for (int row_i = tid; row_i < dim_size; row_i += THREADS_PER_BLOCK) {
        out[row*dim_size + row_i] = data[row*dim_size + row_i] * gamma[row_i] + beta[row_i];
    }
}

// Computes per-row sums:
//   A[row] = Σⱼ gamma[j] * grad[row,j]
//   B[row] = Σⱼ gamma[j] * grad[row,j] * xnorm[row,j]
__global__ void row_grad_sums_kernel(float* gamma, float* xnorm, float* grad, float* A, float* B, int last_dim) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    __shared__ float cache_A[THREADS_PER_BLOCK];
    __shared__ float cache_B[THREADS_PER_BLOCK];

    float acc_A = 0.0f;
    float acc_B = 0.0f;
    for (int i = tid; i < last_dim; i += THREADS_PER_BLOCK) {
        int idx = row * last_dim + i;
        float g = gamma[i] * grad[idx];
        acc_A += g;
        acc_B += g * xnorm[idx];
    }
    cache_A[tid] = acc_A;
    cache_B[tid] = acc_B;
    __syncthreads();

    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            cache_A[tid] += cache_A[tid + stride];
            cache_B[tid] += cache_B[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        A[row] = cache_A[0];
        B[row] = cache_B[0];
    }
}

// Propagates gradient to input a: one block per row
__global__ void data_grad_kernel(float* data_grad, float* grad, float* A, float* B, float* xnorm, float* row_stddevs, float* gamma, int last_dim) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    float inv_Nstd = 1.0f / (row_stddevs[row] * last_dim);
    for (int i = tid; i < last_dim; i += THREADS_PER_BLOCK) {
        int idx = row * last_dim + i;
        data_grad[idx] += inv_Nstd * (last_dim * gamma[i] * grad[idx] - A[row] - xnorm[idx] * B[row]);
    }
}

// Propagates gradient to gamma and beta: one block per column, reduces over rows
__global__ void gamma_beta_grad_kernel(float* gamma_grad, float* beta_grad, float* grad, float* xnorm, int rows, int last_dim) {
    int tid = threadIdx.x;
    int col = blockIdx.x;

    __shared__ float cache_gamma[THREADS_PER_BLOCK];
    __shared__ float cache_beta[THREADS_PER_BLOCK];

    float acc_gamma = 0.0f;
    float acc_beta  = 0.0f;
    for (int i = tid; i < rows; i += THREADS_PER_BLOCK) {
        int idx = i * last_dim + col;
        acc_gamma += grad[idx] * xnorm[idx];
        acc_beta  += grad[idx];
    }
    cache_gamma[tid] = acc_gamma;
    cache_beta[tid]  = acc_beta;
    __syncthreads();

    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            cache_gamma[tid] += cache_gamma[tid + stride];
            cache_beta[tid]  += cache_beta[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        gamma_grad[col] += cache_gamma[0];
        beta_grad[col]  += cache_beta[0];
    }
}

Tensor* layer_norm(Tensor* a, Tensor* gamma, Tensor* beta, float eps) {
    int last_dim = a->shape[a->ndim-1];
    int rows = a->size / last_dim;

    float* rowsums = row_reduce(a->data, rows, last_dim, SumReduceOp{});
    float* rowmeans = elementwise(rowsums, rows, DivideElementwiseOp{(float)last_dim});
    cudaFree(rowsums);

    float* meandiffs = mean_diffs(a->data, rowmeans, a->size, last_dim);

    float* meandiffs_squared = elementwise(meandiffs, a->size, SquareElementwiseOp{});
    cudaFree(meandiffs);

    float* meandiffs_squared_summed_by_row = row_reduce(meandiffs_squared, rows, last_dim, SumReduceOp{});
    cudaFree(meandiffs_squared);

    float* row_variances = elementwise(meandiffs_squared_summed_by_row, rows, DivideElementwiseOp{(float)last_dim});
    cudaFree(meandiffs_squared_summed_by_row);

    float* row_stddevs = elementwise(row_variances, rows, SqrtElementwiseOp{eps});
    cudaFree(row_variances);

    float* xnorm = z_scores(a->data, rowmeans, row_stddevs, a->size, last_dim);
    cudaFree(rowmeans);

    Tensor* out = zeros(a->shape, a->ndim);
    scale_and_bias_kernel<<<rows, THREADS_PER_BLOCK>>>(xnorm, out->data, gamma->data, beta->data, last_dim);

    if (a->requires_grad) {
        out->requires_grad = true;
        out->parents = {a, gamma, beta};
        out->backward_fn = [a, gamma, beta, out, xnorm, row_stddevs, rows, last_dim]() {
            if (!a->grad) {
                cudaMalloc(&a->grad, a->size * sizeof(float));
                cudaMemset(a->grad, 0, a->size * sizeof(float));
            }
            if (!gamma->grad) {
                cudaMalloc(&gamma->grad, gamma->size * sizeof(float));
                cudaMemset(gamma->grad, 0, gamma->size * sizeof(float));
            }
            if (!beta->grad) {
                cudaMalloc(&beta->grad, beta->size * sizeof(float));
                cudaMemset(beta->grad, 0, beta->size * sizeof(float));
            }

            float* A;
            float* B;
            cudaMalloc(&A, rows * sizeof(float));
            cudaMalloc(&B, rows * sizeof(float));
            row_grad_sums_kernel<<<rows, THREADS_PER_BLOCK>>>(gamma->data, xnorm, out->grad, A, B, last_dim);

            data_grad_kernel<<<rows, THREADS_PER_BLOCK>>>(a->grad, out->grad, A, B, xnorm, row_stddevs, gamma->data, last_dim);
            gamma_beta_grad_kernel<<<last_dim, THREADS_PER_BLOCK>>>(gamma->grad, beta->grad, out->grad, xnorm, rows, last_dim);

            cudaFree(A);
            cudaFree(B);
            cudaFree(xnorm);
            cudaFree(row_stddevs);
        };
    } else {
        cudaFree(xnorm);
        cudaFree(row_stddevs);
    }

    return out;
}
