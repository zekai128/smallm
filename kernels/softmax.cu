#include <cuda_runtime.h>
#include <cfloat>
#include "tensor.h"

struct MaxOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
    static constexpr float identity = -FLT_MAX;
};

struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
    static constexpr float identity = 0.0f;
};

// Each block reduces one chunk of one row.
template <typename ReduceOp>
__global__ void row_reduce_kernel(float* out, float* in, int dim, int blocks_per_row, ReduceOp op, float identity) {
    int row   = blockIdx.x / blocks_per_row;
    int chunk = blockIdx.x % blocks_per_row;
    int tid   = threadIdx.x;

    int in_offset = row * dim + chunk * blockDim.x + tid;
    int in_col    = chunk * blockDim.x + tid;

    __shared__ float cache[THREADS_PER_BLOCK];
    cache[tid] = (in_col < dim) ? in[in_offset] : identity;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) cache[tid] = op(cache[tid], cache[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = cache[0];
}

// Returns a device buffer of shape (rows,) with one reduced value per row.
// Caller is responsible for freeing the returned pointer.
template <typename ReduceOp>
float* row_reduce(float* a, int rows, int last_dim, ReduceOp op) {
    if (last_dim == 1) {
        float* output;
        cudaMalloc(&output, rows * sizeof(float));
        cudaMemcpy(output, a, rows * sizeof(float), cudaMemcpyDeviceToDevice);
        return output;
    }

    int dim      = last_dim;
    float* input  = a;
    float* output = nullptr;

    while (dim > 1) {
        int blocks_per_row = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int total_blocks   = rows * blocks_per_row;

        cudaMalloc(&output, total_blocks * sizeof(float));
        row_reduce_kernel<<<total_blocks, THREADS_PER_BLOCK>>>(output, input, dim, blocks_per_row, op, ReduceOp::identity);

        if (input != a) cudaFree(input);
        input = output;
        dim   = blocks_per_row;
    }

    return output;  // shape: (rows,)
}

__global__ void elementwise_shifted_exp_kernel(float* out, float* in, float* rowmaxes, int blocks_per_row, int dim) {
    int row   = blockIdx.x / blocks_per_row;
    int chunk = blockIdx.x % blocks_per_row;
    int tid   = threadIdx.x;

    int in_offset = row * dim + chunk * blockDim.x + tid;
    int in_col    = chunk * blockDim.x + tid;
    if (in_col < dim) {
        out[in_offset] = expf(in[in_offset] - rowmaxes[row]);
    }
}

void elementwise_shifted_exp(float* out, float* in, float* rowmaxes, int rows, int dim) {
    int blocks_per_row = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int total_blocks   = rows * blocks_per_row;
    elementwise_shifted_exp_kernel<<<total_blocks, THREADS_PER_BLOCK>>>(out, in, rowmaxes, blocks_per_row, dim);
}

__global__ void elementwise_divide_by_sum_kernel(float* out, float* in, float* rowsums, int blocks_per_row, int dim) {
    int row   = blockIdx.x / blocks_per_row;
    int chunk = blockIdx.x % blocks_per_row;
    int tid   = threadIdx.x;

    int in_offset = row * dim + chunk * blockDim.x + tid;
    int in_col    = chunk * blockDim.x + tid;
    if (in_col < dim) {
        out[in_offset] = in[in_offset] / rowsums[row];
    }
}

void elementwise_divide_by_sum(float* out, float* in, float* rowsums, int rows, int dim) {
    int blocks_per_row = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int total_blocks   = rows * blocks_per_row;
    elementwise_divide_by_sum_kernel<<<total_blocks, THREADS_PER_BLOCK>>>(out, in, rowsums, blocks_per_row, dim);
}

/**
 * numerically stable softmax over last dim, supports arbitrary tensor shapes.
 * 1. compute row maxes
 * 2. compute exp(x - row_max) elementwise
 * 3. compute row sums of exp
 * 4. divide each element by its row sum
 */
Tensor* softmax(Tensor* a) {
    int last_dim = a->shape[a->ndim - 1];
    int rows     = a->size / last_dim;

    // step 1: row maxes -> (rows,)
    float* rowmaxes = row_reduce(a->data, rows, last_dim, MaxOp{});

    // step 2: exp(x - row_max) -> same size as a
    Tensor* shifted_exp = zeros(a->shape, a->ndim);
    elementwise_shifted_exp(shifted_exp->data, a->data, rowmaxes, rows, last_dim);

    // step 3: row sums of exp -> (rows,)
    float* rowsums = row_reduce(shifted_exp->data, rows, last_dim, SumOp{});

    // step 4: elementwise divide by row sums
    Tensor* output = zeros(a->shape, a->ndim);
    elementwise_divide_by_sum(output->data, shifted_exp->data, rowsums, rows, last_dim);

    cudaFree(rowmaxes);
    cudaFree(rowsums);
    free_tensor(shifted_exp);

    return output;
}
