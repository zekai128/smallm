#include <cfloat>
#include "tensor.h"
#include <cuda_runtime.h>

struct MaxReduceOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
    static constexpr float identity = -FLT_MAX;
};

struct SumReduceOp {
    __device__ float operator()(float a, float b) const { return a + b; }
    static constexpr float identity = 0.0f;
};

struct DivideElementwiseOp {
    float divisor;
    __device__ float operator()(float a) const { return a / divisor; }
};

struct SquareElementwiseOp {
    __device__ float operator()(float a) const { return a*a; }
};

struct SqrtElementwiseOp {
    float eps;
    __device__ float operator()(float a) const { return sqrtf(a+eps); }
};

template <typename ElementwiseOp> 
__global__ void elementwise_kernel(float* in, float* out, int size, ElementwiseOp op) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = bid*THREADS_PER_BLOCK + tid;
    if (index < size) {
        out[index] = op(in[index]);
    }
}

template <typename ElementwiseOp>
float* elementwise(float* in, int size, ElementwiseOp op) {
    float* out;
    cudaMalloc(&out, size*sizeof(float));

    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    elementwise_kernel<<<blocks, THREADS_PER_BLOCK>>>(in, out, size, op);
    return out;
}

// One block per row. Each thread accumulates multiple elements via grid-stride loop,
// then a shared memory tree reduction combines them to a single value per row.
template <typename ReduceOp>
__global__ void row_reduce_kernel(float* out, float* in, int dim, ReduceOp op, float identity) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    __shared__ float cache[THREADS_PER_BLOCK];

    float acc = identity;
    for (int i = tid; i < dim; i += THREADS_PER_BLOCK)
        acc = op(acc, in[row * dim + i]);
    cache[tid] = acc;
    __syncthreads();

    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) cache[tid] = op(cache[tid], cache[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) out[row] = cache[0];
}

// Returns a device buffer of shape (rows,) with one reduced value per row.
// Caller is responsible for freeing the returned pointer.
template <typename ReduceOp>
float* row_reduce(float* a, int rows, int last_dim, ReduceOp op) {
    float* output;
    cudaMalloc(&output, rows * sizeof(float));
    row_reduce_kernel<<<rows, THREADS_PER_BLOCK>>>(output, a, last_dim, op, ReduceOp::identity);
    return output;
}