#include "tensor.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <math.h>

static constexpr float SQRT2     = 1.41421356237f;
static constexpr float SQRT2PI   = 2.50662827463f;  // sqrt(2*pi)

struct GeluOp {
    __device__ float operator()(float x) const {
        return 0.5f * x * (1.0f + erff(x / SQRT2));
    }
};

struct GeluBackwardOp {
    float* x;
    __device__ float operator()(float grad, int i) const {
        float xi = x[i];
        float cdf = 0.5f * (1.0f + erff(xi / SQRT2));
        float pdf = expf(-0.5f * xi * xi) / SQRT2PI;
        return grad * (cdf + xi * pdf);
    }
};

__global__ void gelu_backward_kernel(float* grad_in, float* grad_out, float* x, int size) {
    int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if (i < size) {
        float xi  = x[i];
        float cdf = 0.5f * (1.0f + erff(xi / SQRT2));
        float pdf = expf(-0.5f * xi * xi) / SQRT2PI;
        grad_in[i] += grad_out[i] * (cdf + xi * pdf);
    }
}

Tensor* gelu(Tensor* a) {
    Tensor* out = zeros(a->shape, a->ndim);
    int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    elementwise_kernel<<<blocks, THREADS_PER_BLOCK>>>(a->data, out->data, a->size, GeluOp{});

    if (a->requires_grad) {
        out->requires_grad = true;
        out->parents = {a};
        out->backward_fn = [a, out]() {
            if (!a->grad) {
                cudaMalloc(&a->grad, a->size * sizeof(float));
                cudaMemset(a->grad, 0, a->size * sizeof(float));
            }
            int blocks = (a->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            gelu_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(a->grad, out->grad, a->data, a->size);
        };
    }

    return out;
}
