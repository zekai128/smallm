#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <cfloat>
#include "tensor.h"
#include "utils.cuh"

__global__ void embedding_kernel(float* weight, float* indices, float* out, int dim) {
    int tid = threadIdx.x;
    int index = blockIdx.x;

    int embedding_index = (int)indices[index];

    for (int i = tid; i < dim; i += THREADS_PER_BLOCK) {
        out[index*dim + i] = weight[embedding_index * dim + i];
    }
}

__global__ void embedding_backwards_kernel(float* weights_grad, float* indices, float* output_grad, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int embedding_index = (int)indices[row];
    for (int i = tid; i < dim; i += THREADS_PER_BLOCK) {
        atomicAdd(&weights_grad[embedding_index * dim +i], output_grad[row*dim + i]);
    }
}

Tensor* embedding(Tensor* weight, Tensor* indices) {
    int dim = weight->shape[weight->ndim-1];
    int rows = indices->size;

    std::vector<int> out_shape(indices->shape, indices->shape + indices->ndim);
    out_shape.push_back(dim);
    Tensor* output = zeros(out_shape.data(), out_shape.size());

    embedding_kernel<<<rows, THREADS_PER_BLOCK>>>(weight->data, indices->data, output->data, dim);


    if (weight->requires_grad) {
        output->requires_grad = true;
        output->parents = {weight};
        output->backward_fn = [weight, indices, output, dim, rows](){
            if (!weight->grad) {
                cudaMalloc(&weight->grad, weight->size * sizeof(float));
                cudaMemset(weight->grad, 0, weight->size * sizeof(float));
            }
           embedding_backwards_kernel<<<rows, THREADS_PER_BLOCK>>>(weight->grad, indices->data, output->grad, dim);
        };
    }
    return output;
}