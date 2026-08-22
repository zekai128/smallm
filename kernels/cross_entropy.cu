#include <cuda_runtime.h>
#include "tensor.h"

// labels is a flat int array on device of length batch_size
__global__ void cross_entropy_kernel(float* probs, int* labels, float* loss, int last_dim, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size)
        atomicAdd(loss, -logf(probs[i * last_dim + labels[i]]) / batch_size);
}

// grad_probs[i, label[i]] += -1 / (batch_size * probs[i, label[i]])
__global__ void cross_entropy_backward_kernel(float* grad_probs, float* probs, int* labels, int last_dim, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        int idx = i * last_dim + labels[i];
        grad_probs[idx] += -1.0f / (batch_size * probs[idx]);
    }
}

// labels_tensor: 1D tensor of float, values are integer class indices
Tensor* cross_entropy(Tensor* probs, Tensor* labels_tensor) {
    int batch_size = probs->shape[0];
    int last_dim   = probs->shape[1];

    // cast float labels to int via host round-trip
    int* d_labels;
    cudaMalloc(&d_labels, batch_size * sizeof(int));
    float* h_labels_f = new float[batch_size];
    cudaMemcpy(h_labels_f, labels_tensor->data, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    int* h_labels_i = new int[batch_size];
    for (int i = 0; i < batch_size; i++) h_labels_i[i] = (int)h_labels_f[i];
    cudaMemcpy(d_labels, h_labels_i, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_labels_f;
    delete[] h_labels_i;

    int scalar_shape[] = {1};
    Tensor* loss = zeros(scalar_shape, 1);

    int blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cross_entropy_kernel<<<blocks, THREADS_PER_BLOCK>>>(probs->data, d_labels, loss->data, last_dim, batch_size);

    if (probs->requires_grad) {
        loss->requires_grad = true;
        loss->parents = {probs};
        loss->backward_fn = [probs, loss, d_labels, batch_size, last_dim, blocks]() {
            if (!probs->grad) {
                cudaMalloc(&probs->grad, probs->size * sizeof(float));
                cudaMemset(probs->grad, 0, probs->size * sizeof(float));
            }
            cross_entropy_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(probs->grad, probs->data, d_labels, last_dim, batch_size);
            cudaFree(d_labels);
        };
    } else {
        cudaFree(d_labels);
    }

    return loss;
}
