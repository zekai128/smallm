#include <cuda_runtime.h>
#include "tensor.h"

// labels is a flat int array on device of length batch_size
__global__ void cross_entropy_kernel(float* probs, int* labels, float* loss, int last_dim, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size)
        atomicAdd(loss, -logf(probs[i * last_dim + labels[i]]) / batch_size);
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

    cudaFree(d_labels);
    return loss;
}
