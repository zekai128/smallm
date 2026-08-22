#include <stdio.h>
#include <cuda_runtime.h>
#include "tensor.h"
#include <vector>
#include <stack>
#include <unordered_set>
#include <algorithm>

Tensor* zeros(int* shape, int ndim) {
    Tensor* t = new Tensor();

    t->ndim = ndim;
    t->shape = new int[ndim];
    memcpy(t->shape, shape, sizeof(int)*ndim);

    t->strides = new int[ndim];
    t->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->strides[i] = t->size;
        t->size *= shape[i];
    }

    cudaMalloc(&t->data, t->size * sizeof(float));
    cudaMemset(t->data, 0, t->size * sizeof(float));
    t->grad = nullptr;

    t->requires_grad = false;

    return t;
}

Tensor* from_host(float* data, int* shape, int ndim) {
    Tensor* t = zeros(shape, ndim);
    cudaMemcpy(t->data, data, t->size * sizeof(float), cudaMemcpyHostToDevice);
    return t;
}

void to_host(Tensor* t, float* out) {
    cudaMemcpy(out, t->data, t->size * sizeof(float), cudaMemcpyDeviceToHost);
}

void free_tensor(Tensor* t) {
    cudaFree(t->data);
    if (t->grad) cudaFree(t->grad);
    delete[] t->shape;
    delete[] t->strides;
    delete t;
}

void print_tensor(Tensor* t) {
    float* host = new float[t->size];
    to_host(t, host);

    printf("Tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("])\n");

    int total = t->size;
    for (int i = 0; i < total; i++) {
        int stride = t->size;
        for (int d = 0; d < t->ndim; d++) {
            if (i % stride == 0) printf("[");
            stride /= t->shape[d];
        }

        printf("%.4f", host[i]);

        stride = 1;
        for (int d = t->ndim - 1; d >= 0; d--) {
            stride *= t->shape[d];
            if ((i + 1) % stride == 0) {
                printf("]");
                if (d > 0 && (i + 1) % (stride * t->shape[d-1]) != 0) printf("\n");
            } else {
                printf(", ");
                break;
            }
        }
    }
    printf("\n");

    delete[] host;
}

__global__ void sgd_kernel(float* data, float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] -= lr * grad[i];
}

void sgd_step(std::vector<Tensor*>& params, float lr) {
    for (Tensor* p : params) {
        if (!p->grad) continue;
        int blocks = (p->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sgd_kernel<<<blocks, THREADS_PER_BLOCK>>>(p->data, p->grad, lr, p->size);
    }
}

void zero_grad(std::vector<Tensor*>& params) {
    for (Tensor* p : params) {
        if (p->grad) cudaMemset(p->grad, 0, p->size * sizeof(float));
    }
}

void dfs(std::vector<Tensor*>& topo, std::unordered_set<Tensor*>& visited, Tensor* node) {
    if (visited.count(node)) return;
    visited.insert(node);
    for (Tensor* parent : node->parents) {
        dfs(topo, visited, parent);
    }
    topo.push_back(node);
}

void backward(Tensor* a) {
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;

    cudaMalloc(&a->grad, a->size * sizeof(float));
    std::vector<float> ones(a->size, 1.0f);
    cudaMemcpy(a->grad, ones.data(), a->size * sizeof(float), cudaMemcpyHostToDevice);

    dfs(topo, visited, a);
    std::reverse(topo.begin(), topo.end());
    for (int i = 0; i < topo.size(); i++) {
        if (topo[i]->backward_fn) topo[i]->backward_fn();
    }
}
