#pragma once

#include <functional>
#include <vector>

constexpr int THREADS_PER_BLOCK = 1024;

struct Tensor {
    float* data;         // GPU memory
    float* grad;         // GPU memory for gradients
    int*   shape;
    int*   strides;
    int    ndim;
    int    size;         // total number of elements
    bool   requires_grad;

    // Autograd
    std::vector<Tensor*>  parents;
    std::function<void()> backward_fn;
};

// ---- Memory ----
Tensor* zeros(int* shape, int ndim);
Tensor* from_host(float* data, int* shape, int ndim);
void    to_host(Tensor* t, float* out);
void    free_tensor(Tensor* t);

// ---- Print ----
void    print_tensor(Tensor* t);

// ---- Math ----
Tensor* add(Tensor* a, Tensor* b);
Tensor* matmul(Tensor* a, Tensor* b);
Tensor* relu(Tensor* a);
Tensor* gelu(Tensor* a);
Tensor* mul(Tensor* a, Tensor* b);
Tensor* scale(Tensor* a, float s);
Tensor* softmax(Tensor* a);
Tensor* layer_norm(Tensor* a, Tensor* gamma, Tensor* beta, float eps);
Tensor* cross_entropy(Tensor* probs, Tensor* labels);
Tensor* embedding(Tensor* weight, Tensor* indices);
Tensor* permute(Tensor* a, int* order);
Tensor* reshape(Tensor* a, int* new_shape, int new_ndim);
Tensor* masked_fill(Tensor* a, Tensor* mask, float val);

// ---- Autograd ----
void backward(Tensor* a);

// ---- Optimizer ----
void sgd_step(std::vector<Tensor*>& params, float lr);
void zero_grad(std::vector<Tensor*>& params);