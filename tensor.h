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
Tensor* softmax(Tensor* a);
Tensor* cross_entropy(Tensor* probs, Tensor* labels);
