#pragma once

#include <functional>
#include <vector>

struct Tensor {
    float* data;         // GPU memory
    float* grad;         // GPU memory for gradients
    int*   shape;
    int*   strides;
    int    ndim;
    int    size;         // total number of elements
    bool   requires_grad;

    // Autograd
    std::vector<Tensor*>  children;
    std::function<void()> backward_fn;
};

// ---- Memory ----
Tensor* zeros(int* shape, int ndim);
Tensor* from_host(float* data, int* shape, int ndim);
void    to_host(Tensor* t, float* out);
void    free_tensor(Tensor* t);

// ---- Math ----
Tensor* matmul(Tensor* a, Tensor* b);