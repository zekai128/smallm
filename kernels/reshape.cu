#include "tensor.h"
#include <cuda_runtime.h>
#include <cstring>

Tensor* reshape(Tensor* a, int* new_shape, int new_ndim) {
    Tensor* out = new Tensor();
    out->data = a->data;
    out->grad = nullptr;
    out->size = a->size;
    out->ndim = new_ndim;
    out->shape = new int[new_ndim];
    out->strides = new int[new_ndim];
    out->requires_grad = false;
    out->is_view = true;
    memcpy(out->shape, new_shape, new_ndim * sizeof(int));

    int stride = 1;
    for (int i = new_ndim - 1; i >= 0; i--) {
        out->strides[i] = stride;
        stride *= new_shape[i];
    }

    if (a->requires_grad) {
        out->requires_grad = true;
        out->parents = {a};
        out->backward_fn = [a, out]() {
            if (a->grad) cudaFree(a->grad);
            a->grad = out->grad;
            out->grad = nullptr;
        };
    }

    return out;
}
