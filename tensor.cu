#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include "tensor.h"
#include <cublas_v2.h>
#include <cassert>
#include <cfloat>

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

    // print shape
    printf("Tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("])\n");

    // print data using strides to index into the flat array
    // we iterate over all elements and use indentation to show structure
    int total = t->size;
    for (int i = 0; i < total; i++) {
        // open brackets at dimension boundaries (outermost first)
        int stride = t->size;
        for (int d = 0; d < t->ndim; d++) {
            if (i % stride == 0) printf("[");
            stride /= t->shape[d];
        }

        printf("%.4f", host[i]);

        // close brackets and newlines at dimension boundaries
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

// swap trick: pass b first, a second
/**
 * A:
 * flat array: (a,b,c,d,e,f)
 * 
 * row major: 2x3
 * matrix: [ a b c 
 *           d e f ]
 * 
 * B: 
 * flat array: (g,h,i,j,l,k,m,n,o,p,q,r)
 * 
 * row major: 3x4
 * matrix:[ g h i j 
 *          k l m n
 *          o p q r ]
 * 
 * 
 * SWAP TRICK:
 * passing the same leading dimension (3 for A, 4 for B),
 * results in the following col major representations
 * 
 * A:
 * [ a d
 *   b e
 *   c f ]
 * 
 * B:
 * [ g k o
 *   h l p
 *   i m q
 *   j n r ]
 * 
 * cublas reads tensor1 and tensor2 in col major and computes
 * the result, writing it at output pointer in col major. So
 * if we pass tensor1 = B and tensor2 = A, with 4 and 3 as the 
 * respective leading dimension, cublas is going to calculate
 * B^T A^T = C^T. (where AB=C).
 * 
 * It will write the resulting 4x2 matrix in col major form. ie.
 * 
 * [ Ct_11 Ct 12
 *   Ct_21 Ct_22
 *   Ct_31 Ct_32
 *   Ct_41 Ct_42 ]
 * 
 * as a flat array this will be (Ct_11, Ct_21, Ct_31, Ct_41, Ct_12, Ct_22, Ct_32, Ct_42)
 * 
 * Since we will interpret this result in row major (2x4), we will 
 * interpret it as
 * 
 * [ Ct_11 Ct_21 Ct_31 Ct_41
 *   Ct_12 Ct_22 Ct_32 Ct_42 ]
 * 
 * We can see that the our interpreted result is exactly the transpose
 * of the cublas result (Ct), and therefore our interpreted result is
 * C. 
 * 
 * Therefore, to compute C=AB, we pass B as the first tensor param,
 * with the same original leading dimension, and A as the second 
 * tensor param with the same original leading dimension, and the resulting
 * flat array writen at output pointer will be C in row major interpretation. 
 *
 * 
 * 
 * 
 * High dimensional tensors:
 * 
 * - >2 dimensional tensors are just batched 2D matrices.
 * 
 * contiguous representations: 3D (2x2x2)
 * A A
 * 
 * 4D (2x2x2x2)
 * 
 * (A A) (A A)
 * (B B) (B B)
 */
Tensor* matmul(Tensor* a, Tensor* b) {
    assert(a->ndim == b->ndim);
    assert(a->shape[a->ndim-1] == b->shape[b->ndim-2]);
    
    for (int i = 0; i < a->ndim-2; i++) {
        assert(a->shape[i] == b->shape[i]);
    }

    int ndim = a->ndim;

    int m = a->shape[ndim-2];
    int n = b->shape[ndim-1];
    int k = a->shape[ndim-1];

    std::vector<int> c_shape(a->shape, a->shape + ndim - 2);
    c_shape.push_back(m);
    c_shape.push_back(n);
    Tensor* c = zeros(c_shape.data(), ndim);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    long long stride_A = m*k;
    long long stride_B = k*n;
    long long stride_C = m*n;
    int batchCount = 1;
    for (int i = 0; i < a->ndim-2; i++) {
        batchCount *= a->shape[i];
    }

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b->data, n, stride_B,
        a->data, k, stride_A,
        &beta,
        c->data, n, stride_C,
        batchCount
    );

    cublasDestroy(handle);
    return c;
};

__global__ void add_kernel(float* a, float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) c[i] = a[i] + b[i];
}

Tensor* add(Tensor* a, Tensor* b) {
    assert(a->ndim == b->ndim);
    for (int i = 0; i < a->ndim; i++) assert(a->shape[i] == b->shape[i]);

    Tensor* c = zeros(a->shape, a->ndim);

    int threads = 256;
    int blocks = (a->size + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a->data, b->data, c->data, a->size);

    return c;
}

// one block per row, blockDim.x must be >= last_dim (padded to power of 2)
__global__ void softmax_kernel(float* input, float* output, int last_dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* in_row  = input  + row * last_dim;
    float* out_row = output + row * last_dim;

    __shared__ float cache[256];

    // step 1: find max for numerical stability
    cache[tid] = (tid < last_dim) ? in_row[tid] : -FLT_MAX;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) cache[tid] = fmaxf(cache[tid], cache[tid + stride]);
        __syncthreads();
    }
    float max_val = cache[0];
    __syncthreads();

    // step 2: exp(x - max)
    float val = (tid < last_dim) ? expf(in_row[tid] - max_val) : 0.0f;
    cache[tid] = val;
    __syncthreads();

    // step 3: sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) cache[tid] += cache[tid + stride];
        __syncthreads();
    }
    float sum = cache[0];

    // step 4: normalize
    if (tid < last_dim) out_row[tid] = val / sum;
}

Tensor* softmax(Tensor* a) {
    Tensor* c = zeros(a->shape, a->ndim);

    int last_dim = a->shape[a->ndim - 1];
    int rows     = a->size / last_dim;

    // next power of 2 >= last_dim, capped at 256
    int threads = 1;
    while (threads < last_dim) threads *= 2;

    softmax_kernel<<<rows, threads>>>(a->data, c->data, last_dim);
    return c;
}

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

    int threads = 256, blocks = (batch_size + 255) / 256;
    cross_entropy_kernel<<<blocks, threads>>>(probs->data, d_labels, loss->data, last_dim, batch_size);

    cudaFree(d_labels);
    return loss;
}

__global__ void relu_kernel(float* a, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) c[i] = fmaxf(0.0f, a[i]); 
}

Tensor* relu(Tensor* a) {
    Tensor* c = zeros(a->shape, a->ndim);

    int threads = 256;
    int blocks = (a->size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(a->data, c->data, a->size);

    return c;
}

