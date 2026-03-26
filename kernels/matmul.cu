#include <vector>
#include <cassert>
#include <cublas_v2.h>
#include "tensor.h"

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
}
