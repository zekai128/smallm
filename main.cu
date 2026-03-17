#include <stdio.h>
#include <cublas_v2.h>
#include "tensor.h"

int main() {
    // A = [[1, 2], [3, 4]]
    float a_data[] = {1, 2};
    int a_shape[] = {1, 2};
    Tensor* a = from_host(a_data, a_shape, 2);

    // B = [[5, 6], [7, 8]]
    float b_data[] = {1,2,3,4,5,6};
    int b_shape[] = {2,3};
    Tensor* b = from_host(b_data, b_shape, 2);

    int c_shape[] = {1,3};
    // output buffer
    Tensor* c = zeros(c_shape, 2);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    int m = 1, n = 3, k = 2;

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
     */
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        a->data, m,
        b->data, k,
        &beta,
        c->data, n
    );

    // read result back
    float out[3];
    to_host(c, out);
    printf("result:\n%f %f\n%f %f\n", out[0], out[1], out[2]);

    cublasDestroy(handle);
    free_tensor(a); free_tensor(b); free_tensor(c);
    return 0;
}
