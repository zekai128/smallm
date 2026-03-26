#include <stdio.h>
#include <cublas_v2.h>
#include "tensor.h"

int main() {
    // --- matmul test ---
    // shape (2,2,2,2): 4 batches of 2x2 matrices multiplied by identity
    // expected output == a unchanged
    printf("=== matmul ===\n");
    float a_data[] = {1,2,3,4,  5,6,7,8,  9,10,11,12,  13,14,15,16};
    float b_data[] = {1,0,0,1,  1,0,0,1,  1,0,0,1,     1,0,0,1};
    int shape[] = {2, 2, 2, 2};
    Tensor* a = from_host(a_data, shape, 4);
    Tensor* b = from_host(b_data, shape, 4);
    print_tensor(matmul(a, b));
    // expected: same as a

    // --- add test ---
    // [1,2,3,4] + [10,20,30,40] = [11,22,33,44]
    printf("=== add ===\n");
    float x_data[] = {1, 2, 3, 4};
    float y_data[] = {10, 20, 30, 40};
    int shape2[] = {4};
    Tensor* x = from_host(x_data, shape2, 1);
    Tensor* y = from_host(y_data, shape2, 1);
    print_tensor(add(x, y));
    // expected: [11, 22, 33, 44]

    // --- softmax test (2D) ---
    // row 0: [1,2,3] -> [0.0900, 0.2447, 0.6652]
    // row 1: [1,1,1] -> [0.3333, 0.3333, 0.3333]
    printf("=== softmax 2D ===\n");
    float s_data[] = {1, 2, 3,  1, 1, 1};
    int s_shape[] = {2, 3};
    Tensor* s = from_host(s_data, s_shape, 2);
    print_tensor(softmax(s));

    // --- softmax test (3D, THREADS_PER_BLOCK=2 to stress test padding) ---
    // set THREADS_PER_BLOCK=2 before running this: last_dim=3, blocks_per_row=2
    printf("=== softmax 3D ===\n");
    float s3_data[] = {1,2,3, 1,1,1,  1,2,3, 1,1,1};
    int s3_shape[] = {2, 2, 3};
    Tensor* s3 = from_host(s3_data, s3_shape, 3);
    print_tensor(softmax(s3));
    // expected: all 4 rows same as above

    // --- relu test ---
    // [-2, -1, 0, 1, 2] -> [0, 0, 0, 1, 2]
    printf("=== relu ===\n");
    float z_data[] = {-2, -1, 0, 1, 2};
    int shape3[] = {5};
    Tensor* z = from_host(z_data, shape3, 1);
    print_tensor(relu(z));
    // expected: [0, 0, 0, 1, 2]
}
