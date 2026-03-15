#include <stdio.h>
#include "tensor.h"

int main() {
    // simple sanity check — create a 2x3 zero tensor and print it
    int shape[] = {2, 3};
    Tensor* t = zeros(shape, 2);

    printf("zeros tensor created at GPU mem address %p", t->data);
    free_tensor(t);
    return 0;
}
