#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cublas_v2.h>
#include "tensor.h"

// ---- data loading ----

float* load_bin(const char* path, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    float* buf = new float[n];
    fread(buf, sizeof(float), n, f);
    fclose(f);
    return buf;
}

// ---- weight init: kaiming uniform ----

void kaiming_uniform(float* buf, int fan_in, int n) {
    float bound = sqrtf(1.0f / fan_in);
    for (int i = 0; i < n; i++) {
        float u = (float)rand() / RAND_MAX;  // [0, 1]
        buf[i] = (2.0f * u - 1.0f) * bound; // [-bound, bound]
    }
}

int main() {
    srand(42);

    // ---- hyperparameters ----
    const int INPUT_DIM  = 784;
    const int H1         = 256;
    const int H2         = 128;
    const int OUTPUT_DIM = 10;
    const int BATCH_SIZE = 64;
    const int EPOCHS     = 5;
    const float LR       = 0.01f;
    const int N_TRAIN    = 60000;

    // ---- load data ----
    float* h_images = load_bin("data/train_images.bin", N_TRAIN * INPUT_DIM);
    float* h_labels = load_bin("data/train_labels.bin", N_TRAIN);

    // ---- init weights ----
    int w1_shape[] = {INPUT_DIM, H1};
    int b1_shape[] = {H1};
    int w2_shape[] = {H1, H2};
    int b2_shape[] = {H2};
    int w3_shape[] = {H2, OUTPUT_DIM};
    int b3_shape[] = {OUTPUT_DIM};

    float* h_w1 = new float[INPUT_DIM * H1];
    float* h_w2 = new float[H1 * H2];
    float* h_w3 = new float[H2 * OUTPUT_DIM];

    kaiming_uniform(h_w1, INPUT_DIM, INPUT_DIM * H1);
    kaiming_uniform(h_w2, H1,        H1 * H2);
    kaiming_uniform(h_w3, H2,        H2 * OUTPUT_DIM);

    Tensor* W1 = from_host(h_w1, w1_shape, 2);  W1->requires_grad = true;
    Tensor* b1 = zeros(b1_shape, 1);              b1->requires_grad = true;
    Tensor* W2 = from_host(h_w2, w2_shape, 2);  W2->requires_grad = true;
    Tensor* b2 = zeros(b2_shape, 1);              b2->requires_grad = true;
    Tensor* W3 = from_host(h_w3, w3_shape, 2);  W3->requires_grad = true;
    Tensor* b3 = zeros(b3_shape, 1);              b3->requires_grad = true;

    delete[] h_w1; delete[] h_w2; delete[] h_w3;

    std::vector<Tensor*> params = {W1, b1, W2, b2, W3, b3};

    int steps_per_epoch = N_TRAIN / BATCH_SIZE;

    // ---- training loop ----
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0f;

        for (int step = 0; step < steps_per_epoch; step++) {
            int offset = step * BATCH_SIZE;

            // load batch onto GPU
            int x_shape[] = {BATCH_SIZE, INPUT_DIM};
            int y_shape[] = {BATCH_SIZE};
            Tensor* x      = from_host(h_images + offset * INPUT_DIM, x_shape, 2);
            Tensor* labels = from_host(h_labels + offset,              y_shape, 1);

            // forward pass: x -> h1 -> h2 -> logits -> probs -> loss
            Tensor* h1    = relu(add(matmul(x, W1), b1));
            Tensor* h2    = relu(add(matmul(h1, W2), b2));
            Tensor* logits = add(matmul(h2, W3), b3);
            Tensor* probs  = softmax(logits);
            Tensor* loss   = cross_entropy(probs, labels);

            // backward
            zero_grad(params);
            backward(loss);

            // optimizer step
            sgd_step(params, LR);

            // log loss
            float loss_val;
            to_host(loss, &loss_val);
            epoch_loss += loss_val;

            if (step % 100 == 0) {
                printf("epoch %d step %d/%d  loss=%.4f\n", epoch+1, step, steps_per_epoch, loss_val);
            }

            // free intermediates (not params)
            free_tensor(x);
            free_tensor(labels);
            free_tensor(h1);
            free_tensor(h2);
            free_tensor(logits);
            free_tensor(probs);
            free_tensor(loss);
        }

        printf("=== epoch %d  avg_loss=%.4f ===\n", epoch+1, epoch_loss / steps_per_epoch);
    }

    delete[] h_images;
    delete[] h_labels;

    return 0;
}
