#include <cstdio>
#include <cstdlib>
#include "tensor.h"
#include "dataloader.h"
#include "transformer.h"

static void free_activations(std::vector<Tensor*>& activations) {
    for (Tensor* t : activations) free_tensor(t);
    activations.clear();
}

int main() {
    srand(42);

    const int B        = 4;
    const int T        = 64;
    const int d_model  = 256;
    const int n_heads  = 4;
    const int n_layers = 4;
    const float lr     = 1e-3f;
    const int n_iters  = 2000;

    DataLoader dl;
    dataloader_init(&dl, "data/tinyshakespeare.txt", B, T);

    Transformer model = make_transformer(dl.vocab_size, T, d_model, n_heads, n_layers);
    std::vector<Tensor*> params = transformer_params(&model);

    for (int iter = 0; iter < n_iters; iter++) {
        Tensor* token_ids;
        Tensor* labels;
        dataloader_next_batch(&dl, &token_ids, &labels);

        std::vector<Tensor*> activations;
        Tensor* probs = transformer_forward(&model, token_ids, B, T, d_model, n_heads, activations);

        int labels_shape[] = {B * T};
        Tensor* labels_flat = reshape(labels, labels_shape, 1);

        Tensor* loss = cross_entropy(probs, labels_flat);

        // print loss
        float loss_val;
        to_host(loss, &loss_val);
        printf("iter %d  loss: %.4f\n", iter, loss_val);

        // backward
        backward(loss);

        // optimizer step
        sgd_step(params, lr);
        zero_grad(params);

        // free activations and batch tensors
        free_tensor(loss);
        free_tensor(labels);
        free_tensor(token_ids);
        free_activations(activations);
    }

    dataloader_free(&dl);
    return 0;
}
