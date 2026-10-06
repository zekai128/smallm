#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "tensor.h"
#include "dataloader.h"
#include "transformer.h"

// Sample from probs[0..n) with temperature. Temperature < 1 = more peaked, > 1 = more random.
static int sample(float* probs, int n, float temperature) {
    // Recover log-probs, apply temperature, re-normalize via softmax, then sample
    std::vector<float> logits(n);
    for (int i = 0; i < n; i++)
        logits[i] = logf(probs[i] + 1e-10f) / temperature;

    // Softmax
    float max_l = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { logits[i] = expf(logits[i] - max_l); sum += logits[i]; }
    for (int i = 0; i < n; i++) logits[i] /= sum;

    // Cumulative sampling
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += logits[i];
        if (r < cumsum) return i;
    }
    return n - 1;
}

int main(int argc, char** argv) {
    srand(42);

    const int T       = 64;
    const int d_model  = 1024;
    const int n_heads  = 16;
    const int n_layers = 24;
    const int   n_gen       = 200;    // tokens to generate
    const float temperature = 0.8f;

    const char* prompt = argc > 1 ? argv[1] : "ROMEO:";

    DataLoader dl;
    dataloader_init(&dl, "data/tinyshakespeare.txt", 1, T);

    Transformer model = make_transformer(dl.vocab_size, T, d_model, n_heads, n_layers);
    load_checkpoint(&model, "checkpoint.bin");

    // Encode prompt into token ids
    int prompt_len = (int)strlen(prompt);
    std::vector<int> seq;
    for (int i = 0; i < prompt_len; i++) {
        int id = dl.char_to_id[(unsigned char)prompt[i]];
        if (id == -1) { fprintf(stderr, "unknown char '%c'\n", prompt[i]); exit(1); }
        seq.push_back(id);
    }

    printf("%s", prompt);
    fflush(stdout);

    for (int step = 0; step < n_gen; step++) {
        // Take the last T tokens as context (or pad with 0 if shorter)
        int ctx_len = (int)seq.size() < T ? (int)seq.size() : T;
        int offset  = (int)seq.size() - ctx_len;

        std::vector<float> tok_h(T, 0.0f);
        for (int i = 0; i < ctx_len; i++)
            tok_h[i] = (float)seq[offset + i];

        int shape[] = {1, T};
        Tensor* token_ids = from_host(tok_h.data(), shape, 2);

        std::vector<Tensor*> activations;
        Tensor* probs = transformer_forward(&model, token_ids, 1, T, d_model, n_heads, activations);

        // probs is [T, vocab_size] — take the row for the last real token
        int last_pos = ctx_len - 1;
        std::vector<float> probs_h(T * dl.vocab_size);
        to_host(probs, probs_h.data());

        int next_token = sample(probs_h.data() + last_pos * dl.vocab_size, dl.vocab_size, temperature);
        seq.push_back(next_token);

        printf("%c", dl.id_to_char[next_token]);
        fflush(stdout);

        free_tensor(token_ids);
        for (Tensor* t : activations) free_tensor(t);
    }

    printf("\n");
    dataloader_free(&dl);
    return 0;
}
