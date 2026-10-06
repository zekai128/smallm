#pragma once
#include "tensor.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct DataLoader {
    int*  tokens;      // full tokenized text on CPU
    int   n_tokens;
    int   B;
    int   T;
    int   vocab_size;
    int   char_to_id[256];
    char  id_to_char[256];
};

// Reads text file, maps each unique char to an integer id.
// Fills vocab (indexed by char, value = token id, -1 if not in vocab).
inline void dataloader_init(DataLoader* dl, const char* path, int B, int T) {
    dl->B = B;
    dl->T = T;

    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* text = new char[file_size + 1];
    fread(text, 1, file_size, f);
    text[file_size] = '\0';
    fclose(f);

    // Build vocab: unique chars in order of first appearance
    memset(dl->char_to_id, -1, sizeof(dl->char_to_id));
    int vocab_size = 0;
    for (long i = 0; i < file_size; i++) {
        unsigned char c = (unsigned char)text[i];
        if (dl->char_to_id[c] == -1) {
            dl->char_to_id[c] = vocab_size;
            dl->id_to_char[vocab_size] = (char)c;
            vocab_size++;
        }
    }
    dl->vocab_size = vocab_size;

    // Tokenize
    dl->n_tokens = (int)file_size;
    dl->tokens   = new int[dl->n_tokens];
    for (int i = 0; i < dl->n_tokens; i++)
        dl->tokens[i] = dl->char_to_id[(unsigned char)text[i]];

    delete[] text;

    printf("dataloader: %d tokens, vocab_size=%d\n", dl->n_tokens, vocab_size);
}

// Samples B random chunks of length T+1, splits into token_ids [B,T] and labels [B,T].
// Allocates GPU tensors; caller is responsible for freeing them.
inline void dataloader_next_batch(DataLoader* dl, Tensor** token_ids_out, Tensor** labels_out) {
    int B = dl->B, T = dl->T;

    std::vector<float> tok_h(B * T);
    std::vector<float> lbl_h(B * T);

    for (int b = 0; b < B; b++) {
        // Random start position, leaving room for T+1 tokens
        int start = rand() % (dl->n_tokens - T - 1);
        for (int t = 0; t < T; t++) {
            tok_h[b * T + t] = (float)dl->tokens[start + t];
            lbl_h[b * T + t] = (float)dl->tokens[start + t + 1];
        }
    }

    int shape[] = {B, T};
    *token_ids_out = from_host(tok_h.data(), shape, 2);
    *labels_out    = from_host(lbl_h.data(), shape, 2);
}

inline void dataloader_free(DataLoader* dl) {
    delete[] dl->tokens;
}
