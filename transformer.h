#pragma once
#include "tensor.h"
#include <vector>

struct Block {
    Tensor* ln1_gamma;
    Tensor* ln1_beta;
    Tensor* W_q;
    Tensor* W_k;
    Tensor* W_v;
    Tensor* W_o;
    Tensor* ln2_gamma;
    Tensor* ln2_beta;
    Tensor* W_mlp_up;
    Tensor* W_mlp_down;
};

struct Transformer {
    Tensor* token_emb;
    Tensor* pos_emb;
    Block*  blocks;
    int     num_blocks;
    Tensor* final_ln_gamma;
    Tensor* final_ln_beta;
};

Transformer make_transformer(int vocab_size, int max_seq_len, int d_model, int n_heads, int n_layers);
std::vector<Tensor*> transformer_params(Transformer* model);
Tensor* transformer_forward(Transformer* model, Tensor* token_ids, int B, int T, int d_model, int n_heads, std::vector<Tensor*>& activations);
void save_checkpoint(Transformer* model, const char* path);
void load_checkpoint(Transformer* model, const char* path);