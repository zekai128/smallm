#include "transformer.h"
#include "tensor.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>

static float EPS = 1e-5;

// Allocate a parameter tensor with Xavier uniform init, shape [dims...], requires_grad=true
static Tensor* make_param(std::vector<int> shape) {
    int size = 1;
    for (int d : shape) size *= d;

    int fan_in  = shape[shape.size() - 2];
    int fan_out = shape[shape.size() - 1];
    float limit = sqrtf(6.0f / (fan_in + fan_out));

    float* h = new float[size];
    for (int i = 0; i < size; i++)
        h[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;

    Tensor* t = from_host(h, shape.data(), shape.size());
    t->requires_grad = true;
    delete[] h;
    return t;
}

// Allocate a 1D parameter tensor filled with a constant value
static Tensor* make_param_constant(int size, float val) {
    float* h = new float[size];
    for (int i = 0; i < size; i++) h[i] = val;
    int shape[] = {size};
    Tensor* t = from_host(h, shape, 1);
    t->requires_grad = true;
    delete[] h;
    return t;
}

Transformer make_transformer(int vocab_size, int max_seq_len, int d_model, int n_heads, int n_layers) {
    Transformer model;

    model.token_emb = make_param({vocab_size, d_model});
    model.pos_emb   = make_param({max_seq_len, d_model});

    model.num_blocks = n_layers;
    model.blocks     = new Block[n_layers];

    for (int l = 0; l < n_layers; l++) {
        Block& b = model.blocks[l];

        b.ln1_gamma = make_param_constant(d_model, 1.0f);
        b.ln1_beta  = make_param_constant(d_model, 0.0f);

        // [1, d_model, d_model] — batch dim 1 for matmul ndim compatibility
        b.W_q = make_param({d_model, d_model});
        b.W_k = make_param({d_model, d_model});
        b.W_v = make_param({d_model, d_model});
        b.W_o = make_param({d_model, d_model});

        b.ln2_gamma = make_param_constant(d_model, 1.0f);
        b.ln2_beta  = make_param_constant(d_model, 0.0f);

        b.W_mlp_up   = make_param({1, d_model, 4 * d_model});
        b.W_mlp_down = make_param({1, 4 * d_model, d_model});
    }

    model.final_ln_gamma = make_param_constant(d_model, 1.0f);
    model.final_ln_beta  = make_param_constant(d_model, 0.0f);

    return model;
}

std::vector<Tensor*> transformer_params(Transformer* model) {
    std::vector<Tensor*> params;

    params.push_back(model->token_emb);
    params.push_back(model->pos_emb);

    for (int l = 0; l < model->num_blocks; l++) {
        Block& b = model->blocks[l];
        params.push_back(b.ln1_gamma);
        params.push_back(b.ln1_beta);
        params.push_back(b.W_q);
        params.push_back(b.W_k);
        params.push_back(b.W_v);
        params.push_back(b.W_o);
        params.push_back(b.ln2_gamma);
        params.push_back(b.ln2_beta);
        params.push_back(b.W_mlp_up);
        params.push_back(b.W_mlp_down);
    }

    params.push_back(model->final_ln_gamma);
    params.push_back(model->final_ln_beta);

    return params;
}

Tensor* transformer_forward(Transformer* model, Tensor* token_ids, int B, int T, int d_model, int n_heads, std::vector<Tensor*>& activations) {
    // 1. token and pos embedding 
    Tensor* token_embedding = embedding(model->token_emb, token_ids);
    activations.push_back(token_embedding);

    // build position indices [B, T] = [[0,1,...,T-1], [0,1,...,T-1], ...]
    std::vector<float> pos_h(B * T);
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            pos_h[b * T + t] = (float)t;
    int pos_shape[] = {B, T};
    Tensor* pos_ids = from_host(pos_h.data(), pos_shape, 2);

    activations.push_back(pos_ids);
    Tensor* pos_embedding = embedding(model->pos_emb, pos_ids);
    activations.push_back(pos_embedding);

    Tensor* summed_embeddings = add(token_embedding, pos_embedding);
    activations.push_back(summed_embeddings);


    Tensor* input_to_block = summed_embeddings;

    for (int i = 0; i < model->num_blocks; i++) {
        Block& b = model->blocks[i];
        Tensor* ln_block_input = layer_norm(input_to_block, b.ln1_gamma, b.ln1_beta, EPS);
        activations.push_back(ln_block_input);
       
        int shape_2d[] = {B*T, d_model};
        Tensor* ln_block_input_2D = reshape(ln_block_input, shape_2d, 2);

        Tensor* Q = matmul(ln_block_input_2D, b.W_q);
        activations.push_back(Q);

        Tensor* K = matmul(ln_block_input_2D, b.W_k);
        activations.push_back(K);

        Tensor* V = matmul(ln_block_input_2D, b.W_v);
        activations.push_back(V);


        // reshape Q,K,V
        int new_shape[4] = {B, T, n_heads, d_model/n_heads};

        // (B, T, n_heads, head_dim)
        Tensor* Q_reshaped = reshape(Q, new_shape, 4);
        activations.push_back(Q_reshaped);

        Tensor* K_reshaped = reshape(K, new_shape, 4);
        activations.push_back(K_reshaped);

        Tensor* V_reshaped = reshape(V, new_shape, 4);
        activations.push_back(V_reshaped);
    
        // q_permuted: (B, n_heads, T, head_dim)
        int q_order[4] = {0, 2, 1, 3};

        // k_permuted: (B, n_heads, head_dim, T)
        int k_order[4] = {0, 2, 3, 1};

        Tensor* Q_permuted = permute(Q_reshaped, q_order);
        activations.push_back(Q_permuted);
        Q_permuted = contiguous(Q_permuted);
        activations.push_back(Q_permuted);

        Tensor* K_permuted = permute(K_reshaped, k_order);
        activations.push_back(K_permuted);
        K_permuted = contiguous(K_permuted);
        activations.push_back(K_permuted);

        Tensor* V_permuted = permute(V_reshaped, q_order);
        activations.push_back(V_permuted);
        V_permuted = contiguous(V_permuted);
        activations.push_back(V_permuted);

        Tensor* attn_weights = matmul(Q_permuted, K_permuted);
        activations.push_back(attn_weights);

        float d_head = (float)(d_model / n_heads);
        Tensor* attn_weights_scaled = scale(attn_weights, 1.0f / sqrtf(d_head));
        activations.push_back(attn_weights_scaled);

        // causal mask [B, n_heads, T, T] — 1.0 where j > i (future), 0.0 otherwise
        int mask_shape[] = {B, n_heads, T, T};
        std::vector<float> mask_h(B * n_heads * T * T, 0.0f);
        for (int b = 0; b < B; b++)
            for (int h = 0; h < n_heads; h++)
                for (int i = 0; i < T; i++)
                    for (int j = i + 1; j < T; j++)
                        mask_h[((b * n_heads + h) * T + i) * T + j] = 1.0f;
        Tensor* causal_mask = from_host(mask_h.data(), mask_shape, 4);
        activations.push_back(causal_mask);

        Tensor* attn_weights_masked = masked_fill(attn_weights_scaled, causal_mask, -1e9f);
        activations.push_back(attn_weights_masked);

        Tensor* attn_probs = softmax(attn_weights_masked);
        activations.push_back(attn_probs);

        // (B, n_heads, T, d_head)
        Tensor* attn_values = matmul(attn_probs, V_permuted);
        activations.push_back(attn_values);

        // attn_values_permuted: (B, T, n_heads, head_dim)
        int attn_v_order[4] = {0, 2, 1, 3};
        Tensor* attn_values_permuted = permute(attn_values, attn_v_order);
        activations.push_back(attn_values_permuted);
        attn_values_permuted = contiguous(attn_values_permuted);
        activations.push_back(attn_values_permuted);

        // attn_values_reshaped: (B, T, d_model)
        int attn_v_reshape[3] = {B, T, d_model};
        Tensor* attn_values_reshaped = reshape(attn_values_permuted, attn_v_reshape, 3);
        activations.push_back(attn_values_reshaped);

        // W_o projection: [B*T, d_model] @ [d_model, d_model] -> [B*T, d_model]
        Tensor* attn_out_2d = reshape(attn_values_reshaped, shape_2d, 2);
        Tensor* attn_out_proj = matmul(attn_out_2d, b.W_o);
        activations.push_back(attn_out_proj);

        // reshape back to [B, T, d_model]
        int out_shape_3d[] = {B, T, d_model};
        Tensor* attn_out = reshape(attn_out_proj, out_shape_3d, 3);
        activations.push_back(attn_out);

        // first residual add
        Tensor* x = add(input_to_block, attn_out);
        activations.push_back(x);

        // ln2 + MLP
        Tensor* ln2 = layer_norm(x, b.ln2_gamma, b.ln2_beta, EPS);
        activations.push_back(ln2);

        // MLP up: [B*T, d_model] @ [d_model, 4*d_model] -> [B*T, 4*d_model]
        int mlp_shape_2d[] = {B*T, d_model};
        Tensor* ln2_2d = reshape(ln2, mlp_shape_2d, 2);

        // W_mlp_up is [1, d_model, 4*d_model] — reshape to 2D for matmul
        int w_up_shape[] = {d_model, 4*d_model};
        Tensor* W_up_2d = reshape(b.W_mlp_up, w_up_shape, 2);

        Tensor* mlp_up = matmul(ln2_2d, W_up_2d);
        activations.push_back(mlp_up);

        Tensor* mlp_act = gelu(mlp_up);
        activations.push_back(mlp_act);

        // W_mlp_down is [1, 4*d_model, d_model] — reshape to 2D
        int w_down_shape[] = {4*d_model, d_model};
        Tensor* W_down_2d = reshape(b.W_mlp_down, w_down_shape, 2);

        Tensor* mlp_down = matmul(mlp_act, W_down_2d);
        activations.push_back(mlp_down);

        // reshape MLP output back to [B, T, d_model]
        Tensor* mlp_out = reshape(mlp_down, out_shape_3d, 3);
        activations.push_back(mlp_out);

        // second residual add -> input to next block
        input_to_block = add(x, mlp_out);
        activations.push_back(input_to_block);

    }

    // final layer norm: [B, T, d_model]
    Tensor* final_ln = layer_norm(input_to_block, model->final_ln_gamma, model->final_ln_beta, EPS);
    activations.push_back(final_ln);

    // unembedding: [B*T, d_model] @ [d_model, V] -> [B*T, V]
    // token_emb is [V, d_model] — permute to [d_model, V]
    int emb_order[] = {1, 0};
    Tensor* token_emb_T = permute(model->token_emb, emb_order);
    activations.push_back(token_emb_T);
    token_emb_T = contiguous(token_emb_T);
    activations.push_back(token_emb_T);

    int final_2d[] = {B*T, d_model};
    Tensor* final_ln_2d = reshape(final_ln, final_2d, 2);

    Tensor* logits = matmul(final_ln_2d, token_emb_T); // [B*T, V]
    activations.push_back(logits);

    // softmax over vocab dim -> probs [B*T, V]
    Tensor* probs = softmax(logits);
    activations.push_back(probs);

    return probs;
}

void save_checkpoint(Transformer* model, const char* path) {
    std::vector<Tensor*> params = transformer_params(model);

    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s for writing\n", path); return; }

    for (Tensor* p : params) {
        float* h = new float[p->size];
        to_host(p, h);
        fwrite(h, sizeof(float), p->size, f);
        delete[] h;
    }

    fclose(f);
    printf("checkpoint saved to %s\n", path);
}

void load_checkpoint(Transformer* model, const char* path) {
    std::vector<Tensor*> params = transformer_params(model);

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s for reading\n", path); return; }

    for (Tensor* p : params) {
        float* h = new float[p->size];
        fread(h, sizeof(float), p->size, f);
        cudaMemcpy(p->data, h, p->size * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h;
    }

    fclose(f);
    printf("checkpoint loaded from %s\n", path);
}
