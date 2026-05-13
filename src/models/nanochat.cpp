#include "models.h"

// nanochat (Karpathy) — d34-era architecture (commit 2c4473d on master).
//
// Distinguishing features vs. the Llama family:
//   - parameterless RMS norm everywhere (norm tensors emitted as all-ones by the converter)
//   - post-embedding norm applied before the first block
//   - QK-norm (parameterless RMSnorm on Q and K after RoPE)
//   - ReLU^2 FFN with no gate (4x expansion)
//   - logit softcap = 15 at the output (same shape as Gemma2)
//   - NEOX-style RoPE (rotate-half, base=10000)
//   - untied lm_head; vocab=65536; MHA (n_head == n_kv_head)
//
// The graph below mirrors `forward()` in nanochat/gpt.py @ commit 2c4473d.

void llama_model_nanochat::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    switch (hparams.n_layer) {
        case 12: type = LLM_TYPE_335M; break; // d12 (~$100 model, vocab=65536)
        case 20: type = LLM_TYPE_770M; break; // d20 (~$300 model)
        case 26: type = LLM_TYPE_1B;   break; // d26 (~$1000 model)
        case 34: type = LLM_TYPE_2B;   break; // d34 (~$2500 model)
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_nanochat::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab}, 0);
    // post-embedding norm (parameterless on the python side, emitted as all-ones)
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd},          0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // parameterless QK-norm — converter writes all-ones tensors of length head_dim
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd},        0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff},  0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff,   n_embd}, 0);
        // no ffn_gate — nanochat MLP is up -> ReLU^2 -> down
    }
}

std::unique_ptr<llm_graph_context> llama_model_nanochat::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_nanochat::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // post-embedding RMS norm (nanochat: x = norm(wte(idx)))
    inpL = build_norm(inpL, model.tok_norm, NULL, LLM_NORM_RMS, -1);
    cb(inpL, "tok_norm", -1);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // pre-attn norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // RoPE first, then QK-norm — matches nanochat order.
            //
            // nanochat's apply_rotary_emb uses an inverted-sin convention:
            //   y1 =  x1*cos + x2*sin     (vs ggml NEOX:  y1 = x1*cos - x2*sin)
            //   y2 = -x1*sin + x2*cos     (vs ggml NEOX:  y2 = x1*sin + x2*cos)
            // i.e. it rotates by -theta. We compensate with freq_scale = -1, which
            // gives sin(-theta) = -sin(theta) while leaving cos unchanged. The model
            // weights were trained with this convention, so we MUST match it.
            const float nano_freq_scale = -1.0f;
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, nano_freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, nano_freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // pre-ffn norm
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // ReLU^2 FFN, no gate
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                NULL,                       NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    // logit softcap: y = softcap * tanh(x / softcap), with softcap = 15 (loaded from gguf)
    if (hparams.f_final_logit_softcapping > 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh (ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
