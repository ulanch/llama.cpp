"""
Convert a nanochat checkpoint to GGUF for llama.cpp.

Usage:
    python convert_nanochat_to_gguf.py --src ./checkpoint_dir --out model.gguf

Expects --src to contain:
    model_*.pt        (PyTorch state_dict)
    meta_*.json       (config blob with "model_config" subdict)
    tokenizer.pkl     (pickled tiktoken Encoding)

Depth, width, and vocab are read from meta_*.json, so the converter works
on any depth as long as the underlying architecture is the standard
nanochat transformer (parameterless RMS norm, QK-norm, ReLU² FFN no gate,
NEOX RoPE, untied lm_head, logit softcap=15).

Produces a GGUF with arch="nanochat". The matching llama.cpp side
(arch enum, vocab pre-type, models/nanochat.cpp) must be in place for the
file to load — see NANOCHAT.md.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import gguf


SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]


def bytes_to_unicode():
    """GPT-2 byte->unicode mapping. Identical to transformers.gpt2.tokenization_gpt2.bytes_to_unicode."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENCODER = bytes_to_unicode()


def token_bytes_to_string(b: bytes) -> str:
    return ''.join(BYTE_ENCODER[byte] for byte in b)


def bpe_split(mergeable_ranks: dict, token: bytes, max_rank: int) -> list:
    """Reverse-engineer the merge that produced `token` (with rank `max_rank`).
    Returns the two-piece split. Algorithm cribbed from llama.cpp QwenModel.bpe."""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, (a, b_) in enumerate(zip(parts[:-1], parts[1:])):
            r = mergeable_ranks.get(a + b_)
            if r is not None and (min_rank is None or r < min_rank):
                min_idx = i
                min_rank = r
        if min_rank is None or min_rank >= max_rank:
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def torch_to_numpy(t: torch.Tensor, dtype: str) -> np.ndarray:
    """Convert torch tensor to numpy with target dtype. bf16 returned as uint16 view."""
    t = t.detach()
    if dtype == "f32":
        return t.to(torch.float32).numpy()
    if dtype == "f16":
        return t.to(torch.float16).numpy()
    if dtype == "bf16":
        return t.to(torch.bfloat16).view(torch.uint16).numpy()
    raise ValueError(f"unknown dtype {dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Directory with model_*.pt, meta_*.json, tokenizer.pkl")
    ap.add_argument("--out", required=True, help="Output .gguf path")
    ap.add_argument("--outtype", default="bf16", choices=["f32", "bf16", "f16"],
                    help="Output tensor dtype. Default bf16 (recommended). "
                         "f16 risks overflow in deep checkpoints because the ReLU² FFN "
                         "can produce activations above the fp16 max. bf16 is the safe "
                         "half-precision choice for this architecture.")
    ap.add_argument("--name", default=None, help="general.name (default derived from src dir)")
    args = ap.parse_args()

    src = Path(args.src)
    assert src.is_dir(), f"--src must be a directory: {src}"

    if args.outtype == "f16":
        print("WARNING: --outtype f16 risks NaN logits on deeper checkpoints because "
              "the ReLU² FFN can produce activations above the fp16 max. "
              "Prefer --outtype bf16.")

    # ---- meta ----
    meta_files = sorted(src.glob("meta_*.json"))
    assert meta_files, f"no meta_*.json in {src}"
    meta = json.loads(meta_files[-1].read_text())
    cfg = meta["model_config"]
    print(f"meta: {meta_files[-1].name}  config: {cfg}")

    n_layer = cfg["n_layer"]
    n_head = cfg["n_head"]
    n_kv_head = cfg["n_kv_head"]
    n_embd = cfg["n_embd"]
    vocab_size = cfg["vocab_size"]
    seq_len = cfg["sequence_len"]
    head_dim = n_embd // n_head
    ffn_dim = 4 * n_embd
    assert n_embd % n_head == 0
    print(f"n_layer={n_layer} n_head={n_head} n_kv_head={n_kv_head} n_embd={n_embd} head_dim={head_dim} ffn={ffn_dim} vocab={vocab_size}")

    # ---- weights ----
    pt_files = sorted(src.glob("model_*.pt"))
    assert pt_files, f"no model_*.pt in {src}"
    print(f"loading {pt_files[-1].name}...")
    sd = torch.load(pt_files[-1], map_location="cpu", weights_only=False)
    print(f"  {len(sd)} tensors")

    padded_vocab = sd["lm_head.weight"].shape[0]
    assert padded_vocab >= vocab_size
    print(f"padded_vocab={padded_vocab}")

    # ---- tokenizer ----
    with open(src / "tokenizer.pkl", "rb") as f:
        enc = pickle.load(f)
    pat_str = enc._pat_str
    mergeable_ranks = enc._mergeable_ranks  # dict[bytes, int]
    special_tokens_map = enc._special_tokens  # dict[str, int]
    assert enc.n_vocab == vocab_size, f"tokenizer vocab {enc.n_vocab} != model vocab {vocab_size}"
    assert set(special_tokens_map.keys()) == set(SPECIAL_TOKENS), \
        f"unexpected specials: {set(special_tokens_map.keys()) ^ set(SPECIAL_TOKENS)}"

    # vocab list
    tokens = [None] * padded_vocab
    toktypes = [int(gguf.TokenType.NORMAL)] * padded_vocab
    for tok_bytes, rank in mergeable_ranks.items():
        tokens[rank] = token_bytes_to_string(tok_bytes)
    for name, idx in special_tokens_map.items():
        tokens[idx] = name
        toktypes[idx] = int(gguf.TokenType.CONTROL)
    for i in range(padded_vocab):
        if tokens[i] is None:
            tokens[i] = f"[PAD{i}]"
            toktypes[i] = int(gguf.TokenType.UNUSED)

    # merges (sorted by rank ascending)
    print("reverse-engineering BPE merges...")
    merges = []
    items = sorted(mergeable_ranks.items(), key=lambda kv: kv[1])
    for tok_bytes, rank in items:
        if len(tok_bytes) == 1:
            continue
        parts = bpe_split(mergeable_ranks, tok_bytes, rank)
        if len(parts) != 2:
            raise RuntimeError(f"bad split for rank {rank} {tok_bytes!r}: {parts}")
        merges.append(' '.join(token_bytes_to_string(p) for p in parts))
    print(f"  {len(merges)} merges")

    # ---- writer ----
    arch = "nanochat"
    name = args.name or src.name or "nanochat"
    writer = gguf.GGUFWriter(args.out, arch)
    writer.add_name(name)

    # hparams (using llama-style standard keys; llama.cpp side will read with arch="nanochat")
    writer.add_context_length(seq_len)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(ffn_dim)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_kv_head)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(10000.0)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32   if args.outtype == "f32"  else
                         gguf.LlamaFileType.MOSTLY_BF16 if args.outtype == "bf16" else
                         gguf.LlamaFileType.MOSTLY_F16)
    # logit softcap (same convention as gemma2)
    writer.add_final_logit_softcapping(15.0)

    # tokenizer
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("nanochat")  # we'll register this pre name in llama.cpp
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(special_tokens_map["<|bos|>"])
    writer.add_eos_token_id(special_tokens_map["<|assistant_end|>"])
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)

    # chat template (Jinja). Mirrors nanochat's render_conversation for plain text content.
    chat_template = (
        "{%- for message in messages -%}"
        "{%- if message['role'] == 'user' -%}"
        "<|user_start|>{{ message['content'] }}<|user_end|>"
        "{%- elif message['role'] == 'assistant' -%}"
        "<|assistant_start|>{{ message['content'] }}<|assistant_end|>"
        "{%- elif message['role'] == 'system' -%}"
        "<|user_start|>{{ message['content'] }}<|user_end|>"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|assistant_start|>{%- endif -%}"
    )
    writer.add_chat_template(chat_template)

    # ---- tensors ----
    def add(name, t, force_f32=False):
        if force_f32:
            arr = t.to(torch.float32).numpy()
            writer.add_tensor(name, arr)
        elif args.outtype == "bf16":
            arr = t.to(torch.bfloat16).view(torch.uint16).numpy()
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            writer.add_tensor(name, torch_to_numpy(t, args.outtype))

    print("writing tensors...")
    add("token_embd.weight", sd["transformer.wte.weight"])
    add("token_embd_norm.weight", torch.ones(n_embd), force_f32=True)
    add("output.weight", sd["lm_head.weight"])
    add("output_norm.weight", torch.ones(n_embd), force_f32=True)

    for i in range(n_layer):
        prefix = f"transformer.h.{i}"
        b = f"blk.{i}"
        add(f"{b}.attn_norm.weight",   torch.ones(n_embd),  force_f32=True)
        add(f"{b}.attn_q.weight",      sd[f"{prefix}.attn.c_q.weight"])
        add(f"{b}.attn_k.weight",      sd[f"{prefix}.attn.c_k.weight"])
        add(f"{b}.attn_v.weight",      sd[f"{prefix}.attn.c_v.weight"])
        add(f"{b}.attn_q_norm.weight", torch.ones(head_dim), force_f32=True)
        add(f"{b}.attn_k_norm.weight", torch.ones(head_dim), force_f32=True)
        add(f"{b}.attn_output.weight", sd[f"{prefix}.attn.c_proj.weight"])
        add(f"{b}.ffn_norm.weight",    torch.ones(n_embd),  force_f32=True)
        add(f"{b}.ffn_up.weight",      sd[f"{prefix}.mlp.c_fc.weight"])
        add(f"{b}.ffn_down.weight",    sd[f"{prefix}.mlp.c_proj.weight"])

    print(f"writing GGUF -> {args.out}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("done.")


if __name__ == "__main__":
    main()
