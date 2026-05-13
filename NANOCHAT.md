# nanochat support (this branch)

This branch (`nanochat`) of the llama.cpp fork adds support for Karpathy's
[nanochat](https://github.com/karpathy/nanochat) architecture, specifically the
[nanochat-d34](https://huggingface.co/karpathy/nanochat-d34) checkpoint.

For pre-converted GGUFs and full usage docs, see:

→ **https://huggingface.co/ulanch/nanochat-d34-GGUF**

## What's in the branch

One new file plus a handful of small edits on top of upstream:

```
src/llama-arch.h       +1   LLM_ARCH_NANOCHAT enum value
src/llama-arch.cpp     +1   { LLM_ARCH_NANOCHAT, "nanochat" } in the arch-name map
src/llama-vocab.h      +1   LLAMA_VOCAB_PRE_TYPE_NANOCHAT enum value
src/llama-vocab.cpp    +12  match "nanochat" → pre-type, plus the BPE split regex
src/models/models.h    +13  llama_model_nanochat forward declaration
src/llama-model.cpp    +3   dispatch + NEOX rope_type
src/models/nanochat.cpp +172 (new)  the actual model implementation
```

`master` on this fork is byte-identical to `ggml-org/llama.cpp` master; everything
above lives on `nanochat`.

## Build

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j 8 --target llama-cli llama-completion llama-server llama-quantize
```

Metal on Apple Silicon and AVX2/AVX-512 on x86 are auto-detected. `LLAMA_BUILD_SERVER=ON`
is required even for llama-cli — it's gated on the server build upstream.

## Converting your own nanochat checkpoint

A standalone converter (`convert_nanochat_to_gguf.py`) lives in the project workspace
alongside this fork. It reads `model_*.pt` + `meta_*.json` + `tokenizer.pkl` from a
nanochat checkpoint dir and writes a GGUF with `arch="nanochat"`. Default output is
bf16 — see the HF page for why fp16 is deprecated for this architecture.

```bash
python convert_nanochat_to_gguf.py --src /path/to/checkpoint --out model.gguf
./build/bin/llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M
```

## A couple of notes

- d34 was trained at nanochat commit `2c4473d` (Jan 11 2026). Current `master`
  of nanochat has diverged significantly — smear gates, value embeddings,
  residual lambdas. None of that is in d34. Don't try to match this code
  against the current `gpt.py`.
- The RoPE in this arch uses an inverted-sin convention vs ggml's NEOX. The
  graph compensates by passing `freq_scale = -1.0` to `ggml_rope_ext`. That's
  the only non-obvious thing in `nanochat.cpp`.

## License

MIT, inherited from upstream llama.cpp and from
[karpathy/nanochat](https://github.com/karpathy/nanochat).
