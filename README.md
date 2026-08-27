# nanochat-rs

`nanochat-rs` is an offline-first Rust/Candle implementation of a small, local
language-model pipeline inspired by [karpathy/nanochat]. It is not a line-by-line
translation and does not load Python at runtime.

## Current milestone

The current development milestone contains:

- RMSNorm decoder Transformer with RoPE, QK normalization, causal attention, GQA-ready projections, and `ReLU^2` MLP;
- AdamW training with inspectable model and optimizer SafeTensors;
- single-device checkpoint resume;
- byte tokenizer and standalone byte-level BPE tokenizer;
- bounded-memory plain-text and `.u32` dataset loading;
- masked assistant-only loss and JSON/JSONL SFT training;
- CPU/Metal inference, top-k/top-p sampling, naive KV cache, evaluation, and benchmarking.

The Rust BPE artifact is deliberately standalone. It is not compatible with the
upstream Python `tokenizer.pkl`/tiktoken artifact yet.

```text
cargo test
cargo run -- model info
cargo run -- train --depth 4 --seq-len 32 --steps 100
printf 'hello\n' | cargo run -- chat --checkpoint checkpoints/d4 --temperature 0 --top-p 0.9
cargo run -- eval --checkpoint checkpoints/d4 --steps 10
cargo run -- bench --depth 4 --seq-len 128 --batch-size 1 --steps 10
cargo run -- train --data data/train.txt --seq-len 128 --steps 100
cargo run -- sft --conversation data/example-conversation.json --seq-len 128 --steps 100
cargo run -- tokenizer train --input data/train.txt --output data/tokenizer.json --vocab-size 1024
cargo run -- train --tokenizer data/tokenizer.json --text "hello rust hello rust" --seq-len 8 --steps 10
cargo run -- train --resume checkpoints/d4 --steps 200 --seq-len 128
```

CPU is the portable default. `--device mps` requests Candle Metal and falls back
to CPU with a warning when unavailable. CUDA is not a dependency of this package.

## Layout and roadmap

The model, tokenizer, loader, trainer, checkpoint, loss, SFT, and inference code
are library modules under `src/`.
The implementation boundary and upstream comparison are documented in
`docs/IMPLEMENTATION.md` and `docs/COMPATIBILITY.md`.

## CLI

| Command | Purpose |
|---|---|
| `model info` | Show the default tiny model configuration |
| `tokenizer inspect` | Show the development tokenizer |
| `tokenizer train` | Train and save standalone byte-level BPE JSON |
| `train` | Pretrain on inline text, plain text, or `.u32` data |
| `sft` | Train on JSON or streaming JSONL conversations |
| `eval` | Calculate validation cross-entropy |
| `bench` | Measure forward latency and tokens/sec |
| `chat` | Run local checkpoint-backed generation |

## Checkpoints

Checkpoints contain `config.json`, `model.safetensors`, `optimizer.safetensors`,
`tokenizer.json`, and `training_state.json`. BPE pretraining additionally writes
`bpe-tokenizer.json`. `--resume` restores model weights, AdamW moments, and the
optimizer step for single-device training. Streaming dataset cursor and RNG state
are not restored yet.
Training also writes `optimizer.safetensors`; resume with `--resume <checkpoint>`
and set `--steps` to the desired total step count.
Checkpoints produced with `--tokenizer` also include `bpe-tokenizer.json`; `chat`
detects and uses it automatically.

`--data` enables bounded-memory sequential loading. Regular files are tokenized
line by line; files ending in `.u32` are read as little-endian `u32` token ids.
The library exposes assistant-only masked cross-entropy and typed conversation
rendering. SFT accepts one JSON conversation or a streaming JSONL file, one
conversation per line. The special-token names and order follow upstream, while
Python tool execution is intentionally not enabled.

[karpathy/nanochat]: https://github.com/karpathy/nanochat
