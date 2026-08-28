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
- CPU/Metal inference, greedy/top-k/top-p sampling, naive KV cache, evaluation, and benchmarking.

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
cargo run -- sft --conversation data/example-conversations.jsonl --batch-size 2 --seq-len 128 --steps 100
cargo run -- tokenizer train --input data/train.txt --output data/tokenizer.json --vocab-size 1024
cargo run -- train --tokenizer data/tokenizer.json --text "hello rust hello rust" --seq-len 8 --steps 10 --checkpoint checkpoints/d4-bpe
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

Each checkpoint contains `config.json`, `model.safetensors`, `optimizer.safetensors`,
`tokenizer.json`, and `training_state.json`. BPE training additionally writes
`bpe-tokenizer.json`. `--resume` restores weights, AdamW moments, and optimizer
step for single-device pretraining; `--steps` is the desired total step count.
`chat` and `train --resume` auto-detect a matching BPE artifact. A stale BPE
artifact with a mismatched vocabulary is ignored with a warning. Streaming cursor
and RNG state are not restored yet.

`--data` enables bounded-memory sequential loading. Regular files are tokenized
line by line; files ending in `.u32` are read as little-endian `u32` token ids.
The streaming loader cycles at EOF, so a small local corpus can support a longer
development run without being copied into memory.
The library exposes assistant-only masked cross-entropy and typed conversation
rendering. SFT accepts one JSON conversation or a streaming JSONL file, one
conversation per line. The special-token names and order follow upstream, while
Python tool execution is intentionally not enabled.
`sft --batch-size N` right-pads each batch to its longest rendered conversation;
padding and user tokens are excluded from the loss. Because padding is on the
right, causal attention for real tokens never attends to a padded position.

## Current Limits

- The native byte-level BPE JSON artifact is not compatible with upstream
  `tokenizer.pkl` or tiktoken encodings.
- Upstream PyTorch checkpoints cannot be loaded.
- KV cache is correct by parity test but is intentionally simple and not optimized.
- SFT has no checkpoint resume, packed sequences, system messages, or tool execution.
- Evaluation reports validation cross-entropy only; BPB and upstream CORE tasks are pending.

[karpathy/nanochat]: https://github.com/karpathy/nanochat
