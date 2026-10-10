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
- CPU/Metal model execution, greedy/top-k/top-p sampling, naive KV cache, CPU chat/evaluation, and benchmarking.

The Rust BPE artifact is deliberately standalone. It is not compatible with the
upstream Python `tokenizer.pkl`/tiktoken artifact yet.

## Build and quick start

Use a recent stable Rust toolchain. The current `Cargo.toml` enables Candle's
Metal features unconditionally, so the build is configured for macOS with the
Apple development tools and Metal toolchain available. Selecting CPU at runtime
does not disable those build dependencies; a portable CPU-only build is not yet
exposed as a Cargo feature. Python is not required. The first Cargo build may
need network access to download dependencies; model training uses local data.

These examples train tiny development models, not a pretrained conversational
assistant. Use a new checkpoint directory if you want to preserve previous runs.

```sh
cargo test --locked
cargo run -- model info
cargo run -- train --depth 4 --seq-len 128 --steps 100
printf 'hello\n' | cargo run -- chat --checkpoint checkpoints/d4 --temperature 0 --top-p 0.9
cargo run -- eval --checkpoint checkpoints/d4 --steps 10
cargo run -- bench --depth 4 --seq-len 128 --batch-size 1 --steps 10
cargo run -- train --data data/train.txt --seq-len 128 --steps 100
cargo run -- sft --init-from checkpoints/d4 --conversation data/example-conversation.json --seq-len 128 --steps 100
printf 'What is Rust?\n' | cargo run -- chat --checkpoint checkpoints/sft --conversation --max-new-tokens 32
cargo run -- sft --conversation data/example-conversations.jsonl --batch-size 2 --seq-len 128 --steps 100
mkdir -p checkpoints/bpe-artifacts
cargo run -- tokenizer train --input data/train.txt --output checkpoints/bpe-artifacts/tokenizer.json --vocab-size 1024
cargo run -- train --tokenizer checkpoints/bpe-artifacts/tokenizer.json --text "hello rust hello rust" --seq-len 8 --steps 10 --checkpoint checkpoints/d4-bpe
cargo run -- train --resume checkpoints/d4 --checkpoint checkpoints/d4 --steps 200 --seq-len 128
```

CPU is the default runtime device. `train`, `sft`, and `bench` accept
`--device cpu|mps|metal|auto`; Metal requests fall back to CPU with a warning
when unavailable. The `chat` and `eval` CLI commands always use CPU. CUDA is not
a dependency of this package. Run `cargo run -- <command> --help` for options.

## Layout and roadmap

The model, tokenizer, loader, trainer, checkpoint, loss, SFT, and inference code
are library modules under `src/`.
The implementation boundary and upstream comparison are documented in
[implementation map](docs/IMPLEMENTATION.md) and
[compatibility notes](docs/COMPATIBILITY.md); planned work is in the
[roadmap](docs/ROADMAP.md).

- `src/main.rs`: CLI arguments and backend selection.
- `src/model/`: Transformer blocks, attention, RoPE, and KV cache.
- `src/tokenizer.rs`, `src/bpe.rs`: byte and BPE tokenizers, conversation rendering.
- `src/dataloader.rs`, `src/dataset.rs`: in-memory and streaming data loaders.
- `src/trainer.rs`, `src/sft.rs`, `src/loss.rs`: pretraining and supervised training.
- `src/optimizer.rs`, `src/checkpoint.rs`: AdamW and checkpoint persistence.
- `src/inference.rs`, `src/sampling.rs`, `src/eval.rs`: generation and evaluation.
- `tests/components.rs`, `tests/regressions.rs`: component, numerical parity,
  dataset boundary, optimizer resume, and CLI workflow tests.
- `data/`: small example corpora and conversations.
- `configs/`: reference model JSON files; the CLI has no `--config` option.

Contributor instructions are in [AGENTS.md](AGENTS.md).

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

Each training checkpoint contains `config.json`, `model.safetensors`,
`optimizer.safetensors`, `tokenizer.json`, and `training_state.json`. All BPE
training paths, including SFT and file training, also save `bpe-tokenizer.json`.
New `tokenizer.json` metadata identifies the tokenizer with `kind: byte|bpe`.
Byte checkpoints remove any old BPE artifact when saved.

Resume, SFT initialization, chat, and eval load the same checkpoint tokenizer.
Missing required BPE artifacts and vocabulary mismatches are errors. An explicit
`--tokenizer` on resume or SFT initialization must match the saved tokenizer's
contents, not merely its vocabulary size. Legacy checkpoints without `kind`
are inferred from the presence of the BPE artifact and validated against model
vocabulary size. Legacy BPE-SFT checkpoints that omitted that artifact must be
repaired with their original tokenizer; its identity cannot always be inferred
from vocabulary size alone.

`train --resume` restores weights, AdamW moments, and optimizer step;
`--steps` is the desired total step count and cannot be less than the saved step.
Streaming cursor and RNG state are not restored, so this is not exact replay of
an uninterrupted data sequence. Resume uses the saved model configuration, while
`--seq-len` controls training windows and must fit the saved context length.
Set `--checkpoint` explicitly to choose the output directory; it does not default
to the resume path.

`--data` enables sequential loading and honors `--batch-size` and the selected
byte/BPE tokenizer. Text files are tokenized line by line; `.u32` files contain
little-endian token IDs already encoded with that same tokenizer. IDs must be
below its vocabulary size. The loader cycles at EOF; buffering is proportional
to the current text line, training window, and batch rather than the whole file.
BPE tokenizer training itself reads its entire input corpus into memory.
Empty inline training/evaluation text, zero batch sizes, and invalid training
sequence lengths are rejected.

## Supervised training

The library exposes assistant-only masked cross-entropy and typed conversation
rendering. SFT accepts one JSON conversation or a streaming JSONL file, one
conversation per line. The special-token names and order follow upstream, while
Python tool execution is intentionally not enabled.
`sft --batch-size N` right-pads each batch to its longest rendered conversation;
padding and user tokens are excluded from the loss. Because padding is on the
right, causal attention for real tokens never attends to a padded position.

Conversation input has this shape (JSONL uses one such object per line):

```json
{"messages":[{"role":"user","content":"What is Rust?"},{"role":"assistant","content":"A systems programming language."}]}
```

Roles must alternate `user` and `assistant`, starting with `user`. Rendering is
truncated to `--seq-len + 1`; training rejects examples without supervised
assistant tokens in that window. JSONL input cycles at EOF.

`sft --init-from DIR` loads pretrained model configuration, weights, and tokenizer,
then starts a fresh AdamW optimizer and SFT step count. Without `--init-from`, SFT
starts from random weights. This is initialization, not SFT optimizer resume.
Use a separate `--checkpoint` directory to preserve the pretrained model.

## Generation and evaluation

`chat --conversation` renders the same user/assistant markers as SFT, appends an
assistant-start token, and retains history across input lines. Without the flag,
each line is an independent plain-text completion prompt. Both modes print only
newly generated text. Generation stops on assistant-end (or byte-tokenizer EOS)
and is bounded by the saved context length. A prompt/history that leaves no room
for generation is rejected; history is not silently truncated. CLI sampling uses
top-k 50 and seed 42; temperature 0 selects greedy decoding.

`eval` uses the checkpoint's byte/BPE tokenizer and reports cross-entropy on
`--text` (a built-in toy string by default). Supply held-out text explicitly for
meaningful validation. `bench` measures forward passes on a newly initialized
model, not checkpoint-backed generation throughput.

## Current Limits

- The native byte-level BPE JSON artifact is not compatible with upstream
  `tokenizer.pkl` or tiktoken encodings.
- Upstream PyTorch checkpoints cannot be loaded.
- KV cache is correct by parity test but is intentionally simple and not optimized.
- SFT has no checkpoint resume, packed sequences, system messages, or tool execution.
- Evaluation has no BPB or upstream CORE tasks yet.
- Conversation history must fit the context; automatic history trimming and
  packed sequences are not implemented.
- Distributed training, CUDA, mixed precision, and upstream checkpoint conversion
  are not implemented.

[karpathy/nanochat]: https://github.com/karpathy/nanochat
