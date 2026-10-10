# nanochat-rs

`nanochat-rs` is an offline-first Rust/Candle implementation of a small, local
language-model pipeline inspired by [karpathy/nanochat]. It is not a line-by-line
translation and does not load Python at runtime.

## Current milestone

The current development milestone contains:

- RMSNorm decoder Transformer with RoPE, QK normalization, causal attention, GQA-ready projections, and `ReLU^2` MLP;
- AdamW training with inspectable model and optimizer SafeTensors;
- single-device pretraining/SFT resume with data cursors and shuffle RNG state;
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
- `src/optimizer.rs`, `src/checkpoint.rs`, `src/resume.rs`: AdamW, atomic checkpoint
  publication, and versioned training/data state.
- `src/inference.rs`, `src/sampling.rs`, `src/eval.rs`: generation and evaluation.
- `tests/components.rs`, `tests/regressions.rs`, `tests/reliability.rs`: component,
  numerical parity, exact resume, publication failure, process interruption, and CLI tests.
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

Training writes immutable generations under `CHECKPOINT/snapshots/<generation>/`
and atomically replaces the small `CHECKPOINT/CURRENT` pointer only after all
files have been written and synced. Each generation contains `config.json`, `model.safetensors`,
`optimizer.safetensors`, `tokenizer.json`, and `training_state.json`. All BPE
training paths, including SFT and file training, also save `bpe-tokenizer.json`.
New `tokenizer.json` metadata identifies the tokenizer with `kind: byte|bpe`.
Byte generations do not contain a BPE artifact. CLI commands accept the checkpoint
root or a specific generation directory; library readers can use
`checkpoint::resolve` to pin a generation before reading several files.

Previous generations are retained so active readers and interrupted saves remain
safe. Disk usage grows with each save; automatic retention/garbage collection is
not implemented. Never delete a generation still referenced by `CURRENT` or an
active reader. Concurrent writers publish complete generations, with the last
publication winning; use a separate output directory for each training run.

Resume, SFT initialization, chat, and eval load the same checkpoint tokenizer.
Missing required BPE artifacts and vocabulary mismatches are errors. An explicit
`--tokenizer` on resume or SFT initialization must match the saved tokenizer's
contents, not merely its vocabulary size. Legacy checkpoints without `kind`
are inferred from the presence of the BPE artifact and validated against model
vocabulary size. Legacy BPE-SFT checkpoints that omitted that artifact must be
repaired with their original tokenizer; its identity cannot always be inferred
from vocabulary size alone.

`train --resume` and `sft --resume` restore weights, AdamW moments/settings/step,
and data progress. Inline pretraining restores the shuffled window order, cursor,
and ChaCha12 RNG. File pretraining restores the byte offset plus pending tokens;
JSONL SFT restores its next-line offset. The model currently has no stochastic
training layers, so the data shuffler is the only RNG consumed after initialization.

`--steps` is the desired total count and cannot decrease. On resume, omitted
`--batch-size` and `--seq-len` inherit saved values; explicitly changing them is
rejected. Supply the original `--text`, `--data`, or `--conversation` again.
A streaming SHA-256 check rejects changed data content or format; files may be
moved without invalidating resume. Data files must remain unchanged during a run:
their hash is rechecked before each save, which adds a full sequential read.
The saved backend is required; CPU exactness is tested, while bitwise equivalence
across hardware, dependency versions, or Metal executions is not promised.

Use `--checkpoint-every N` for periodic recovery points. The default `0` saves only
at completion. A killed process can lose updates since the latest published
checkpoint. Set `--checkpoint` explicitly for the output directory; it does not
default to the resume path.

```sh
cargo run -- train --data data/train.txt --seq-len 128 --steps 100 --checkpoint-every 20 --checkpoint checkpoints/pretrain
cargo run -- train --resume checkpoints/pretrain --data data/train.txt --steps 200 --checkpoint-every 20 --checkpoint checkpoints/pretrain
cargo run -- sft --init-from checkpoints/pretrain --conversation data/example-conversations.jsonl --seq-len 128 --batch-size 2 --steps 100 --checkpoint-every 20 --checkpoint checkpoints/sft
cargo run -- sft --resume checkpoints/sft --conversation data/example-conversations.jsonl --steps 200 --checkpoint-every 20 --checkpoint checkpoints/sft
```

Legacy flat checkpoint directories remain readable. Pretraining resume from a
legacy checkpoint without data state emits a warning and restarts the data
sequence, so it is not exact continuation. Legacy SFT checkpoints require
`--init-from` (fresh optimizer); `sft --resume` requires the new versioned state.
A save into a legacy directory publishes a generation and leaves the old flat
files intact; subsequent reads follow `CURRENT`.

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
starts from random weights. Use `sft --resume DIR` to continue SFT with its saved
optimizer and data position. `--init-from` and `--resume` are mutually exclusive.
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
- SFT has no packed sequences, system messages, or tool execution.
- Evaluation has no BPB or upstream CORE tasks yet.
- Conversation history must fit the context; automatic history trimming and
  packed sequences are not implemented.
- Distributed training, CUDA, mixed precision, and upstream checkpoint conversion
  are not implemented.

[karpathy/nanochat]: https://github.com/karpathy/nanochat
