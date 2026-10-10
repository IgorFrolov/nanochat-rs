# Repository instructions

## Scope and purpose

These instructions apply to the whole repository. `nanochat-rs` is a single
Rust 2021 Cargo package with a library and CLI, using Candle for a small local
language-model pipeline. Keep the runtime independent of Python. Treat upstream
nanochat as a design reference, not a claim of checkpoint or numerical compatibility.

## Code map

- `src/main.rs`: Clap CLI, defaults, device selection, command dispatch.
- `src/config.rs`: model/training configuration and validation.
- `src/model/`: GPT, RMSNorm, attention/RoPE, MLP, and KV cache.
- `src/tokenizer.rs`, `src/bpe.rs`: tokenizer formats and conversation rendering.
- `src/dataloader.rs`, `src/dataset.rs`: random in-memory and sequential streaming input.
- `src/trainer.rs`, `src/sft.rs`, `src/loss.rs`: training loops and supervised masks.
- `src/optimizer.rs`, `src/checkpoint.rs`: AdamW state and native SafeTensors persistence.
- `src/inference.rs`, `src/sampling.rs`, `src/eval.rs`: generation and validation loss.
- `tests/components.rs`, `tests/regressions.rs`: component and CLI tests, including
  cache/generation parity, optimizer resume, dataset block boundaries, BPE checkpoint
  identity, file batching, pretrained SFT initialization, and SFT padding masks.
- `configs/`: example configurations; not loaded by a CLI config flag.
- `data/`: tracked small fixtures. `docs/`: implementation, compatibility, roadmap.

## Working conventions

Read the affected implementation before updating behavior or documentation.
Keep changes focused; follow existing Rust style and use `anyhow::Result` and
contextual errors at fallible boundaries. Keep reusable behavior in library
modules and argument handling in `main.rs`.

Preserve tensor shapes, causal attention, RoPE positions, KV-cache parity, and
assistant-only loss masking when changing model or SFT code. Padding must not
contribute to loss. Add meaningful regression coverage when changing these
behaviors. Preserve tokenizer IDs, checkpoint tensor names, and serialization
formats unless the task calls for a format change; document compatibility impact.

Do not overwrite existing user checkpoints or tracked data for smoke tests.
Use a fresh temporary directory for generated models/tokenizers. `target/` and
`checkpoints/` are ignored; keep generated weights and large corpora out of Git.

## Verification

For Rust changes, run `cargo fmt --all -- --check` and `cargo test --locked`.
Use focused component tests during iteration. Exercise the relevant CLI command
for changes to argument dispatch or checkpoint workflows, using tiny step counts
and temporary output paths. Report checks that could not run and why.
For documentation-only edits, verify statements and flags against source and
run `git diff --check`; a full training run is unnecessary.

The manifest currently enables Candle Metal features unconditionally. Runtime
CPU selection does not make the build independent of Apple tooling. Do not
silently change backend dependencies just to work around a local build failure.

## Current behavior to account for

- `train`, `sft`, and `bench` select CPU/Metal; CLI `chat` and `eval` use CPU.
- File pretraining honors batching and byte/BPE tokenization. `.u32` input must
  already use the selected tokenizer's IDs. Text loading buffers one line at a time.
- All training paths save tokenizer metadata with `kind: byte|bpe`; BPE checkpoints
  include their exact BPE artifact. Missing artifacts, stale artifacts on explicitly
  byte checkpoints, and vocabulary mismatches fail instead of falling back silently.
  Explicit tokenizers on resume/initialization must match saved contents.
- Resume restores model config, weights, AdamW moments, and step count, but not
  dataset cursor or RNG state. `--steps` cannot decrease; output `--checkpoint`
  and training `--seq-len` remain separate CLI arguments.
- SFT accepts `--init-from` for pretrained config/weights/tokenizer with a fresh
  optimizer. Otherwise it starts from random weights. SFT optimizer resume is
  not implemented. JSON/JSONL supports alternating user/assistant roles and
  right padding; all tokenizer artifacts are saved with the result.
- Evaluation loads the checkpoint tokenizer. `chat --conversation` uses SFT role
  formatting and retains history; default chat is independent text completion.
  Both return only new text and stop at assistant-end/EOS or the context limit.
  Overlong prompts/history are rejected; no automatic history trimming exists.
- Native checkpoints and BPE JSON are not upstream PyTorch/tiktoken artifacts.

When changing these boundaries, update this file and README.md. Update the
relevant files under `docs/` when architecture, compatibility, or roadmap status
changes; implemented behavior in source takes precedence over older roadmap text.
