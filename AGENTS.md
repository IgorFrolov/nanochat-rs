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
- `src/optimizer.rs`, `src/checkpoint.rs`, `src/resume.rs`: AdamW, immutable checkpoint
  generations, atomic publication, and versioned data/training state.
- `src/inference.rs`, `src/sampling.rs`, `src/eval.rs`: generation and validation loss.
- `tests/components.rs`, `tests/regressions.rs`, `tests/reliability.rs`: tests including
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
- Training saves complete immutable `snapshots/<generation>/` directories and
  atomically replaces `CURRENT` after syncing data. CLI/library reads resolve the
  pointer; pin one generation for multi-file reads. Previous generations are
  retained. Do not mutate published generations or remove them while readers use
  them. Low-level `save`/`save_with_tokenizer` write flat model artifacts; training
  must use `save_training` to include optimizer/data state in one publication.
- Resume restores model config, optimizer moments/settings/step and dataset
  cursor/buffer. Inline loading also saves the permutation and ChaCha12 RNG.
  Verify exact uninterrupted/resumed CPU tensor equality when changing this path.
  Dataset SHA-256, loader format, batch/context sizes, training mode, and backend
  are checked. Source data must be supplied again and stay unchanged during a run.
- CLI resume inherits saved batch/context sizes when omitted. `--steps` cannot
  decrease. `--checkpoint-every N` enables periodic saves (0 = completion only);
  output `--checkpoint` is independent of the resume directory.
- Legacy pretraining resume without versioned data state warns and restarts data.
  Legacy SFT supports only `--init-from` with a fresh optimizer. Preserve this
  distinction; never describe legacy resume as exact continuation.
- SFT `--init-from` loads pretrained weights with a fresh optimizer; `--resume`
  restores SFT optimizer and JSON/JSONL progress. These flags are mutually exclusive.
  Without either flag it starts from random weights. Rendering uses alternating
  user/assistant roles, right padding, and saved tokenizer artifacts.
- Evaluation loads the checkpoint tokenizer. `chat --conversation` uses SFT role
  formatting and retains history; default chat is independent text completion.
  Both return only new text and stop at assistant-end/EOS or the context limit.
  Overlong prompts/history are rejected; no automatic history trimming exists.
- Native checkpoints and BPE JSON are not upstream PyTorch/tiktoken artifacts.

When changing these boundaries, update this file and README.md. Update the
relevant files under `docs/` when architecture, compatibility, or roadmap status
changes; implemented behavior in source takes precedence over older roadmap text.
