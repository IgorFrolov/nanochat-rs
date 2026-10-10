# Compatibility

`nanochat-rs` uses the `karpathy/nanochat` repository as a behavioral
reference, not as a binary checkpoint format.

| Area | Current status |
|---|---|
| Runtime | Rust stable, Candle; CPU/Metal runtime, Metal enabled at build time |
| Python runtime | Not required |
| Model checkpoint | Native SafeTensors schema; upstream `.pt` files are not loadable |
| Tokenizer | Native byte tokenizer or native byte-level BPE JSON |
| Upstream tokenizer | Special-token names/order are mirrored; tiktoken/pickle artifacts are not loadable |
| Pretraining | Causal next-token prediction, AdamW, batched byte/BPE text and matching `.u32` input |
| SFT | JSON/JSONL, padded batches, assistant-only loss; pretrained initialization or SFT optimizer/data resume |
| Tool use | Python/output tokens are reserved but execution is disabled |
| Inference | Autoregressive generation, greedy/top-k/top-p, native KV cache; chunk/batch/GQA and greedy generation parity tests |
| Distributed training | Not implemented |

Resume selects the checkpoint's saved model configuration. All consumers share
checkpoint tokenizer loading. New metadata records `kind: byte|bpe`; BPE requires
`bpe-tokenizer.json`. A missing or mismatched tokenizer is an error, and explicitly
supplied tokenizers must match the saved vocabulary, merges, and special tokens.
Legacy metadata without `kind` is inferred from its BPE artifact when present;
legacy BPE-SFT checkpoints without the artifact need their original tokenizer.
Tensor names and SafeTensors weight/moment schemas are unchanged.

New training checkpoints use immutable `snapshots/<generation>/` directories and an
atomically replaced `CURRENT` pointer. Weight/moment tensor schemas remain the
same; `training_state.json` version 1 adds training mode, backend, optimizer
settings, dataset SHA-256, and loader progress. CLI readers accept roots, pinned
generations, and legacy flat directories. Direct filesystem consumers must resolve
`CURRENT` before reading files in new checkpoints.

Pretraining/SFT resume restores the data cursor/buffer; inline training also
restores window permutation and ChaCha12 RNG. Saved batch size, sequence length,
backend and dataset contents/format must match. CPU interrupted/uninterrupted
runs are tested for exact tensor equality with the current dependency lockfile;
cross-version/hardware/Metal bitwise reproducibility is not guaranteed.
Legacy pretraining state without cursors warns and restarts the data sequence;
legacy SFT needs `--init-from` with a fresh optimizer. SFT `--resume` requires a
versioned SFT checkpoint; `--init-from` remains a fresh training phase.

Chat's optional `--conversation` mode uses native SFT role
formatting and history; plain completion remains the default. Generation returns
only new text, stops at assistant-end/EOS, and respects the model context limit.

Compatibility claims should be added only after deterministic numerical fixtures
are checked against a pinned upstream revision.
