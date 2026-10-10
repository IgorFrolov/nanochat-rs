# Roadmap

- [x] Phase 0: single Cargo package, config, CLI and CPU/Metal backend selection
- [x] Phase 1: tiny GPT primitives, loss, deterministic loader and unit tests
- [x] Phase 2: AdamW training and inspectable model checkpoint
- [x] Phase 3: development tokenizer and local text input
- [x] Phase 4: naive inference and CLI chat
- [x] Phase 4a: logits checkpoint roundtrip, tiny overfit smoke test, top-p sampling
- [x] Phase 4b: validation loss and measured forward benchmark commands
- [x] Phase 4c: bounded-memory plain-text and little-endian u32 dataset input
- [x] Phase 4d: serializable AdamW moments and single-device resume training
- [x] Phase 4e: assistant-only masked cross-entropy foundation for SFT
- [x] Phase 4f: typed conversation rendering with upstream special-token layout
- [x] Phase 4g: single-conversation SFT training command with assistant-only loss
- [x] Phase 4h: streaming JSONL SFT conversations
- [x] Phase 5a: standalone Rust byte-level BPE trainer and JSON tokenizer artifact
- [x] Phase 5b: BPE integration into inline/file pretraining, SFT, chat, and evaluation
- [ ] Phase 5c: upstream-compatible tiktoken/BPE artifact (native byte/BPE `.u32` streaming is implemented)
- [x] Phase 6a: SFT rendering, masked loss, JSON and streaming JSONL training
- [x] Phase 6b: batched SFT with right padding and masked loss
- [ ] Phase 6c: system/tool message support and packed attention masks
- [x] Phase 7a: validation cross-entropy and forward benchmark
- [ ] Phase 7b: bits-per-byte evaluation and upstream evaluation tasks
- [x] Phase 8: KV cache with chunk/batch/GQA and greedy generation parity tests
- [ ] Phase 9: nanochat advanced features and profiling
- [ ] Phase 10: optional CUDA/distributed backends

## Audit follow-up plan

Completed corrective work:

- [x] Preserve every `.u32` token across block boundaries and EOF; reject partial tokens.
- [x] Fix cached attention masks for multi-token continuations; cover batches and GQA.
- [x] Reject empty training/evaluation text, invalid batch/context sizes, and backwards resume steps.
- [x] Share byte/BPE selection across file training, resume, SFT, chat, and eval.
- [x] Save tokenizer kind and BPE artifacts; reject tokenizer identity mismatches.
- [x] Honor file-training batch size and verify the update against a reference batch.
- [x] Add `sft --init-from` with fresh optimizer and `chat --conversation` with history.
- [x] Test custom AdamW continuation and the CLI pretrain → SFT → chat/eval workflow.
- [x] Update README, contributor instructions, and compatibility/implementation notes.

Remaining work, separate from the corrective pass:

1. SFT optimizer resume and exact pretraining data cursor/RNG restoration.
2. Phase 5c: pin an upstream tokenizer revision and add import/parity fixtures.
3. Phase 6c: system/tool rendering and packed attention masks (tool execution remains separate).
4. Phase 7b: BPB and pinned upstream evaluation fixtures/tasks.
5. Phase 9: establish CPU/Metal latency and memory baselines; profile and optimize
   KV-cache allocation; evaluate sliding-window attention and advanced residual
   mechanisms as separate changes with numerical tests.
6. Phase 10: optional build backends, CUDA, then distributed training and resume.

Phase checkboxes describe the implemented subset, not full upstream compatibility.
The automated numerical suite runs on CPU; Metal needs separate parity and
performance validation. Checkpoints still do not provide atomic multi-file saves.
