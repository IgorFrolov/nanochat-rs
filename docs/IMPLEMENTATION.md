# Implementation Map

| Upstream component | Rust component | Status | Tests | Compatibility |
|---|---|---|---|---|
| `gpt.py` GPT core | `src/model/gpt.rs` | implemented subset | forward/loss smoke | base architecture, not advanced value embeddings |
| RMSNorm | `src/model/norm.rs` | implemented | component test | epsilon-stable, no learnable scale |
| RoPE | `src/model/attention.rs` | implemented | shape/position test | upstream split-half rotation |
| causal attention | `src/model/attention.rs` | implemented | forward smoke | MHA and GQA-ready projections |
| `ReLU^2` MLP | `src/model/mlp.rs` | implemented | model smoke | matches upstream activation |
| `tokenizer.py` | `src/tokenizer.rs`, `src/bpe.rs` | Rust BPE subset | roundtrip/determinism tests | standalone byte-level BPE JSON format; pretraining integration; not tiktoken pickle-compatible |
| `dataloader.py` | `src/dataloader.rs` | implemented subset | loader smoke | deterministic in-memory windows |
| dataset input | `src/dataset.rs` | implemented subset | text/u32 tests | bounded-memory sequential windows; no shuffle/resume yet |
| `optim.py` AdamW | `src/optimizer.rs` | implemented subset | training/resume smoke | decoupled decay, serializable single-device moments |
| `checkpoint_manager.py` | `src/checkpoint.rs`, `src/optimizer.rs` | implemented subset | CLI/resume smoke | model and AdamW moments in separate SafeTensors; resume state is single-device |
| `engine.py` | `src/inference.rs`, `src/model/cache.rs` | implemented subset | generation/cache parity | byte/BPE generation, top-k/top-p, naive KV cache |
| `loss_eval.py` | `src/eval.rs` | implemented subset | CLI smoke | validation cross-entropy, no BPB yet |
| SFT masked loss | `src/loss.rs`, `src/sft.rs` | implemented subset | masking/rendering/CLI smoke tests | JSON and streaming JSONL; no batching yet |
| benchmark tooling | `bench` CLI subcommand | implemented subset | CLI smoke | measured forward step time and tokens/sec |
| SFT / RL / CORE eval | future modules | planned | - | not implemented |
| Muon | future optimizer module | planned | - | not implemented |

## Current Boundary

The implemented model intentionally follows the stable base path rather than all
current upstream optimizations. It has untied token embedding and LM head,
pre-norm residual blocks, RMSNorm, QK-normalized causal attention, split-half
RoPE, and `ReLU^2` MLP. Value embeddings, residual lambdas, smear/backout,
sliding windows, FlashAttention, mixed precision, and KV cache are not included.

The checkpoint schema is native to this project. Matching tensor names or special
token names does not imply that an upstream PyTorch checkpoint can be loaded.
