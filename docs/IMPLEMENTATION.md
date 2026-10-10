# Implementation Map

| Upstream component | Rust component | Status | Tests | Compatibility |
|---|---|---|---|---|
| `gpt.py` GPT core | `src/model/gpt.rs` | implemented subset | forward/loss/cache parity | untied embedding/head; excludes advanced upstream residual mechanisms |
| RMSNorm | `src/model/norm.rs` | implemented | exercised by model tests | epsilon-stable, no learnable scale |
| RoPE | `src/model/attention.rs` | implemented | shape/position test | upstream split-half rotation |
| causal attention | `src/model/attention.rs` | implemented subset | forward/batch/cache parity | MHA and GQA-ready projections; no sliding window |
| `ReLU^2` MLP | `src/model/mlp.rs` | implemented | model smoke | matches upstream activation |
| `tokenizer.py` | `src/tokenizer.rs`, `src/bpe.rs` | Rust BPE subset | roundtrip/determinism tests | standalone byte-level BPE JSON format; pretraining integration; not tiktoken pickle-compatible |
| `dataloader.py` | `src/dataloader.rs` | implemented subset | evaluation and training tests | deterministic in-memory windows |
| dataset input | `src/dataset.rs` | implemented subset | long u32/block/EOF and batching tests | byte/BPE sequential cyclic windows; cursor and pending-token resume |
| `optim.py` AdamW | `src/optimizer.rs` | implemented subset | uninterrupted vs restored optimizer update parity | decoupled decay, serializable single-device moments |
| `checkpoint_manager.py` | `src/checkpoint.rs`, `src/optimizer.rs`, `src/resume.rs` | implemented subset | exact resume, publication failure, process-kill tests | immutable generations and atomic CURRENT; single-device optimizer/data state |
| `engine.py` | `src/inference.rs`, `src/model/cache.rs` | implemented subset | chunk/batch/GQA cache parity, greedy generation parity, CLI workflow | byte/BPE generation, greedy/top-k/top-p, naive KV cache |
| `loss_eval.py` | `src/eval.rs` | implemented subset | CLI smoke | byte/BPE cross-entropy, no BPB yet |
| SFT masked loss | `src/loss.rs`, `src/sft.rs` | implemented subset | masking/rendering/batch/CLI smoke tests | JSON and streaming JSONL; right-padded batches, pretrained initialization and SFT resume; no packed sequences |
| benchmark tooling | `bench` CLI subcommand | implemented subset | manual CLI smoke | measured forward step time and tokens/sec |
| RL / CORE eval | future modules | planned | - | not implemented |
| Muon | future optimizer module | planned | - | not implemented |

## Current Boundary

The implemented model intentionally follows the stable base path rather than all
current upstream optimizations. It has untied token embedding and LM head,
pre-norm residual blocks, RMSNorm, QK-normalized causal attention, split-half
RoPE, and `ReLU^2` MLP. Value embeddings, residual lambdas, smear/backout,
sliding windows, FlashAttention, and mixed precision are not included. Inference
uses a native, correctness-tested KV cache; it is not a FlashAttention cache.

The checkpoint schema is native to this project. Matching tensor names or special
token names does not imply that an upstream PyTorch checkpoint can be loaded.

## Regression coverage

`tests/regressions.rs` covers long `.u32` streams across read boundaries and EOF,
truncated tokens, batched file updates against an explicit reference batch,
chunked cached attention with GQA and multiple batch rows, greedy cached versus
uncached generation, context/stop limits, and custom AdamW save/load continuation.
It also checks BPE evaluation against explicitly tokenized input, tokenizer identity
(including equal-size byte/BPE vocabularies), and CLI pretrain → SFT → chat/eval.
CLI checks reject empty input, zero batch sizes, and backwards resume step counts.
CPU is the automated numerical test backend; Metal parity/performance is not
established by those tests.

`tests/reliability.rs` checks full model/moment tensor equality after interrupted
inline, text-file, native `.u32`, JSON and JSONL training (byte/BPE), including
shuffle/EOF boundaries and restored optimizer settings. It tests CLI SFT resume,
changed-data/batch/mode rejection, and forced process termination after periodic
publication. A checkpoint unit test injects failure before publication and checks
that the prior generation remains readable. State restoration targets the current
single-device implementation; no global device RNG is needed after loading weights
because the model has no stochastic training layers.
