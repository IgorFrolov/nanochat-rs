# Compatibility

`nanochat-rs` uses the current `karpathy/nanochat` repository as a behavioral
reference, not as a binary checkpoint format.

| Area | Current status |
|---|---|
| Runtime | Rust stable, Candle, CPU and optional Metal |
| Python runtime | Not required |
| Model checkpoint | Native SafeTensors schema; upstream `.pt` files are not loadable |
| Tokenizer | Native byte tokenizer or native byte-level BPE JSON |
| Upstream tokenizer | Special-token names/order are mirrored; tiktoken/pickle artifacts are not loadable |
| Pretraining | Causal next-token prediction, AdamW, local text and `.u32` input |
| SFT | JSON/JSONL conversations, assistant-only loss mask |
| Tool use | Python/output tokens are reserved but execution is disabled |
| Inference | Naive autoregressive generation, greedy/top-k/top-p; no KV cache |
| Distributed training | Not implemented |

Compatibility claims should be added only after deterministic numerical fixtures
are checked against a pinned upstream revision.
