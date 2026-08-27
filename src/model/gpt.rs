use super::{attention::Attention, cache::KvCache, mlp::Mlp, norm::rms_norm};
use crate::config::GptConfig;
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{embedding, loss, Embedding, Linear, Module, VarBuilder};
struct Block {
    attn: Attention,
    mlp: Mlp,
}
pub struct Gpt {
    pub config: GptConfig,
    tok_emb: Embedding,
    blocks: Vec<Block>,
    norm_eps: f64,
    lm_head: Linear,
}
impl Gpt {
    pub fn new(config: GptConfig, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let mut blocks = Vec::new();
        for i in 0..config.depth {
            let b = vb.pp(format!("blocks.{i}"));
            blocks.push(Block {
                attn: Attention::new(
                    config.model_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rms_norm_eps,
                    b.pp("attn"),
                )?,
                mlp: Mlp::new(config.model_dim, config.mlp_dim, b.pp("mlp"))?,
            });
        }
        Ok(Self {
            tok_emb: embedding(config.vocab_size, config.model_dim, vb.pp("tok_emb"))?,
            lm_head: candle_nn::linear(config.model_dim, config.vocab_size, vb.pp("lm_head"))?,
            norm_eps: config.rms_norm_eps,
            config,
            blocks,
        })
    }
    pub fn forward_with_cache(
        &self,
        ids: &Tensor,
        targets: Option<&Tensor>,
        mut cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let mut x = self.tok_emb.forward(ids)?;
        x = rms_norm(&x, self.norm_eps)?;
        for (blocks_index, block) in self.blocks.iter().enumerate() {
            let h = rms_norm(&x, self.norm_eps)?;
            let attention_cache = cache.as_deref_mut().map(|cache| (cache, blocks_index));
            x = x.add(
                &block
                    .attn
                    .forward(&h, self.config.context_length, attention_cache)?,
            )?;
            let h = rms_norm(&x, self.norm_eps)?;
            x = x.add(&block.mlp.forward(&h)?)?;
        }
        let logits = self.lm_head.forward(&rms_norm(&x, self.norm_eps)?)?;
        if let Some(cache) = cache {
            cache.advance(ids.dim(1)?);
        }
        match targets {
            Some(y) => Ok(loss::cross_entropy(
                &logits.flatten(0, 1)?,
                &y.flatten(0, 1)?,
            )?),
            None => Ok(logits),
        }
    }
    pub fn forward(&self, ids: &Tensor, targets: Option<&Tensor>) -> Result<Tensor> {
        self.forward_with_cache(ids, targets, None)
    }
}
