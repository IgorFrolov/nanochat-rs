use crate::bpe::BpeTokenizer;
use crate::{
    model::{cache::KvCache, Gpt},
    sampling::next_token,
    tokenizer::Tokenizer,
};
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use rand::{rngs::StdRng, SeedableRng};
pub struct InferenceEngine<'a> {
    pub model: &'a Gpt,
    pub tokenizer: &'a Tokenizer,
    pub device: &'a Device,
}

pub struct BpeInferenceEngine<'a> {
    pub model: &'a Gpt,
    pub tokenizer: &'a BpeTokenizer,
    pub device: &'a Device,
}
impl<'a> BpeInferenceEngine<'a> {
    pub fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        temperature: f64,
        top_k: usize,
        top_p: f64,
        seed: u64,
    ) -> Result<String> {
        let mut ids = vec![self.tokenizer.special_id("<|bos|>")?];
        ids.extend(self.tokenizer.encode(prompt));
        let mut rng = StdRng::seed_from_u64(seed);
        let mut cache = KvCache::new(self.model.config.depth);
        for _ in 0..max_new {
            let input_ids = if cache.position == 0 {
                ids.clone()
            } else {
                vec![*ids.last().unwrap_or(&0)]
            };
            let input = Tensor::from_vec(
                input_ids,
                (1, if cache.position == 0 { ids.len() } else { 1 }),
                self.device,
            )?;
            let logits = self
                .model
                .forward_with_cache(&input, None, Some(&mut cache))?
                .i((0, input.dim(1)? - 1))?
                .to_dtype(DType::F32)?;
            ids.push(next_token(&logits, temperature, top_k, top_p, &mut rng)?);
        }
        Ok(self.tokenizer.decode(&ids))
    }
}
impl<'a> InferenceEngine<'a> {
    pub fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        temperature: f64,
        top_k: usize,
        top_p: f64,
        seed: u64,
    ) -> Result<String> {
        let mut ids = self.tokenizer.encode_with_bos(prompt);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut cache = KvCache::new(self.model.config.depth);
        for _ in 0..max_new {
            let input_ids = if cache.position == 0 {
                ids.clone()
            } else {
                vec![*ids.last().unwrap_or(&0)]
            };
            let input = Tensor::from_vec(
                input_ids,
                (1, if cache.position == 0 { ids.len() } else { 1 }),
                self.device,
            )?;
            let logits = self
                .model
                .forward_with_cache(&input, None, Some(&mut cache))?
                .i((0, input.dim(1)? - 1))?
                .to_dtype(DType::F32)?;
            let id = next_token(&logits, temperature, top_k, top_p, &mut rng)?;
            ids.push(id);
            if id == self.tokenizer.eos_id {
                break;
            }
        }
        Ok(self.tokenizer.decode(&ids))
    }
}
