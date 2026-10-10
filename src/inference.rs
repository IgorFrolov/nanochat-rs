use crate::{
    bpe::BpeTokenizer,
    model::{cache::KvCache, Gpt},
    sampling::next_token,
    tokenizer::{ModelTokenizer, Tokenizer},
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

#[derive(Clone, Copy)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub seed: u64,
}

/// Returns only new tokens, stopping at assistant-end/EOS or the context limit.
pub fn generate_tokens(
    model: &Gpt,
    tokenizer: &ModelTokenizer,
    device: &Device,
    prompt: &[u32],
    options: GenerationOptions,
) -> Result<Vec<u32>> {
    tokenizer.validate_vocab(model.config.vocab_size)?;
    anyhow::ensure!(
        !prompt.is_empty() && prompt.len() < model.config.context_length,
        "prompt must leave room for generation within context {}",
        model.config.context_length
    );
    let stops = tokenizer.stop_ids()?;
    let mut rng = StdRng::seed_from_u64(options.seed);
    let mut cache = KvCache::new(model.config.depth);
    let mut generated = Vec::new();
    for _ in 0..options
        .max_new_tokens
        .min(model.config.context_length - prompt.len())
    {
        let input_ids = if generated.is_empty() {
            prompt.to_vec()
        } else {
            vec![*generated.last().unwrap()]
        };
        let length = input_ids.len();
        let input = Tensor::from_vec(input_ids, (1, length), device)?;
        let logits = model
            .forward_with_cache(&input, None, Some(&mut cache))?
            .i((0, length - 1))?
            .to_dtype(DType::F32)?;
        let id = next_token(
            &logits,
            options.temperature,
            options.top_k,
            options.top_p,
            &mut rng,
        )?;
        if stops.contains(&id) {
            break;
        }
        generated.push(id);
    }
    Ok(generated)
}

impl InferenceEngine<'_> {
    pub fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        temperature: f64,
        top_k: usize,
        top_p: f64,
        seed: u64,
    ) -> Result<String> {
        let tokenizer = ModelTokenizer::Byte(self.tokenizer.clone());
        let ids = self.tokenizer.encode_with_bos(prompt);
        let generated = generate_tokens(
            self.model,
            &tokenizer,
            self.device,
            &ids,
            GenerationOptions {
                max_new_tokens: max_new,
                temperature,
                top_k,
                top_p,
                seed,
            },
        )?;
        Ok(tokenizer.decode(&generated))
    }
}
impl BpeInferenceEngine<'_> {
    pub fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        temperature: f64,
        top_k: usize,
        top_p: f64,
        seed: u64,
    ) -> Result<String> {
        let tokenizer = ModelTokenizer::Bpe(self.tokenizer.clone());
        let mut ids = vec![tokenizer.special_id("<|bos|>")?];
        ids.extend(tokenizer.encode(prompt));
        let generated = generate_tokens(
            self.model,
            &tokenizer,
            self.device,
            &ids,
            GenerationOptions {
                max_new_tokens: max_new,
                temperature,
                top_k,
                top_p,
                seed,
            },
        )?;
        Ok(tokenizer.decode(&generated))
    }
}
