use crate::{
    checkpoint, config::GptConfig, dataloader::DataLoader, model::Gpt, tokenizer::Tokenizer,
};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

pub fn load_model(dir: &str, device: &Device) -> Result<(Gpt, VarMap, GptConfig)> {
    let config = checkpoint::load_config(dir)?;
    let vars = VarMap::new();
    let model = Gpt::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, device).pp("model"),
    )?;
    let mut loaded = vars;
    checkpoint::load_vars(dir, &mut loaded)?;
    Ok((model, loaded, config))
}

pub fn validation_loss(dir: &str, text: &str, steps: usize, device: &Device) -> Result<f32> {
    let (model, _vars, config) = load_model(dir, device)?;
    let tokenizer = Tokenizer::default();
    let mut tokens = Vec::new();
    while tokens.len() < config.context_length + 1 {
        tokens.extend(tokenizer.encode(text));
    }
    let mut loader = DataLoader::new(tokens, 1, config.context_length, 42)?;
    let mut total = 0.0;
    for _ in 0..steps.max(1) {
        let (x, y) = loader.next(device)?;
        total += model.forward(&x, Some(&y))?.to_scalar::<f32>()?;
    }
    Ok(total / steps.max(1) as f32)
}
