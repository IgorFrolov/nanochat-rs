use crate::{
    checkpoint,
    config::{GptConfig, TrainConfig},
    dataloader::DataLoader,
    dataset::StreamingDataset,
    model::Gpt,
    optimizer::AdamWState,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::time::Instant;
pub fn train(
    config: GptConfig,
    train: TrainConfig,
    text: &str,
    device: &Device,
    resume: Option<&str>,
) -> Result<()> {
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, device).pp("model");
    let model = Gpt::new(config.clone(), vb)?;
    if let Some(dir) = resume {
        checkpoint::load_vars(dir, &mut vars.clone())?;
    }
    let mut opt = AdamWState::new(&vars, train.learning_rate, train.weight_decay)?;
    let start_step = if let Some(dir) = resume {
        let step = checkpoint::load_step(dir)?;
        opt.load(
            std::path::Path::new(dir).join("optimizer.safetensors"),
            step,
        )?;
        step
    } else {
        0
    };
    let mut tokens = Vec::new();
    while tokens.len() < train.sequence_length + 1 {
        tokens.extend(crate::tokenizer::Tokenizer::default().encode(text));
    }
    let mut loader = DataLoader::new(tokens, train.batch_size, train.sequence_length, train.seed)?;
    let start = Instant::now();
    for step in start_step..train.steps {
        let (x, y) = loader.next(device)?;
        let loss = model.forward(&x, Some(&y))?;
        let value = loss.to_scalar::<f32>()?;
        opt.backward_step(&loss)?;
        if step == 0 || (step + 1) % 10 == 0 {
            println!(
                "step {}/{} loss={value:.5} tok/s={:.0}",
                step + 1,
                train.steps,
                ((step + 1) * train.batch_size * train.sequence_length) as f64
                    / start.elapsed().as_secs_f64()
            );
        }
    }
    checkpoint::save(
        &train.checkpoint,
        &config,
        &crate::tokenizer::Tokenizer::default(),
        &vars,
        train.steps,
    )?;
    opt.save(std::path::Path::new(&train.checkpoint).join("optimizer.safetensors"))
}

pub fn train_with_bpe(
    config: GptConfig,
    train: TrainConfig,
    text: &str,
    tokenizer: &crate::bpe::BpeTokenizer,
    device: &Device,
) -> Result<()> {
    let vars = VarMap::new();
    let model = Gpt::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, device).pp("model"),
    )?;
    let mut optimizer = AdamWState::new(&vars, train.learning_rate, train.weight_decay)?;
    let mut tokens = Vec::new();
    while tokens.len() < train.sequence_length + 1 {
        tokens.extend(tokenizer.encode(text));
    }
    let mut loader = DataLoader::new(tokens, train.batch_size, train.sequence_length, train.seed)?;
    for step in 0..train.steps {
        let (x, y) = loader.next(device)?;
        let loss = model.forward(&x, Some(&y))?;
        optimizer.backward_step(&loss)?;
        if step == 0 || (step + 1) % 10 == 0 {
            println!(
                "step {}/{} loss={:.5}",
                step + 1,
                train.steps,
                loss.to_scalar::<f32>()?
            );
        }
    }
    checkpoint::save(
        &train.checkpoint,
        &config,
        &crate::tokenizer::Tokenizer::default(),
        &vars,
        train.steps,
    )?;
    std::fs::write(
        std::path::Path::new(&train.checkpoint).join("bpe-tokenizer.json"),
        serde_json::to_vec_pretty(tokenizer)?,
    )?;
    optimizer.save(std::path::Path::new(&train.checkpoint).join("optimizer.safetensors"))
}

pub fn train_file(
    config: GptConfig,
    train: TrainConfig,
    path: &str,
    device: &Device,
    resume: Option<&str>,
) -> Result<()> {
    let vars = VarMap::new();
    let model = Gpt::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, device).pp("model"),
    )?;
    if let Some(dir) = resume {
        checkpoint::load_vars(dir, &mut vars.clone())?;
    }
    let mut opt = AdamWState::new(&vars, train.learning_rate, train.weight_decay)?;
    let start_step = if let Some(dir) = resume {
        let step = checkpoint::load_step(dir)?;
        opt.load(
            std::path::Path::new(dir).join("optimizer.safetensors"),
            step,
        )?;
        step
    } else {
        0
    };
    let mut loader = StreamingDataset::open(path, crate::tokenizer::Tokenizer::default())?;
    for step in start_step..train.steps {
        let (input, target) = loader
            .next_window(train.sequence_length)?
            .context("dataset ended before a complete training window was available")?;
        let x = Tensor::from_vec(input, (1, train.sequence_length), device)?;
        let y = Tensor::from_vec(target, (1, train.sequence_length), device)?;
        let loss = model.forward(&x, Some(&y))?;
        opt.backward_step(&loss)?;
        if step == 0 || (step + 1) % 10 == 0 {
            println!(
                "step {}/{} loss={:.5}",
                step + 1,
                train.steps,
                loss.to_scalar::<f32>()?
            );
        }
    }
    checkpoint::save(
        &train.checkpoint,
        &config,
        &crate::tokenizer::Tokenizer::default(),
        &vars,
        train.steps,
    )?;
    opt.save(std::path::Path::new(&train.checkpoint).join("optimizer.safetensors"))
}
