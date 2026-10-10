use crate::{
    checkpoint,
    config::{GptConfig, TrainConfig},
    dataloader::DataLoader,
    dataset::StreamingDataset,
    model::Gpt,
    optimizer::AdamWState,
    tokenizer::ModelTokenizer,
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
    train_source(
        config,
        train,
        text,
        false,
        ModelTokenizer::default(),
        device,
        resume,
    )
}

pub fn train_with_bpe(
    config: GptConfig,
    train: TrainConfig,
    text: &str,
    tokenizer: &crate::bpe::BpeTokenizer,
    device: &Device,
    resume: Option<&str>,
) -> Result<()> {
    train_source(
        config,
        train,
        text,
        false,
        ModelTokenizer::Bpe(tokenizer.clone()),
        device,
        resume,
    )
}

pub fn train_file(
    config: GptConfig,
    train: TrainConfig,
    path: &str,
    device: &Device,
    resume: Option<&str>,
) -> Result<()> {
    train_source(
        config,
        train,
        path,
        true,
        ModelTokenizer::default(),
        device,
        resume,
    )
}

pub fn train_source(
    config: GptConfig,
    train: TrainConfig,
    source: &str,
    is_file: bool,
    tokenizer: ModelTokenizer,
    device: &Device,
    resume: Option<&str>,
) -> Result<()> {
    let start_step = resume.map(checkpoint::load_step).transpose()?.unwrap_or(0);
    train.validate(&config, start_step)?;
    tokenizer.validate_vocab(config.vocab_size)?;
    if let Some(dir) = resume {
        anyhow::ensure!(
            checkpoint::load_tokenizer(dir)? == tokenizer,
            "resume tokenizer differs from checkpoint"
        );
    }
    let mut loader = if is_file {
        TrainingData::File(StreamingDataset::open_with_tokenizer(
            source,
            tokenizer.clone(),
        )?)
    } else {
        let encoded = tokenizer.encode(source);
        anyhow::ensure!(!encoded.is_empty(), "training text must not be empty");
        let mut tokens = Vec::new();
        while tokens.len() < train.sequence_length + 1 {
            tokens.extend_from_slice(&encoded);
        }
        TrainingData::Memory(DataLoader::new(
            tokens,
            train.batch_size,
            train.sequence_length,
            train.seed,
        )?)
    };
    let mut vars = VarMap::new();
    let model = Gpt::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, device).pp("model"),
    )?;
    if let Some(dir) = resume {
        checkpoint::load_vars(dir, &mut vars)?;
    }
    let mut opt = AdamWState::new(&vars, train.learning_rate, train.weight_decay)?;
    if let Some(dir) = resume {
        opt.load(
            std::path::Path::new(dir).join("optimizer.safetensors"),
            start_step,
        )?;
    }
    let start = Instant::now();
    for step in start_step..train.steps {
        let (x, y) = loader.next(&train, device)?;
        let loss = model.forward(&x, Some(&y))?;
        let value = loss.to_scalar::<f32>()?;
        opt.backward_step(&loss)?;
        if step == start_step || (step + 1) % 10 == 0 {
            println!(
                "step {}/{} loss={value:.5} tok/s={:.0}",
                step + 1,
                train.steps,
                ((step + 1 - start_step) * train.batch_size * train.sequence_length) as f64
                    / start.elapsed().as_secs_f64()
            );
        }
    }
    checkpoint::save_with_tokenizer(&train.checkpoint, &config, &tokenizer, &vars, opt.step)?;
    opt.save(std::path::Path::new(&train.checkpoint).join("optimizer.safetensors"))
}

enum TrainingData {
    Memory(DataLoader),
    File(StreamingDataset),
}
impl TrainingData {
    fn next(&mut self, train: &TrainConfig, device: &Device) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Memory(loader) => loader.next(device),
            Self::File(loader) => {
                let mut inputs = Vec::with_capacity(train.batch_size * train.sequence_length);
                let mut targets = Vec::with_capacity(inputs.capacity());
                for _ in 0..train.batch_size {
                    let (x, y) = loader
                        .next_window(train.sequence_length)?
                        .context("dataset is empty")?;
                    inputs.extend(x);
                    targets.extend(y);
                }
                let shape = (train.batch_size, train.sequence_length);
                Ok((
                    Tensor::from_vec(inputs, shape, device)?,
                    Tensor::from_vec(targets, shape, device)?,
                ))
            }
        }
    }
}
