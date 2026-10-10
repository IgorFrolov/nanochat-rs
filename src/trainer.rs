use crate::{
    checkpoint,
    config::{GptConfig, TrainConfig},
    dataloader::DataLoader,
    dataset::StreamingDataset,
    model::Gpt,
    optimizer::AdamWState,
    resume::{LoaderState, SourceIdentity, TrainingState},
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
    mut train: TrainConfig,
    source: &str,
    is_file: bool,
    tokenizer: ModelTokenizer,
    device: &Device,
    resume: Option<&str>,
) -> Result<()> {
    let pinned = resume.map(checkpoint::resolve).transpose()?;
    let resume = pinned
        .as_ref()
        .map(|p| p.to_str().context("checkpoint path is not UTF-8"))
        .transpose()?;
    let start_step = resume.map(checkpoint::load_step).transpose()?.unwrap_or(0);
    let saved = resume
        .map(checkpoint::load_training_state)
        .transpose()?
        .flatten();
    train.validate(&config, start_step)?;
    let source_kind = if std::path::Path::new(source)
        .extension()
        .is_some_and(|e| e == "u32")
    {
        "u32"
    } else {
        "text"
    };
    let identify = || -> Result<SourceIdentity> {
        if is_file {
            SourceIdentity::file(source, source_kind)
        } else {
            Ok(SourceIdentity::text(source))
        }
    };
    let identity = identify()?;
    if let Some(state) = &saved {
        state.validate("pretrain", &identity, &train, device)?;
        train.learning_rate = state.optimizer.learning_rate;
        train.weight_decay = state.optimizer.weight_decay;
        train.seed = state.train.seed;
    } else if resume.is_some() {
        eprintln!(
            "warning: legacy checkpoint has no data/RNG state; input restarts from the beginning"
        );
    }
    tokenizer.validate_vocab(config.vocab_size)?;
    if let Some(dir) = resume {
        anyhow::ensure!(
            checkpoint::load_config(dir)? == config,
            "resume model configuration differs from checkpoint"
        );
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
    if let Some(state) = &saved {
        loader.restore(&state.loader)?;
    }
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
    if let Some(state) = &saved {
        opt.restore_settings(&state.optimizer)?;
    }
    let save_progress = |loader: &mut TrainingData, opt: &AdamWState| -> Result<()> {
        anyhow::ensure!(
            identify()? == identity,
            "dataset changed during training; checkpoint not published"
        );
        let state = TrainingState {
            version: 1,
            step: opt.step,
            mode: "pretrain".into(),
            backend: crate::resume::backend(device),
            train: train.clone(),
            source: identity.clone(),
            loader: loader.state()?,
            optimizer: opt.settings(),
        };
        checkpoint::save_training(&train.checkpoint, &config, &tokenizer, &vars, opt, &state)
    };
    let start = Instant::now();
    for step in start_step..train.steps {
        let (x, y) = loader.next(&train, device)?;
        let loss = model.forward(&x, Some(&y))?;
        let value = loss.to_scalar::<f32>()?;
        opt.backward_step(&loss)?;
        if train.checkpoint_every > 0
            && opt.step % train.checkpoint_every == 0
            && opt.step < train.steps
        {
            save_progress(&mut loader, &opt)?;
        }
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
    save_progress(&mut loader, &opt)
}

enum TrainingData {
    Memory(DataLoader),
    File(StreamingDataset),
}
impl TrainingData {
    fn state(&mut self) -> Result<LoaderState> {
        match self {
            Self::Memory(l) => Ok(LoaderState::Memory(l.state())),
            Self::File(l) => Ok(LoaderState::File(l.state()?)),
        }
    }
    fn restore(&mut self, state: &LoaderState) -> Result<()> {
        match (self, state) {
            (Self::Memory(l), LoaderState::Memory(s)) => l.restore(s),
            (Self::File(l), LoaderState::File(s)) => l.restore(s),
            _ => anyhow::bail!("checkpoint data loader type mismatch"),
        }
    }

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
