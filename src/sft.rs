use crate::resume::{LoaderState, SourceIdentity, TrainingState};
use crate::{
    checkpoint,
    config::{GptConfig, TrainConfig},
    loss::masked_cross_entropy,
    model::Gpt,
    optimizer::AdamWState,
    tokenizer::{Conversation, RenderedConversation},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::io::{BufRead, Seek, SeekFrom};

pub struct SftBatch {
    pub input_ids: Vec<u32>,
    pub target_ids: Vec<u32>,
    pub loss_mask: Vec<u32>,
    pub sequence_length: usize,
}

pub fn batch_rendered(examples: &[RenderedConversation], pad_id: u32) -> Result<SftBatch> {
    let sequence_length = examples
        .iter()
        .map(|example| example.ids.len().saturating_sub(1))
        .max()
        .context("SFT batch is empty")?;
    if sequence_length == 0
        || examples
            .iter()
            .any(|e| e.ids.len() < 2 || e.ids.len() != e.loss_mask.len())
    {
        anyhow::bail!("SFT examples must contain at least two tokens")
    }
    let mut inputs = Vec::with_capacity(examples.len() * sequence_length);
    let mut targets = Vec::with_capacity(examples.len() * sequence_length);
    let mut loss_mask = Vec::with_capacity(examples.len() * sequence_length);
    for example in examples {
        let length = example.ids.len() - 1;
        inputs.extend_from_slice(&example.ids[..length]);
        targets.extend_from_slice(&example.ids[1..]);
        loss_mask.extend(example.loss_mask[1..].iter().map(|value| u32::from(*value)));
        inputs.extend(std::iter::repeat_n(pad_id, sequence_length - length));
        targets.extend(std::iter::repeat_n(pad_id, sequence_length - length));
        loss_mask.extend(std::iter::repeat_n(0, sequence_length - length));
    }
    Ok(SftBatch {
        input_ids: inputs,
        target_ids: targets,
        loss_mask,
        sequence_length,
    })
}

pub fn train(
    config: GptConfig,
    train: TrainConfig,
    conversation_path: &str,
    tokenizer_path: Option<&str>,
    device: &Device,
) -> Result<()> {
    train_from_checkpoint(
        config,
        train,
        conversation_path,
        tokenizer_path,
        device,
        None,
    )
}

pub fn train_from_checkpoint(
    config: GptConfig,
    train: TrainConfig,
    conversation_path: &str,
    tokenizer_path: Option<&str>,
    device: &Device,
    init_from: Option<&str>,
) -> Result<()> {
    train_run(
        config,
        train,
        conversation_path,
        tokenizer_path,
        device,
        init_from,
        None,
    )
}

pub fn train_run(
    config: GptConfig,
    mut train: TrainConfig,
    conversation_path: &str,
    tokenizer_path: Option<&str>,
    device: &Device,
    init_from: Option<&str>,
    resume: Option<&str>,
) -> Result<()> {
    anyhow::ensure!(
        init_from.is_none() || resume.is_none(),
        "--init-from and --resume are mutually exclusive"
    );
    let pinned = init_from.or(resume).map(checkpoint::resolve).transpose()?;
    let input = pinned
        .as_ref()
        .map(|p| p.to_str().context("checkpoint path is not UTF-8"))
        .transpose()?;
    let saved = if resume.is_some() {
        Some(checkpoint::load_training_state(input.unwrap())?.context(
            "legacy SFT checkpoint has no data state; use --init-from to start a new run",
        )?)
    } else {
        None
    };
    let start_step = saved.as_ref().map_or(0, |s| s.step);
    train.validate(&config, start_step)?;
    if let Some(dir) = input {
        anyhow::ensure!(
            checkpoint::load_config(dir)? == config,
            "SFT model configuration differs from checkpoint"
        );
    }
    let tokenizer = checkpoint::select_tokenizer(input, tokenizer_path)?;
    tokenizer.validate_vocab(config.vocab_size)?;
    let is_jsonl = std::path::Path::new(conversation_path)
        .extension()
        .and_then(|x| x.to_str())
        == Some("jsonl");
    let source_kind = if is_jsonl { "jsonl" } else { "json" };
    let identity = SourceIdentity::file(conversation_path, source_kind)?;
    if let Some(state) = &saved {
        state.validate("sft", &identity, &train, device)?;
        train.learning_rate = state.optimizer.learning_rate;
        train.weight_decay = state.optimizer.weight_decay;
        train.seed = state.train.seed;
    }
    let single = if is_jsonl {
        None
    } else {
        let conversation: Conversation = serde_json::from_slice(
            &std::fs::read(conversation_path)
                .with_context(|| format!("reading conversation {conversation_path}"))?,
        )?;
        Some(tokenizer.render_conversation(&conversation, train.sequence_length + 1)?)
    };
    let mut vars = VarMap::new();
    let model = Gpt::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, device).pp("model"),
    )?;
    if let Some(dir) = input {
        checkpoint::load_vars(dir, &mut vars)?;
    }
    let mut optimizer = AdamWState::new(&vars, train.learning_rate, train.weight_decay)?;
    let mut reader = if is_jsonl {
        Some(std::io::BufReader::new(std::fs::File::open(
            conversation_path,
        )?))
    } else {
        None
    };
    if let Some(state) = &saved {
        match (&mut reader, &state.loader) {
            (
                Some(reader),
                LoaderState::Sft {
                    offset: Some(offset),
                },
            ) => {
                anyhow::ensure!(
                    *offset <= reader.get_ref().metadata()?.len(),
                    "SFT cursor exceeds file size"
                );
                reader.seek(SeekFrom::Start(*offset))?;
            }
            (None, LoaderState::Sft { offset: None }) => {}
            _ => anyhow::bail!("checkpoint SFT data loader type mismatch"),
        }
        optimizer.load(
            std::path::Path::new(input.unwrap()).join("optimizer.safetensors"),
            state.step,
        )?;
        optimizer.restore_settings(&state.optimizer)?;
    }
    let save_progress = |reader: &mut Option<std::io::BufReader<std::fs::File>>,
                         optimizer: &AdamWState|
     -> Result<()> {
        anyhow::ensure!(
            SourceIdentity::file(conversation_path, source_kind)? == identity,
            "dataset changed during training; checkpoint not published"
        );
        let offset = reader.as_mut().map(|r| r.stream_position()).transpose()?;
        let state = TrainingState {
            version: 1,
            step: optimizer.step,
            mode: "sft".into(),
            backend: crate::resume::backend(device),
            train: train.clone(),
            source: identity.clone(),
            loader: LoaderState::Sft { offset },
            optimizer: optimizer.settings(),
        };
        checkpoint::save_training(
            &train.checkpoint,
            &config,
            &tokenizer,
            &vars,
            optimizer,
            &state,
        )
    };
    for step in start_step..train.steps {
        let mut examples = Vec::with_capacity(train.batch_size);
        for _ in 0..train.batch_size {
            let rendered = if let Some(single) = &single {
                single.clone()
            } else {
                let reader = reader.as_mut().context("missing SFT JSONL reader")?;
                let mut line = String::new();
                if reader.read_line(&mut line)? == 0 {
                    *reader = std::io::BufReader::new(std::fs::File::open(conversation_path)?);
                    reader.read_line(&mut line)?;
                }
                let conversation: Conversation =
                    serde_json::from_str(line.trim()).context("parsing SFT JSONL conversation")?;
                tokenizer.render_conversation(&conversation, train.sequence_length + 1)?
            };
            if rendered.ids.len() < 2 || !rendered.loss_mask[1..].iter().any(|value| *value) {
                anyhow::bail!("conversation has no supervised assistant tokens within context")
            }
            examples.push(rendered);
        }
        let pad_id = tokenizer.special_id("<|bos|>")?;
        let batch = batch_rendered(&examples, pad_id)?;
        let input_ids = Tensor::from_vec(
            batch.input_ids,
            (train.batch_size, batch.sequence_length),
            device,
        )?;
        let target_ids = Tensor::from_vec(
            batch.target_ids,
            (train.batch_size, batch.sequence_length),
            device,
        )?;
        let mask = Tensor::from_vec(
            batch.loss_mask,
            (train.batch_size, batch.sequence_length),
            device,
        )?;
        let logits = model.forward(&input_ids, None)?;
        let loss = masked_cross_entropy(&logits, &target_ids, &mask)?;
        optimizer.backward_step(&loss)?;
        if train.checkpoint_every > 0
            && optimizer.step % train.checkpoint_every == 0
            && optimizer.step < train.steps
        {
            save_progress(&mut reader, &optimizer)?;
        }
        if step == 0 || (step + 1) % 10 == 0 {
            println!(
                "sft step {}/{} loss={:.5}",
                step + 1,
                train.steps,
                loss.to_scalar::<f32>()?
            );
        }
    }
    save_progress(&mut reader, &optimizer)
}
