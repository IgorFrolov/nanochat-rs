use crate::{
    checkpoint,
    config::{GptConfig, TrainConfig},
    loss::masked_cross_entropy,
    model::Gpt,
    optimizer::AdamWState,
    tokenizer::{Conversation, RenderedConversation, Tokenizer},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::io::BufRead;

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
    if sequence_length == 0 {
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
    let tokenizer = Tokenizer::default();
    let bpe = tokenizer_path
        .map(crate::bpe::BpeTokenizer::load)
        .transpose()?;
    let is_jsonl = std::path::Path::new(conversation_path)
        .extension()
        .and_then(|x| x.to_str())
        == Some("jsonl");
    let single = if is_jsonl {
        None
    } else {
        let conversation: Conversation = serde_json::from_slice(
            &std::fs::read(conversation_path)
                .with_context(|| format!("reading conversation {conversation_path}"))?,
        )?;
        Some(match &bpe {
            Some(tokenizer) => {
                tokenizer.render_conversation(&conversation, config.context_length + 1)?
            }
            None => tokenizer.render_conversation(&conversation, config.context_length + 1)?,
        })
    };
    let vars = VarMap::new();
    let model = Gpt::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, device).pp("model"),
    )?;
    let mut optimizer = AdamWState::new(&vars, train.learning_rate, train.weight_decay)?;
    let mut reader = if is_jsonl {
        Some(std::io::BufReader::new(std::fs::File::open(
            conversation_path,
        )?))
    } else {
        None
    };
    for step in 0..train.steps {
        let mut examples = Vec::with_capacity(train.batch_size);
        for _ in 0..train.batch_size {
            let rendered =
                if let Some(single) = &single {
                    single.clone()
                } else {
                    let reader = reader.as_mut().context("missing SFT JSONL reader")?;
                    let mut line = String::new();
                    if reader.read_line(&mut line)? == 0 {
                        *reader = std::io::BufReader::new(std::fs::File::open(conversation_path)?);
                        reader.read_line(&mut line)?;
                    }
                    let conversation: Conversation = serde_json::from_str(line.trim())
                        .context("parsing SFT JSONL conversation")?;
                    match &bpe {
                        Some(tokenizer) => tokenizer
                            .render_conversation(&conversation, config.context_length + 1)?,
                        None => tokenizer
                            .render_conversation(&conversation, config.context_length + 1)?,
                    }
                };
            if rendered.ids.len() < 2 || !rendered.loss_mask[1..].iter().any(|value| *value) {
                anyhow::bail!("conversation has no supervised assistant tokens within context")
            }
            examples.push(rendered);
        }
        let pad_id = match &bpe {
            Some(tokenizer) => tokenizer.special_id("<|bos|>")?,
            None => tokenizer.bos_id,
        };
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
        if step == 0 || (step + 1) % 10 == 0 {
            println!(
                "sft step {}/{} loss={:.5}",
                step + 1,
                train.steps,
                loss.to_scalar::<f32>()?
            );
        }
    }
    checkpoint::save(
        &train.checkpoint,
        &config,
        &Tokenizer::default(),
        &vars,
        train.steps,
    )?;
    optimizer.save(std::path::Path::new(&train.checkpoint).join("optimizer.safetensors"))
}
