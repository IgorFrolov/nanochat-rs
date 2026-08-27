use crate::{
    checkpoint,
    config::{GptConfig, TrainConfig},
    loss::masked_cross_entropy,
    model::Gpt,
    optimizer::AdamWState,
    tokenizer::{Conversation, Tokenizer},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::io::BufRead;

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
            match &bpe {
                Some(tokenizer) => {
                    tokenizer.render_conversation(&conversation, config.context_length + 1)?
                }
                None => tokenizer.render_conversation(&conversation, config.context_length + 1)?,
            }
        };
        if rendered.ids.len() < 2 || !rendered.loss_mask[1..].iter().any(|value| *value) {
            anyhow::bail!("conversation has no supervised assistant tokens within context")
        }
        let input_ids = Tensor::from_vec(
            rendered.ids[..rendered.ids.len() - 1].to_vec(),
            (1, rendered.ids.len() - 1),
            device,
        )?;
        let target_ids = Tensor::from_vec(
            rendered.ids[1..].to_vec(),
            (1, rendered.ids.len() - 1),
            device,
        )?;
        let mask = Tensor::from_vec(
            rendered.loss_mask[1..]
                .iter()
                .map(|value| u32::from(*value))
                .collect(),
            (1, rendered.ids.len() - 1),
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
