use crate::config::GptConfig;
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result};
use candle_nn::VarMap;
use std::{fs, path::Path};

pub fn save(
    dir: impl AsRef<Path>,
    config: &GptConfig,
    tokenizer: &Tokenizer,
    vars: &VarMap,
    step: usize,
) -> Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir).with_context(|| format!("creating checkpoint {}", dir.display()))?;
    fs::write(dir.join("config.json"), serde_json::to_vec_pretty(config)?)?;
    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec_pretty(tokenizer)?,
    )?;
    fs::write(
        dir.join("training_state.json"),
        serde_json::json!({"step":step}).to_string(),
    )?;
    vars.save(dir.join("model.safetensors"))
        .context("saving model safetensors")
}
pub fn load_config(dir: impl AsRef<Path>) -> Result<GptConfig> {
    Ok(serde_json::from_slice(&fs::read(
        dir.as_ref().join("config.json"),
    )?)?)
}
pub fn load_vars(dir: impl AsRef<Path>, vars: &mut VarMap) -> Result<()> {
    vars.load(dir.as_ref().join("model.safetensors"))
        .context("loading model safetensors")
}
pub fn load_step(dir: impl AsRef<Path>) -> Result<usize> {
    let state: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.as_ref().join("training_state.json"))?)?;
    state["step"]
        .as_u64()
        .map(|step| step as usize)
        .context("checkpoint training_state.json has no numeric step")
}

/// New checkpoints identify the tokenizer kind; legacy checkpoints are inferred
/// from their BPE artifact and validated against the model vocabulary.
pub fn load_tokenizer(dir: impl AsRef<Path>) -> Result<crate::tokenizer::ModelTokenizer> {
    use crate::tokenizer::ModelTokenizer;
    let dir = dir.as_ref();
    let metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("tokenizer.json"))?)?;
    let bpe_path = dir.join("bpe-tokenizer.json");
    let tokenizer = match metadata.get("kind").and_then(|x| x.as_str()) {
        Some("bpe") => ModelTokenizer::Bpe(crate::bpe::BpeTokenizer::load(&bpe_path)?),
        Some("byte") => {
            anyhow::ensure!(
                !bpe_path.exists(),
                "byte checkpoint contains a stale BPE artifact"
            );
            ModelTokenizer::Byte(serde_json::from_value(metadata)?)
        }
        None if bpe_path.exists() => {
            ModelTokenizer::Bpe(crate::bpe::BpeTokenizer::load(&bpe_path)?)
        }
        None => ModelTokenizer::Byte(serde_json::from_value(metadata)?),
        Some(kind) => anyhow::bail!("unknown checkpoint tokenizer kind: {kind}"),
    };
    tokenizer.validate_vocab(load_config(dir)?.vocab_size)?;
    Ok(tokenizer)
}

pub fn select_tokenizer(
    dir: Option<&str>,
    explicit: Option<&str>,
) -> Result<crate::tokenizer::ModelTokenizer> {
    use crate::tokenizer::ModelTokenizer;
    if let Some(dir) = dir {
        let saved = load_tokenizer(dir)?;
        if let Some(path) = explicit {
            let supplied = ModelTokenizer::from_path(Some(path))?;
            anyhow::ensure!(
                supplied == saved,
                "explicit tokenizer differs from checkpoint tokenizer"
            );
        }
        Ok(saved)
    } else {
        ModelTokenizer::from_path(explicit)
    }
}

pub fn save_with_tokenizer(
    dir: impl AsRef<Path>,
    config: &GptConfig,
    tokenizer: &crate::tokenizer::ModelTokenizer,
    vars: &VarMap,
    step: usize,
) -> Result<()> {
    use crate::tokenizer::ModelTokenizer;
    tokenizer.validate_vocab(config.vocab_size)?;
    let dir = dir.as_ref();
    let byte = match tokenizer {
        ModelTokenizer::Byte(t) => t.clone(),
        _ => Tokenizer::default(),
    };
    save(dir, config, &byte, vars, step)?;
    let mut metadata = serde_json::to_value(&byte)?;
    let artifact = dir.join("bpe-tokenizer.json");
    match tokenizer {
        ModelTokenizer::Byte(_) => {
            metadata["kind"] = "byte".into();
            if artifact.exists() {
                fs::remove_file(artifact)?;
            }
        }
        ModelTokenizer::Bpe(bpe) => {
            bpe.save(artifact)?;
            metadata["kind"] = "bpe".into();
            metadata["vocab_size"] = bpe.vocab_size().into();
            metadata["bos_id"] = bpe.special_id("<|bos|>")?.into();
            metadata.as_object_mut().unwrap().remove("eos_id");
        }
    }
    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(())
}
