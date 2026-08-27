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
