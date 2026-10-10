use crate::config::GptConfig;
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result};
use candle_nn::VarMap;
use std::{
    fs,
    path::{Path, PathBuf},
};

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
    let dir = resolve(dir)?;
    Ok(serde_json::from_slice(&fs::read(dir.join("config.json"))?)?)
}
pub fn load_vars(dir: impl AsRef<Path>, vars: &mut VarMap) -> Result<()> {
    vars.load(resolve(dir)?.join("model.safetensors"))
        .context("loading model safetensors")
}
pub fn load_step(dir: impl AsRef<Path>) -> Result<usize> {
    let dir = resolve(dir)?;
    let state: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("training_state.json"))?)?;
    state["step"]
        .as_u64()
        .map(|step| step as usize)
        .context("checkpoint training_state.json has no numeric step")
}

/// New checkpoints identify the tokenizer kind; legacy checkpoints are inferred
/// from their BPE artifact and validated against the model vocabulary.
pub fn load_tokenizer(dir: impl AsRef<Path>) -> Result<crate::tokenizer::ModelTokenizer> {
    use crate::tokenizer::ModelTokenizer;
    let resolved = resolve(dir)?;
    let dir = resolved.as_path();
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

/// Pin one published generation before reading multiple checkpoint files.
/// Direct legacy directories and already-resolved generations remain readable.
pub fn resolve(dir: impl AsRef<Path>) -> Result<PathBuf> {
    let dir = dir.as_ref();
    match fs::read_to_string(dir.join("CURRENT")) {
        Ok(name) => {
            let name = name.trim();
            anyhow::ensure!(
                !name.is_empty() && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-'),
                "invalid checkpoint CURRENT pointer"
            );
            let generation = dir.join("snapshots").join(name);
            anyhow::ensure!(
                generation.is_dir(),
                "checkpoint CURRENT generation is missing"
            );
            Ok(generation)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(dir.to_path_buf()),
        Err(e) => Err(e.into()),
    }
}

pub fn load_training_state(dir: impl AsRef<Path>) -> Result<Option<crate::resume::TrainingState>> {
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(resolve(dir)?.join("training_state.json"))?)?;
    if value.get("version").is_none() {
        return Ok(None);
    }
    let state: crate::resume::TrainingState = serde_json::from_value(value)?;
    anyhow::ensure!(
        state.version == 1,
        "unsupported training state version {}",
        state.version
    );
    Ok(Some(state))
}

/// Publish a complete immutable generation, including optimizer and data state.
pub fn save_training(
    dir: impl AsRef<Path>,
    config: &GptConfig,
    tokenizer: &crate::tokenizer::ModelTokenizer,
    vars: &VarMap,
    optimizer: &crate::optimizer::AdamWState,
    state: &crate::resume::TrainingState,
) -> Result<()> {
    anyhow::ensure!(
        optimizer.step == state.step,
        "optimizer/state step mismatch"
    );
    publish(dir.as_ref(), |staging| {
        save_with_tokenizer(staging, config, tokenizer, vars, state.step)?;
        optimizer.save(staging.join("optimizer.safetensors"))?;
        fs::write(
            staging.join("training_state.json"),
            serde_json::to_vec_pretty(state)?,
        )?;
        Ok(())
    })
}

fn sync_dir(dir: &Path) -> Result<()> {
    fs::File::open(dir)?.sync_all()?;
    Ok(())
}

fn publish(dir: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fs::create_dir_all(dir)?;
    let snapshots = dir.join("snapshots");
    fs::create_dir_all(&snapshots)?;
    let name = format!(
        "{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let staging = snapshots.join(format!(".pending-{name}"));
    fs::create_dir(&staging)?;
    let pointer = dir.join(format!(".CURRENT-{name}"));
    let result = (|| {
        write(&staging)?;
        for entry in fs::read_dir(&staging)? {
            fs::File::open(entry?.path())?.sync_all()?;
        }
        sync_dir(&staging)?;
        fs::rename(&staging, snapshots.join(&name))?;
        sync_dir(&snapshots)?;
        fs::write(&pointer, format!("{name}\n"))?;
        fs::File::open(&pointer)?.sync_all()?;
        // Persist the snapshots directory entry before the commit point.
        sync_dir(dir)?;
        fs::rename(&pointer, dir.join("CURRENT"))?;
        sync_dir(dir)?;
        if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
            sync_dir(parent)?;
        }
        Ok(())
    })();
    // Never remove a published generation: concurrent readers may have pinned it.
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
        let _ = fs::remove_file(&pointer);
    }
    result
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    #[test]
    fn failed_write_does_not_replace_published_generation() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("nanochat-atomic-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        publish(&dir, |p| {
            fs::write(p.join("sentinel"), "old")?;
            Ok(())
        })?;
        let pinned = resolve(&dir)?;
        let failure = publish(&dir, |p| {
            fs::write(p.join("sentinel"), "partial")?;
            anyhow::bail!("injected failure before publication")
        });
        assert!(failure.is_err());
        assert_eq!(resolve(&dir)?, pinned);
        assert_eq!(fs::read_to_string(pinned.join("sentinel"))?, "old");
        publish(&dir, |p| {
            fs::write(p.join("sentinel"), "new")?;
            Ok(())
        })?;
        assert_eq!(fs::read_to_string(resolve(&dir)?.join("sentinel"))?, "new");
        assert_eq!(fs::read_to_string(pinned.join("sentinel"))?, "old");
        fs::remove_dir_all(dir)?;
        Ok(())
    }
}
