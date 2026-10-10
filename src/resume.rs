//! Versioned training progress. Model initialization is random, but after weights
//! are restored the current model has no stochastic layers; only window shuffling
//! consumes RNG during training.
use crate::{
    config::TrainConfig, dataloader::DataLoaderState, dataset::StreamingState,
    optimizer::OptimizerSettings,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceIdentity {
    pub kind: String,
    pub sha256: String,
}
impl SourceIdentity {
    pub fn text(text: &str) -> Self {
        Self {
            kind: "inline".into(),
            sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
        }
    }
    pub fn file(path: &str, kind: &str) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut hash = Sha256::new();
        let mut buf = [0u8; 65536];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
        }
        Ok(Self {
            kind: kind.into(),
            sha256: format!("{:x}", hash.finalize()),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "state")]
pub enum LoaderState {
    Memory(DataLoaderState),
    File(StreamingState),
    Sft { offset: Option<u64> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingState {
    pub version: u32,
    pub step: usize,
    pub mode: String,
    pub backend: String,
    pub train: TrainConfig,
    pub source: SourceIdentity,
    pub loader: LoaderState,
    pub optimizer: OptimizerSettings,
}
impl TrainingState {
    pub fn validate(
        &self,
        mode: &str,
        source: &SourceIdentity,
        train: &TrainConfig,
        device: &candle_core::Device,
    ) -> Result<()> {
        anyhow::ensure!(
            self.version == 1,
            "unsupported training state version {}",
            self.version
        );
        anyhow::ensure!(
            self.mode == mode,
            "checkpoint training mode is {}, expected {mode}; use --init-from for fresh SFT",
            self.mode
        );
        anyhow::ensure!(
            &self.source == source,
            "dataset content or format differs from checkpoint"
        );
        anyhow::ensure!(
            self.train.batch_size == train.batch_size
                && self.train.sequence_length == train.sequence_length,
            "resume requires saved batch size {} and sequence length {}",
            self.train.batch_size,
            self.train.sequence_length
        );
        anyhow::ensure!(
            self.backend == backend(device),
            "exact resume requires saved backend {}",
            self.backend
        );
        Ok(())
    }
}
pub fn backend(device: &candle_core::Device) -> String {
    if matches!(device, candle_core::Device::Cpu) {
        "cpu"
    } else {
        "metal"
    }
    .into()
}
