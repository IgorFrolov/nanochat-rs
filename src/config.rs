use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DeviceKind {
    Cpu,
    Metal,
    Auto,
}

impl DeviceKind {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "mps" | "metal" => Ok(Self::Metal),
            "auto" => Ok(Self::Auto),
            other => bail!("unknown device '{other}', expected cpu, mps, or auto"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GptConfig {
    pub depth: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub model_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub mlp_dim: usize,
    pub rms_norm_eps: f64,
}

impl Default for GptConfig {
    fn default() -> Self {
        Self::from_depth(4, 266, 128)
    }
}

impl GptConfig {
    pub fn from_depth(depth: usize, vocab_size: usize, context_length: usize) -> Self {
        let model_dim = depth * 32;
        let num_heads = 4;
        Self {
            depth,
            vocab_size,
            context_length,
            model_dim,
            num_heads,
            num_kv_heads: num_heads,
            head_dim: model_dim / num_heads,
            mlp_dim: model_dim * 4,
            rms_norm_eps: 1e-5,
        }
    }
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.depth == 0 || self.model_dim == 0 || self.context_length == 0 {
            bail!("model dimensions must be non-zero")
        }
        if self.num_heads == 0
            || self.num_kv_heads == 0
            || !self.num_heads.is_multiple_of(self.num_kv_heads)
        {
            bail!("num_heads must be divisible by num_kv_heads")
        }
        if self.model_dim != self.num_heads * self.head_dim {
            bail!("model_dim must equal num_heads * head_dim")
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub seed: u64,
    pub checkpoint: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            batch_size: 1,
            sequence_length: 64,
            learning_rate: 3e-4,
            weight_decay: 0.1,
            seed: 42,
            checkpoint: "checkpoints/d4".into(),
        }
    }
}
