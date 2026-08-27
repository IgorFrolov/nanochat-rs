use anyhow::{bail, Result};
use candle_core::Tensor;
use candle_core::{DType, D};
use candle_nn::ops::log_softmax;

/// Cross entropy over only supervised positions. `logits` is `[B, T, V]`,
/// `targets` is `[B, T]`, and `mask` is `[B, T]` with 1 for assistant tokens.
pub fn masked_cross_entropy(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b, t, _v) = logits.dims3()?;
    if targets.dims() != [b, t] || mask.dims() != [b, t] {
        bail!("SFT loss expects logits [B,T,V], targets/mask [B,T]");
    }
    let flat_logits = logits.reshape((b * t, logits.dim(D::Minus1)?))?;
    let log_probs = log_softmax(&flat_logits, 1)?;
    let selected = log_probs
        .gather(&targets.flatten_all()?.unsqueeze(1)?, 1)?
        .flatten_all()?
        .neg()?;
    let weights = mask.flatten_all()?.to_dtype(DType::F32)?;
    let total = weights.sum_all()?;
    Ok((selected.broadcast_mul(&weights)?)
        .sum_all()?
        .broadcast_div(&total)?)
}
