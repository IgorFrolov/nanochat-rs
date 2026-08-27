use anyhow::Result;
use candle_core::Tensor;
pub fn rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    Ok(x.broadcast_div(&(variance + eps)?.sqrt()?)?)
}
