use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};
pub struct Mlp {
    up: Linear,
    down: Linear,
}
impl Mlp {
    pub fn new(dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            up: linear(dim, hidden, vb.pp("up"))?,
            down: linear(hidden, dim, vb.pp("down"))?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.up.forward(x)?;
        Ok(self.down.forward(&y.relu()?.sqr()?)?)
    }
}
