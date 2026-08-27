use anyhow::Result;
use candle_core::Tensor;

pub struct KvCache {
    pub keys: Vec<Option<Tensor>>,
    pub values: Vec<Option<Tensor>>,
    pub position: usize,
}
impl KvCache {
    pub fn new(layers: usize) -> Self {
        Self {
            keys: vec![None; layers],
            values: vec![None; layers],
            position: 0,
        }
    }
    pub fn append(
        &mut self,
        layer: usize,
        key: Tensor,
        value: Tensor,
    ) -> Result<(Tensor, Tensor, usize)> {
        let keys = match self.keys[layer].take() {
            Some(previous) => Tensor::cat(&[&previous, &key], 2)?,
            None => key,
        };
        let values = match self.values[layer].take() {
            Some(previous) => Tensor::cat(&[&previous, &value], 2)?,
            None => value,
        };
        let start = self.position;
        self.keys[layer] = Some(keys.clone());
        self.values[layer] = Some(values.clone());
        Ok((keys, values, start))
    }
    pub fn advance(&mut self, tokens: usize) {
        self.position += tokens;
    }
}
