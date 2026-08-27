use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

pub struct DataLoader {
    tokens: Vec<u32>,
    batch_size: usize,
    seq_len: usize,
    cursor: usize,
    order: Vec<usize>,
    rng: StdRng,
}
impl DataLoader {
    pub fn new(tokens: Vec<u32>, batch_size: usize, seq_len: usize, seed: u64) -> Result<Self> {
        if tokens.len() < seq_len + 1 || batch_size == 0 {
            bail!("dataset needs at least seq_len+1 tokens and nonzero batch size")
        }
        let mut order: Vec<_> = (0..tokens.len() - seq_len).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        order.shuffle(&mut rng);
        Ok(Self {
            tokens,
            batch_size,
            seq_len,
            cursor: 0,
            order,
            rng,
        })
    }
    pub fn next(&mut self, device: &Device) -> Result<(Tensor, Tensor)> {
        let mut xs = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut ys = xs.clone();
        for _ in 0..self.batch_size {
            if self.cursor >= self.order.len() {
                self.cursor = 0;
                self.order.shuffle(&mut self.rng);
            }
            let p = self.order[self.cursor];
            self.cursor += 1;
            xs.extend_from_slice(&self.tokens[p..p + self.seq_len]);
            ys.extend_from_slice(&self.tokens[p + 1..p + self.seq_len + 1]);
        }
        Ok((
            Tensor::from_vec(xs, (self.batch_size, self.seq_len), device)?,
            Tensor::from_vec(ys, (self.batch_size, self.seq_len), device)?,
        ))
    }
}
