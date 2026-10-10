use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

pub struct DataLoader {
    tokens: Vec<u32>,
    batch_size: usize,
    seq_len: usize,
    cursor: usize,
    order: Vec<usize>,
    rng: ChaCha12Rng,
}
impl DataLoader {
    pub fn new(tokens: Vec<u32>, batch_size: usize, seq_len: usize, seed: u64) -> Result<Self> {
        if seq_len == 0 || tokens.len() < seq_len + 1 || batch_size == 0 {
            bail!("dataset needs at least seq_len+1 tokens and nonzero batch size")
        }
        let mut order: Vec<_> = (0..tokens.len() - seq_len).collect();
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataLoaderState {
    pub cursor: usize,
    pub order: Vec<usize>,
    pub rng: ChaCha12Rng,
}
impl DataLoader {
    pub fn state(&self) -> DataLoaderState {
        DataLoaderState {
            cursor: self.cursor,
            order: self.order.clone(),
            rng: self.rng.clone(),
        }
    }
    pub fn restore(&mut self, state: &DataLoaderState) -> Result<()> {
        let mut sorted = state.order.clone();
        sorted.sort_unstable();
        anyhow::ensure!(
            sorted == (0..self.tokens.len() - self.seq_len).collect::<Vec<_>>()
                && state.cursor <= state.order.len(),
            "invalid loader permutation/cursor"
        );
        self.cursor = state.cursor;
        self.order = state.order.clone();
        self.rng = state.rng.clone();
        Ok(())
    }
}
