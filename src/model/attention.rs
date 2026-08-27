use super::norm::rms_norm;
use anyhow::Result;
use candle_core::{DType, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub fn apply_rope(x: &Tensor, base: f64) -> Result<Tensor> {
    let (_b, _h, t, d) = x.dims4()?;
    let half = d / 2;
    let dev = x.device();
    let pos = Tensor::arange(0u32, t as u32, dev)?
        .to_dtype(DType::F32)?
        .reshape((t, 1))?;
    let freq = Tensor::arange(0u32, half as u32, dev)?
        .to_dtype(DType::F32)?
        .affine(2.0 / d as f64, 0.)?
        .affine(-(base.ln()), 0.)?;
    let inv = freq.exp()?.reshape((1, half))?;
    let angles = pos.matmul(&inv)?;
    let cos = angles
        .cos()?
        .to_dtype(x.dtype())?
        .reshape((1, 1, t, half))?;
    let sin = angles
        .sin()?
        .to_dtype(x.dtype())?
        .reshape((1, 1, t, half))?;
    let a = x.narrow(D::Minus1, 0, half)?;
    let b = x.narrow(D::Minus1, half, half)?;
    let first = (&a.broadcast_mul(&cos)? - &b.broadcast_mul(&sin)?)?;
    let second = (&a.broadcast_mul(&sin)? + &b.broadcast_mul(&cos)?)?;
    Ok(Tensor::cat(&[first, second], D::Minus1)?)
}

pub struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
}
impl Attention {
    pub fn new(
        dim: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            q: linear(dim, heads * head_dim, vb.pp("q"))?,
            k: linear(dim, kv_heads * head_dim, vb.pp("k"))?,
            v: linear(dim, kv_heads * head_dim, vb.pp("v"))?,
            out: linear(heads * head_dim, dim, vb.pp("out"))?,
            heads,
            kv_heads,
            head_dim,
            eps,
        })
    }
    pub fn forward(&self, x: &Tensor, context: usize) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = apply_rope(
            &self
                .q
                .forward(x)?
                .reshape((b, t, self.heads, self.head_dim))?
                .transpose(1, 2)?,
            100000.,
        )?;
        let mut k = apply_rope(
            &self
                .k
                .forward(x)?
                .reshape((b, t, self.kv_heads, self.head_dim))?
                .transpose(1, 2)?,
            100000.,
        )?;
        let q = rms_norm(&q, self.eps)?;
        k = rms_norm(&k, self.eps)?;
        let mut v = self
            .v
            .forward(x)?
            .reshape((b, t, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        if self.kv_heads < self.heads {
            let repeats = self.heads / self.kv_heads;
            k = k
                .unsqueeze(2)?
                .broadcast_as((b, self.kv_heads, repeats, t, self.head_dim))?
                .reshape((b, self.heads, t, self.head_dim))?;
            v = v
                .unsqueeze(2)?
                .broadcast_as((b, self.kv_heads, repeats, t, self.head_dim))?
                .reshape((b, self.heads, t, self.head_dim))?;
        }
        let scale = (self.head_dim as f64).sqrt();
        let mut scores = q.matmul(&k.transpose(2, 3)?)?.affine(1. / scale, 0.)?;
        let mask_data: Vec<f32> = (0..t)
            .flat_map(|row| (0..t).map(move |col| if col <= row { 0.0 } else { -1e9 }))
            .collect();
        let mask = Tensor::from_vec(mask_data, (t, t), x.device())?.to_dtype(scores.dtype())?;
        scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
        let y = candle_nn::ops::softmax(&scores, D::Minus1)?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, self.heads * self.head_dim))?;
        let _ = context;
        Ok(self.out.forward(&y)?)
    }
}
