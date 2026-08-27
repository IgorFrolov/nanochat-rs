use anyhow::Result;
use candle_core::Tensor;
use rand::Rng;
pub fn next_token(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    rng: &mut impl Rng,
) -> Result<u32> {
    let v = logits.to_vec1::<f32>()?;
    if temperature <= 0.0 {
        return Ok(v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0));
    }
    let mut ix: Vec<usize> = (0..v.len()).collect();
    ix.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    ix.truncate(if top_k == 0 {
        v.len()
    } else {
        top_k.min(v.len())
    });
    let max = ix.iter().map(|&i| v[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f64> = ix
        .iter()
        .map(|&i| ((v[i] - max) as f64 / temperature).exp())
        .collect();
    if (0.0..1.0).contains(&top_p) {
        let total: f64 = weights.iter().sum();
        let mut cumulative = 0.0;
        let keep = weights
            .iter()
            .position(|weight| {
                cumulative += *weight / total;
                cumulative >= top_p
            })
            .map(|index| index + 1)
            .unwrap_or(weights.len());
        ix.truncate(keep);
        weights.truncate(keep);
    }
    let total: f64 = weights.iter().sum();
    let mut r = rng.random::<f64>() * total;
    for (i, w) in ix.iter().zip(weights) {
        r -= w;
        if r <= 0. {
            return Ok(*i as u32);
        }
    }
    Ok(*ix.last().unwrap_or(&0) as u32)
}
