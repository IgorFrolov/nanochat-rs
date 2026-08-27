use anyhow::{Context, Result};
use candle_core::{backprop::GradStore, DType, Tensor, Var};
use candle_nn::VarMap;

pub struct AdamWState {
    entries: Vec<(String, Var, Var, Var)>,
    pub step: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl AdamWState {
    pub fn new(vars: &VarMap, learning_rate: f64, weight_decay: f64) -> Result<Self> {
        let mut named: Vec<_> = vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("VarMap lock poisoned"))
            .context("reading model variables")?
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));
        let entries: Vec<_> = named
            .into_iter()
            .map(|(name, var)| {
                let shape = var.shape().clone();
                let zero = Tensor::zeros(shape, DType::F32, var.device())?;
                let first = Var::from_tensor(&zero)?;
                let second = Var::from_tensor(&zero)?;
                Ok((name, var, first, second))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            entries,
            step: 0,
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        })
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step += 1;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let correction1 = 1.0 - b1.powi(self.step as i32);
        let correction2 = 1.0 - b2.powi(self.step as i32);
        for (_, var, m, v) in &mut self.entries {
            let Some(grad) = grads.get(var) else { continue };
            let grad = grad.to_dtype(DType::F32)?;
            let next_m = ((m.as_tensor() * b1)? + (&grad * (1.0 - b1))?)?;
            let next_v = ((v.as_tensor() * b2)? + (&grad.sqr()? * (1.0 - b2))?)?;
            let update = (&next_m / correction1)?;
            let variance = (&next_v / correction2)?.sqrt()?;
            let denominator = (&variance + self.eps)?;
            let update = (&update / &denominator)?;
            let decayed = (var.as_tensor() * (1.0 - self.learning_rate * self.weight_decay))?;
            let delta = (&update * self.learning_rate)?;
            let next = (&decayed - &delta)?;
            var.set(&next)?;
            m.set(&next_m)?;
            v.set(&next_v)?;
        }
        Ok(())
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let mut tensors = std::collections::HashMap::new();
        for (name, _, m, v) in &self.entries {
            tensors.insert(format!("{name}.exp_avg"), m.as_tensor().clone());
            tensors.insert(format!("{name}.exp_avg_sq"), v.as_tensor().clone());
        }
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    pub fn load(&mut self, path: impl AsRef<std::path::Path>, step: usize) -> Result<()> {
        let tensors = candle_core::safetensors::load(
            path,
            self.entries
                .first()
                .map(|x| x.1.device())
                .context("optimizer has no parameters")?,
        )?;
        for (name, _, m, v) in &mut self.entries {
            m.set(
                tensors
                    .get(&format!("{name}.exp_avg"))
                    .context("missing AdamW first moment")?,
            )?;
            v.set(
                tensors
                    .get(&format!("{name}.exp_avg_sq"))
                    .context("missing AdamW second moment")?,
            )?;
        }
        self.step = step;
        Ok(())
    }
}
