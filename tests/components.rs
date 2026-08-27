use approx::assert_relative_eq;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use nanochat_rs::bpe::BpeTokenizer;
use nanochat_rs::dataset::StreamingDataset;
use nanochat_rs::loss::masked_cross_entropy;
use nanochat_rs::{
    checkpoint,
    model::{cache::KvCache, Gpt},
};
use nanochat_rs::{
    config::GptConfig,
    model::attention::apply_rope,
    tokenizer::{Conversation, Message, Tokenizer},
};

#[test]
fn tokenizer_roundtrip_ignores_control_tokens() {
    let tokenizer = Tokenizer::default();
    let ids = tokenizer.encode_with_bos("hello Rust");
    assert_eq!(ids[0], tokenizer.bos_id);
    assert_eq!(tokenizer.decode(&ids), "hello Rust");
}

#[test]
fn conversation_masks_only_assistant_content() -> anyhow::Result<()> {
    let tokenizer = Tokenizer::default();
    let conversation = Conversation {
        messages: vec![
            Message {
                role: "user".into(),
                content: "hi".into(),
            },
            Message {
                role: "assistant".into(),
                content: "hello".into(),
            },
        ],
    };
    let rendered = tokenizer.render_conversation(&conversation, 100)?;
    assert_eq!(rendered.ids.len(), rendered.loss_mask.len());
    let assistant_start = tokenizer.special_id("<|assistant_start|>")?;
    let start = rendered
        .ids
        .iter()
        .position(|id| *id == assistant_start)
        .unwrap();
    assert!(!rendered.loss_mask[start]);
    assert!(rendered.loss_mask[start + 1..start + 6]
        .iter()
        .all(|value| *value));
    Ok(())
}

#[test]
fn bpe_training_is_deterministic_and_roundtrips() -> anyhow::Result<()> {
    let text = "hello hello hello rust rust rust\nhello rust\n";
    let first = BpeTokenizer::train(text, 280)?;
    let second = BpeTokenizer::train(text, 280)?;
    assert_eq!(first.merges, second.merges);
    let ids = first.encode("hello rust");
    assert_eq!(first.decode(&ids), "hello rust");
    assert!(first.vocab_size() <= 289);
    Ok(())
}

#[test]
fn kv_cache_matches_full_forward_for_next_token() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let config = GptConfig::from_depth(1, 266, 8);
    let vars = VarMap::new();
    let model = Gpt::new(config, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
    let prefix = Tensor::new(&[[1u32, 2, 3]], &device)?;
    let next = Tensor::new(&[[4u32]], &device)?;
    let full = Tensor::new(&[[1u32, 2, 3, 4]], &device)?;
    let expected = model.forward(&full, None)?.i((0, 3))?.to_vec1::<f32>()?;
    let mut cache = KvCache::new(1);
    let _ = model.forward_with_cache(&prefix, None, Some(&mut cache))?;
    let actual = model
        .forward_with_cache(&next, None, Some(&mut cache))?
        .i((0, 0))?
        .to_vec1::<f32>()?;
    for (left, right) in expected.iter().zip(actual.iter()) {
        assert_relative_eq!(left, right, epsilon = 1e-4);
    }
    Ok(())
}

#[test]
fn rope_preserves_shape_and_position_zero() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let x = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
    let y = apply_rope(&x, 100_000.)?;
    assert_eq!(y.dims(), x.dims());
    let first = y.i((0, 0, 0, 0))?.to_scalar::<f32>()?;
    assert_relative_eq!(first, 1.0, epsilon = 1e-6);
    Ok(())
}

#[test]
fn depth_config_is_explicit_and_valid() {
    let config = GptConfig::from_depth(4, 266, 32);
    assert_eq!(config.depth, 4);
    assert!(config.validate().is_ok());
}

#[test]
fn checkpoint_roundtrip_preserves_logits() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let config = GptConfig::from_depth(1, 266, 8);
    let vars = VarMap::new();
    let model = Gpt::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    let ids = Tensor::new(&[[1u32, 2, 3, 4]], &device)?;
    let before = model.forward(&ids, None)?.to_vec3::<f32>()?;
    let dir = std::env::temp_dir().join(format!("nanochat-rs-test-{}", std::process::id()));
    checkpoint::save(&dir, &config, &Tokenizer::default(), &vars, 0)?;
    let restored_vars = VarMap::new();
    let restored = Gpt::new(
        config,
        VarBuilder::from_varmap(&restored_vars, DType::F32, &device),
    )?;
    let mut restored_vars = restored_vars;
    checkpoint::load_vars(&dir, &mut restored_vars)?;
    let after = restored.forward(&ids, None)?.to_vec3::<f32>()?;
    assert_eq!(before, after);
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn tiny_training_loss_decreases() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let config = GptConfig::from_depth(1, 266, 8);
    let vars = VarMap::new();
    let model = Gpt::new(config, VarBuilder::from_varmap(&vars, DType::F32, &device))?;
    let x = Tensor::new(
        &[[b'a' as u32, b'b' as u32, b'a' as u32, b'b' as u32]],
        &device,
    )?;
    let y = Tensor::new(
        &[[b'b' as u32, b'a' as u32, b'b' as u32, b'a' as u32]],
        &device,
    )?;
    let mut opt = AdamW::new(
        vars.all_vars(),
        ParamsAdamW {
            lr: 0.02,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    let initial = model.forward(&x, Some(&y))?.to_scalar::<f32>()?;
    for _ in 0..25 {
        let loss = model.forward(&x, Some(&y))?;
        opt.backward_step(&loss)?;
    }
    let final_loss = model.forward(&x, Some(&y))?.to_scalar::<f32>()?;
    assert!(
        final_loss < initial,
        "loss did not decrease: {initial} -> {final_loss}"
    );
    Ok(())
}

#[test]
fn streaming_text_dataset_returns_shifted_windows() -> anyhow::Result<()> {
    let path = std::env::temp_dir().join(format!("nanochat-rs-data-{}.txt", std::process::id()));
    std::fs::write(&path, "abcdefghi")?;
    let mut dataset = StreamingDataset::open(&path, Tokenizer::default())?;
    let (input, target) = dataset.next_window(4)?.expect("first window");
    assert_eq!(
        input,
        vec![b'a' as u32, b'b' as u32, b'c' as u32, b'd' as u32]
    );
    assert_eq!(
        target,
        vec![b'b' as u32, b'c' as u32, b'd' as u32, b'e' as u32]
    );
    std::fs::remove_file(path)?;
    Ok(())
}

#[test]
fn streaming_u32_dataset_reads_little_endian_tokens() -> anyhow::Result<()> {
    let path = std::env::temp_dir().join(format!("nanochat-rs-data-{}.u32", std::process::id()));
    let mut bytes = Vec::new();
    for id in [1u32, 2, 3, 4, 5] {
        bytes.extend(id.to_le_bytes());
    }
    std::fs::write(&path, bytes)?;
    let mut dataset = StreamingDataset::open(&path, Tokenizer::default())?;
    let (input, target) = dataset.next_window(3)?.expect("first window");
    assert_eq!(input, vec![1, 2, 3]);
    assert_eq!(target, vec![2, 3, 4]);
    std::fs::remove_file(path)?;
    Ok(())
}

#[test]
fn masked_loss_ignores_unsupervised_positions() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[[10f32, 0.0], [0.0, 10.0]]], &device)?;
    let targets = Tensor::new(&[[0u32, 0]], &device)?;
    let mask = Tensor::new(&[[1u32, 0]], &device)?;
    let loss = masked_cross_entropy(&logits, &targets, &mask)?.to_scalar::<f32>()?;
    assert!(loss < 0.001, "unexpected supervised loss: {loss}");
    Ok(())
}
