use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, VarMap};
use nanochat_rs::{
    bpe::BpeTokenizer,
    checkpoint,
    config::GptConfig,
    dataloader::DataLoader,
    dataset::StreamingDataset,
    model::{cache::KvCache, Gpt},
    optimizer::AdamWState,
    tokenizer::{Conversation, Message, ModelTokenizer, Tokenizer},
};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "nanochat-regression-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn u32_stream_preserves_block_boundaries_and_wraps() -> Result<()> {
    let dir = Scratch::new();
    let path = dir.file("tokens.u32");
    let tokens: Vec<u32> = (0..2300).map(|i| i % 266).collect();
    std::fs::write(
        &path,
        tokens
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let mut data = StreamingDataset::open(&path, Tokenizer::default())?;
    for window in 0..3 {
        let (x, y) = data.next_window(1100)?.unwrap();
        for i in 0..1100 {
            assert_eq!(x[i], tokens[(window * 1100 + i) % tokens.len()]);
            assert_eq!(y[i], tokens[(window * 1100 + i + 1) % tokens.len()]);
        }
    }
    let mut bytes = 1u32.to_le_bytes().to_vec();
    bytes.push(2);
    std::fs::write(&path, bytes)?;
    assert!(StreamingDataset::open(&path, Tokenizer::default())?
        .next_window(4)
        .is_err());
    Ok(())
}

#[test]
fn cached_chunks_match_full_forward_with_batch_and_gqa() -> Result<()> {
    let d = Device::Cpu;
    let mut config = GptConfig::from_depth(2, 266, 8);
    config.num_kv_heads = 2;
    let vars = VarMap::new();
    let model = Gpt::new(config, VarBuilder::from_varmap(&vars, DType::F32, &d))?;
    let ids = Tensor::new(&[[1u32, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], &d)?;
    let expected = model.forward(&ids, None)?;
    let mut cache = KvCache::new(2);
    let mut position = 0;
    for count in [3, 2, 1] {
        let actual =
            model.forward_with_cache(&ids.narrow(1, position, count)?, None, Some(&mut cache))?;
        let error = (actual - expected.narrow(1, position, count)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(error < 1e-4, "cache error {error}");
        position += count;
    }
    assert_eq!(cache.position, 6);
    assert!(model
        .forward_with_cache(&ids, None, Some(&mut cache))
        .is_err());
    Ok(())
}

#[test]
fn adamw_resume_matches_uninterrupted_update() -> Result<()> {
    let dir = Scratch::new();
    let d = Device::Cpu;
    let config = GptConfig::from_depth(1, 266, 8);
    let vars = VarMap::new();
    let model = Gpt::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, &d).pp("model"),
    )?;
    let x = Tensor::new(&[[1u32, 2, 3, 4]], &d)?;
    let y = Tensor::new(&[[2u32, 3, 4, 5]], &d)?;
    let mut opt = AdamWState::new(&vars, 0.001, 0.1)?;
    for _ in 0..2 {
        opt.backward_step(&model.forward(&x, Some(&y))?)?;
    }
    checkpoint::save_with_tokenizer(&dir.0, &config, &ModelTokenizer::default(), &vars, opt.step)?;
    opt.save(dir.file("optimizer.safetensors"))?;
    let (restored, restored_vars, _) = nanochat_rs::eval::load_model(dir.0.to_str().unwrap(), &d)?;
    let mut resumed = AdamWState::new(&restored_vars, 0.001, 0.1)?;
    resumed.load(
        dir.file("optimizer.safetensors"),
        checkpoint::load_step(&dir.0)?,
    )?;
    opt.backward_step(&model.forward(&x, Some(&y))?)?;
    resumed.backward_step(&restored.forward(&x, Some(&y))?)?;
    assert_eq!(resumed.step, opt.step);
    let error = (model.forward(&x, None)? - restored.forward(&x, None)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(error < 1e-6, "resume error {error}");
    Ok(())
}

#[test]
fn bpe_evaluation_and_checkpoint_identity() -> Result<()> {
    let dir = Scratch::new();
    let d = Device::Cpu;
    // Exactly 266 tokens: vocabulary size alone cannot distinguish this from byte.
    let bpe = BpeTokenizer::train("aaaa aaaa aaaa", 257)?;
    let tokenizer = ModelTokenizer::Bpe(bpe.clone());
    assert_eq!(tokenizer.vocab_size(), 266);
    let config = GptConfig::from_depth(1, 266, 4);
    let vars = VarMap::new();
    let model = Gpt::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, &d).pp("model"),
    )?;
    checkpoint::save_with_tokenizer(&dir.0, &config, &tokenizer, &vars, 0)?;
    assert_eq!(checkpoint::load_tokenizer(&dir.0)?, tokenizer);
    let text = "aaaa aaaa aaaa";
    let mut loader = DataLoader::new(tokenizer.encode(text), 1, 4, 42)?;
    let (x, y) = loader.next(&d)?;
    let expected = model.forward(&x, Some(&y))?.to_scalar::<f32>()?;
    let actual = nanochat_rs::eval::validation_loss(dir.0.to_str().unwrap(), text, 1, &d)?;
    assert!((actual - expected).abs() < 1e-6);
    let other = dir.file("other.json");
    BpeTokenizer::train("bbbb bbbb bbbb", 257)?.save(&other)?;
    assert!(checkpoint::select_tokenizer(
        Some(dir.0.to_str().unwrap()),
        Some(other.to_str().unwrap())
    )
    .is_err());
    std::fs::remove_file(dir.file("bpe-tokenizer.json"))?;
    assert!(checkpoint::load_tokenizer(&dir.0).is_err());
    // Saving byte weights removes an old BPE artifact.
    bpe.save(dir.file("bpe-tokenizer.json"))?;
    checkpoint::save_with_tokenizer(&dir.0, &config, &ModelTokenizer::default(), &vars, 0)?;
    assert!(!dir.file("bpe-tokenizer.json").exists());
    assert_eq!(
        checkpoint::load_tokenizer(&dir.0)?,
        ModelTokenizer::default()
    );
    Ok(())
}

#[test]
fn conversation_prompt_matches_sft_prefix() -> Result<()> {
    for tokenizer in [
        ModelTokenizer::default(),
        ModelTokenizer::Bpe(BpeTokenizer::train("hello hello", 270)?),
    ] {
        let mut conversation = Conversation {
            messages: vec![Message {
                role: "user".into(),
                content: "hello".into(),
            }],
        };
        let prompt = tokenizer.conversation_prompt(&conversation)?;
        conversation.messages.push(Message {
            role: "assistant".into(),
            content: "reply".into(),
        });
        let rendered = tokenizer.render_conversation(&conversation, 100)?;
        assert_eq!(&rendered.ids[..prompt.len()], prompt.as_slice());
        assert!(tokenizer
            .stop_ids()?
            .contains(&tokenizer.special_id("<|assistant_end|>")?));
    }
    Ok(())
}

fn cli(args: &[&str], input: &str, success: bool) -> Result<String> {
    use std::{
        io::Write,
        process::{Command, Stdio},
        time::{Duration, Instant},
    };
    let mut child = Command::new(env!("CARGO_BIN_EXE_nanochat-rs"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child.stdin.take().unwrap().write_all(input.as_bytes())?;
    let started = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if started.elapsed() > Duration::from_secs(20) {
            child.kill()?;
            child.wait()?;
            anyhow::bail!("CLI timed out: {args:?}");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.success(), success, "{args:?}: {text}");
    Ok(text)
}
fn path(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn cli_bpe_pretrain_sft_chat_and_invalid_inputs() -> Result<()> {
    let dir = Scratch::new();
    let tok = dir.file("bpe.json");
    let data = dir.file("train.txt");
    let pre = dir.file("pre");
    let sft = dir.file("sft");
    let copy = dir.file("copy");
    let convo = dir.file("conversations.jsonl");
    std::fs::write(&data, "hello hello rust rust hello rust\n")?;
    std::fs::write(&convo,"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"hello\"}]}\n{\"messages\":[{\"role\":\"user\",\"content\":\"rust\"},{\"role\":\"assistant\",\"content\":\"hello rust\"}]}\n")?;
    cli(
        &[
            "tokenizer",
            "train",
            "--input",
            path(&data),
            "--output",
            path(&tok),
            "--vocab-size",
            "280",
        ],
        "",
        true,
    )?;
    cli(
        &[
            "train",
            "--data",
            path(&data),
            "--tokenizer",
            path(&tok),
            "--depth",
            "1",
            "--seq-len",
            "32",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--checkpoint",
            path(&pre),
        ],
        "",
        true,
    )?;
    assert!(pre.join("bpe-tokenizer.json").exists());
    let original = std::fs::read(pre.join("model.safetensors"))?;
    cli(
        &[
            "sft",
            "--init-from",
            path(&pre),
            "--conversation",
            path(&convo),
            "--seq-len",
            "32",
            "--steps",
            "0",
            "--checkpoint",
            path(&copy),
        ],
        "",
        true,
    )?;
    // Compare tensors, since serialization order is unspecified.
    let before = candle_core::safetensors::load(pre.join("model.safetensors"), &Device::Cpu)?;
    let copied = candle_core::safetensors::load(copy.join("model.safetensors"), &Device::Cpu)?;
    for (name, tensor) in &before {
        assert_eq!(
            tensor.flatten_all()?.to_vec1::<f32>()?,
            copied[name].flatten_all()?.to_vec1::<f32>()?
        );
    }
    cli(
        &[
            "sft",
            "--init-from",
            path(&pre),
            "--conversation",
            path(&convo),
            "--seq-len",
            "32",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--checkpoint",
            path(&sft),
        ],
        "",
        true,
    )?;
    assert_eq!(
        checkpoint::load_tokenizer(&sft)?,
        checkpoint::load_tokenizer(&pre)?
    );
    assert_eq!(std::fs::read(pre.join("model.safetensors"))?, original);
    let chat = cli(
        &[
            "chat",
            "--checkpoint",
            path(&sft),
            "--conversation",
            "--max-new-tokens",
            "1",
        ],
        "hi\nrust\n",
        true,
    )?;
    assert_eq!(chat.matches("assistant> ").count(), 2);
    cli(
        &[
            "eval",
            "--checkpoint",
            path(&sft),
            "--steps",
            "1",
            "--text",
            "hello rust",
        ],
        "",
        true,
    )?;
    let err = cli(
        &[
            "train",
            "--resume",
            path(&pre),
            "--seq-len",
            "32",
            "--steps",
            "0",
            "--checkpoint",
            path(&pre),
        ],
        "",
        false,
    )?;
    assert!(err.contains("less than checkpoint step"));
    assert_eq!(checkpoint::load_step(&pre)?, 1);
    assert_eq!(std::fs::read(pre.join("model.safetensors"))?, original);
    for extra in [
        vec!["--text", ""],
        vec!["--text", "", "--tokenizer", path(&tok)],
        vec!["--batch-size", "0"],
    ] {
        let mut args = vec![
            "train",
            "--depth",
            "1",
            "--seq-len",
            "8",
            "--steps",
            "1",
            "--checkpoint",
            path(&copy),
        ];
        args.extend(extra);
        cli(&args, "", false)?;
    }
    cli(
        &["eval", "--checkpoint", path(&sft), "--text", ""],
        "",
        false,
    )?;
    cli(
        &[
            "train",
            "--resume",
            path(&pre),
            "--data",
            path(&data),
            "--seq-len",
            "32",
            "--steps",
            "2",
            "--checkpoint",
            path(&copy),
        ],
        "",
        true,
    )?;
    assert_eq!(checkpoint::load_step(&copy)?, 2);
    assert_eq!(
        checkpoint::load_tokenizer(&copy)?,
        checkpoint::load_tokenizer(&pre)?
    );
    Ok(())
}

#[test]
fn file_training_uses_all_batch_rows() -> Result<()> {
    use nanochat_rs::{config::TrainConfig, trainer};
    let dir = Scratch::new();
    let base = dir.file("base");
    let output = dir.file("output");
    let data = dir.file("train.txt");
    std::fs::write(&data, "abcdwxyz1234")?;
    let d = Device::Cpu;
    let config = GptConfig::from_depth(1, 266, 4);
    let vars = VarMap::new();
    let model = Gpt::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, &d).pp("model"),
    )?;
    let mut opt = AdamWState::new(&vars, 3e-4, 0.1)?;
    checkpoint::save_with_tokenizer(&base, &config, &ModelTokenizer::default(), &vars, 0)?;
    opt.save(base.join("optimizer.safetensors"))?;
    let x = Tensor::new(&[[97u32, 98, 99, 100], [119, 120, 121, 122]], &d)?;
    let y = Tensor::new(&[[98u32, 99, 100, 119], [120, 121, 122, 49]], &d)?;
    opt.backward_step(&model.forward(&x, Some(&y))?)?;
    trainer::train_file(
        config,
        TrainConfig {
            steps: 1,
            batch_size: 2,
            sequence_length: 4,
            checkpoint: path(&output).into(),
            ..Default::default()
        },
        path(&data),
        &d,
        Some(path(&base)),
    )?;
    let (actual, _, _) = nanochat_rs::eval::load_model(path(&output), &d)?;
    let error = (actual.forward(&x, None)? - model.forward(&x, None)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(error < 1e-6, "batch update differs: {error}");
    Ok(())
}

#[test]
fn greedy_generation_matches_uncached_loop_and_stops() -> Result<()> {
    use nanochat_rs::inference::{generate_tokens, GenerationOptions};
    let d = Device::Cpu;
    let vars = VarMap::new();
    let config = GptConfig::from_depth(1, 266, 8);
    let model = Gpt::new(
        config,
        VarBuilder::from_varmap(&vars, DType::F32, &d).pp("model"),
    )?;
    let tokenizer = ModelTokenizer::default();
    let options = GenerationOptions {
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: 42,
    };
    let prompt = vec![256u32, 97, 98];
    let actual = generate_tokens(&model, &tokenizer, &d, &prompt, options)?;
    let mut ids = prompt.clone();
    let stops = tokenizer.stop_ids()?;
    for _ in 0..4 {
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &d)?;
        let next = model
            .forward(&input, None)?
            .i((0, ids.len() - 1))?
            .argmax(0)?
            .to_scalar::<u32>()?;
        if stops.contains(&next) {
            break;
        }
        ids.push(next);
    }
    assert_eq!(actual, ids[prompt.len()..]);
    // Force a known non-stop token, then assistant-end, to verify both limits.
    let bias = vars
        .data()
        .lock()
        .unwrap()
        .get("model.lm_head.bias")
        .unwrap()
        .clone();
    for (id, expected_len) in [(97usize, 5usize), (260, 0)] {
        let mut values = vec![-1e6f32; 266];
        values[id] = 1e6;
        bias.set(&Tensor::from_vec(values, 266, &d)?)?;
        let generated = generate_tokens(
            &model,
            &tokenizer,
            &d,
            &prompt,
            GenerationOptions {
                max_new_tokens: 100,
                ..options
            },
        )?;
        assert_eq!(generated.len(), expected_len);
    }
    assert!(generate_tokens(&model, &tokenizer, &d, &[0; 8], options).is_err());
    Ok(())
}
