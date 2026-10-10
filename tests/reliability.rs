use anyhow::Result;
use candle_core::Device;
use nanochat_rs::{
    bpe::BpeTokenizer,
    checkpoint,
    config::{GptConfig, TrainConfig},
    dataloader::DataLoader,
    dataset::StreamingDataset,
    sft,
    tokenizer::ModelTokenizer,
    trainer,
};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "nanochat-reliability-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn string(p: &Path) -> &str {
    p.to_str().unwrap()
}
fn train_config(dir: &Path, steps: usize) -> TrainConfig {
    TrainConfig {
        steps,
        batch_size: 2,
        sequence_length: 8,
        learning_rate: 0.001,
        weight_decay: 0.02,
        seed: 17,
        checkpoint: string(dir).into(),
        ..Default::default()
    }
}
fn same_tensors(left: &Path, right: &Path) -> Result<()> {
    for name in ["model.safetensors", "optimizer.safetensors"] {
        let left =
            candle_core::safetensors::load(checkpoint::resolve(left)?.join(name), &Device::Cpu)?;
        let right =
            candle_core::safetensors::load(checkpoint::resolve(right)?.join(name), &Device::Cpu)?;
        assert_eq!(left.len(), right.len());
        for (key, tensor) in &left {
            assert_eq!(
                tensor.flatten_all()?.to_vec1::<f32>()?,
                right[key].flatten_all()?.to_vec1::<f32>()?,
                "{name}: {key}"
            );
        }
    }
    assert_eq!(checkpoint::load_step(left)?, checkpoint::load_step(right)?);
    let l = checkpoint::load_training_state(left)?.unwrap();
    let r = checkpoint::load_training_state(right)?.unwrap();
    assert_eq!(
        serde_json::to_value(l.loader)?,
        serde_json::to_value(r.loader)?
    );
    Ok(())
}

#[test]
fn loader_rng_roundtrip_crosses_shuffle_boundaries() -> Result<()> {
    let tokens = (0..31u32).collect::<Vec<_>>();
    let mut original = DataLoader::new(tokens.clone(), 3, 8, 99)?;
    for _ in 0..5 {
        original.next(&Device::Cpu)?;
    }
    let encoded = serde_json::to_vec(&original.state())?;
    let mut restored = DataLoader::new(tokens, 3, 8, 0)?;
    restored.restore(&serde_json::from_slice(&encoded)?)?;
    for _ in 0..25 {
        let (a, b) = original.next(&Device::Cpu)?;
        let (c, d) = restored.next(&Device::Cpu)?;
        assert_eq!(a.to_vec2::<u32>()?, c.to_vec2::<u32>()?);
        assert_eq!(b.to_vec2::<u32>()?, d.to_vec2::<u32>()?);
    }
    Ok(())
}

#[test]
fn streaming_cursor_preserves_buffer_and_eof() -> Result<()> {
    let scratch = Scratch::new();
    let path = scratch.path("tokens.u32");
    let tokens = (0..2300u32).map(|x| x % 256).collect::<Vec<_>>();
    std::fs::write(
        &path,
        tokens
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let mut a = StreamingDataset::open_with_tokenizer(&path, ModelTokenizer::default())?;
    a.next_window(1010)?;
    let state = a.state()?;
    assert!(!state.pending.is_empty());
    let mut b = StreamingDataset::open_with_tokenizer(&path, ModelTokenizer::default())?;
    b.restore(&state)?;
    for _ in 0..4 {
        assert_eq!(a.next_window(1100)?, b.next_window(1100)?);
    }
    Ok(())
}

#[test]
fn pretraining_resume_is_exact_for_inline_text_and_streams() -> Result<()> {
    let scratch = Scratch::new();
    let corpus = "hello hello rust rust\naaa bbb ccc\n";
    let bpe = ModelTokenizer::Bpe(BpeTokenizer::train(corpus, 275)?);
    for (case, tokenizer, file) in [
        ("inline", ModelTokenizer::default(), false),
        ("bpe-inline", bpe.clone(), false),
        ("text", bpe.clone(), true),
        ("u32", bpe.clone(), true),
    ] {
        let base = scratch.path(&format!("{case}-base"));
        let full = scratch.path(&format!("{case}-full"));
        let split = scratch.path(&format!("{case}-split"));
        let path = scratch.path(if case == "u32" {
            "data.u32"
        } else {
            "data.txt"
        });
        if file {
            if case == "u32" {
                std::fs::write(
                    &path,
                    tokenizer
                        .encode(corpus)
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            } else {
                std::fs::write(&path, corpus)?;
            }
        }
        let source = if file { string(&path) } else { corpus };
        let config = GptConfig::from_depth(1, tokenizer.vocab_size(), 8);
        trainer::train_source(
            config.clone(),
            train_config(&base, 0),
            source,
            file,
            tokenizer.clone(),
            &Device::Cpu,
            None,
        )?;
        trainer::train_source(
            config.clone(),
            train_config(&full, 12),
            source,
            file,
            tokenizer.clone(),
            &Device::Cpu,
            Some(string(&base)),
        )?;
        let mut first = train_config(&split, 5);
        first.checkpoint_every = 2;
        trainer::train_source(
            config.clone(),
            first,
            source,
            file,
            tokenizer.clone(),
            &Device::Cpu,
            Some(string(&base)),
        )?;
        assert_eq!(std::fs::read_dir(split.join("snapshots"))?.count(), 3);
        let pinned = checkpoint::resolve(&split)?;
        let mut resumed = train_config(&split, 12);
        // Resume must restore optimizer hyperparameters and the RNG, not these defaults.
        resumed.learning_rate = 0.5;
        resumed.weight_decay = 0.9;
        resumed.seed = 100;
        trainer::train_source(
            config,
            resumed,
            source,
            file,
            tokenizer,
            &Device::Cpu,
            Some(string(&split)),
        )?;
        same_tensors(&full, &split)?;
        assert_eq!(checkpoint::load_step(&pinned)?, 5);
    }
    Ok(())
}

#[test]
fn sft_resume_is_exact_for_json_and_jsonl() -> Result<()> {
    let scratch = Scratch::new();
    let first =
        r#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}"#;
    let second =
        r#"{"messages":[{"role":"user","content":"c"},{"role":"assistant","content":"d"}]}"#;
    for bpe in [false, true] {
        for jsonl in [false, true] {
            let name = format!("{bpe}-{jsonl}");
            let base = scratch.path(&format!("{name}-base"));
            let full = scratch.path(&format!("{name}-full"));
            let split = scratch.path(&format!("{name}-split"));
            let data = scratch.path(if jsonl { "sft.jsonl" } else { "sft.json" });
            std::fs::write(
                &data,
                if jsonl {
                    format!("{first}\n{second}\n{first}\n")
                } else {
                    first.into()
                },
            )?;
            let tok_path = scratch.path("bpe.json");
            let tokenizer = if bpe {
                let t = BpeTokenizer::train("aaaa bbbb aaaa bbbb", 270)?;
                t.save(&tok_path)?;
                Some(string(&tok_path))
            } else {
                None
            };
            let selected = ModelTokenizer::from_path(tokenizer)?;
            let config = GptConfig::from_depth(1, selected.vocab_size(), 8);
            sft::train(
                config.clone(),
                train_config(&base, 0),
                string(&data),
                tokenizer,
                &Device::Cpu,
            )?;
            sft::train_run(
                config.clone(),
                train_config(&full, 7),
                string(&data),
                None,
                &Device::Cpu,
                None,
                Some(string(&base)),
            )?;
            let mut initial = train_config(&split, 2);
            initial.checkpoint_every = 1;
            sft::train_run(
                config.clone(),
                initial,
                string(&data),
                None,
                &Device::Cpu,
                None,
                Some(string(&base)),
            )?;
            let mut next = train_config(&split, 7);
            next.learning_rate = 0.8;
            sft::train_run(
                config,
                next,
                string(&data),
                None,
                &Device::Cpu,
                None,
                Some(string(&split)),
            )?;
            same_tensors(&full, &split)?;
        }
    }
    Ok(())
}

#[test]
fn resume_rejects_changed_data_batch_and_training_mode_without_publishing() -> Result<()> {
    let scratch = Scratch::new();
    let base = scratch.path("base");
    let config = GptConfig::from_depth(1, 266, 8);
    trainer::train(
        config.clone(),
        train_config(&base, 1),
        "abcdefghijk",
        &Device::Cpu,
        None,
    )?;
    let published = std::fs::read(base.join("CURRENT"))?;
    assert!(trainer::train(
        config.clone(),
        train_config(&base, 2),
        "abcdefghijX",
        &Device::Cpu,
        Some(string(&base))
    )
    .is_err());
    let mut train = train_config(&base, 2);
    train.batch_size = 3;
    assert!(trainer::train(
        config.clone(),
        train,
        "abcdefghijk",
        &Device::Cpu,
        Some(string(&base))
    )
    .is_err());
    let data = scratch.path("sft.json");
    std::fs::write(
        &data,
        r#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}"#,
    )?;
    assert!(sft::train_run(
        config,
        train_config(&base, 2),
        string(&data),
        None,
        &Device::Cpu,
        None,
        Some(string(&base))
    )
    .is_err());
    assert_eq!(std::fs::read(base.join("CURRENT"))?, published);
    Ok(())
}

fn cli(args: &[&str], success: bool) -> Result<String> {
    use std::process::{Command, Stdio};
    let mut child = Command::new(env!("CARGO_BIN_EXE_nanochat-rs"))
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let start = std::time::Instant::now();
    while child.try_wait()?.is_none() {
        if start.elapsed().as_secs() > 20 {
            child.kill()?;
            child.wait()?;
            anyhow::bail!("CLI timeout: {args:?}");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
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

#[test]
fn sft_cli_resume_inherits_batch_context_and_preserves_optimizer() -> Result<()> {
    let scratch = Scratch::new();
    let data = scratch.path("sft.jsonl");
    let base = scratch.path("base");
    let full = scratch.path("full");
    let split = scratch.path("split");
    let line = r#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}"#;
    std::fs::write(&data, format!("{line}\n{line}\n{line}\n"))?;
    cli(
        &[
            "sft",
            "--conversation",
            string(&data),
            "--depth",
            "1",
            "--seq-len",
            "8",
            "--batch-size",
            "2",
            "--steps",
            "0",
            "--checkpoint",
            string(&base),
        ],
        true,
    )?;
    cli(
        &[
            "sft",
            "--conversation",
            string(&data),
            "--resume",
            string(&base),
            "--steps",
            "5",
            "--checkpoint",
            string(&full),
        ],
        true,
    )?;
    cli(
        &[
            "sft",
            "--conversation",
            string(&data),
            "--resume",
            string(&base),
            "--steps",
            "2",
            "--checkpoint-every",
            "1",
            "--checkpoint",
            string(&split),
        ],
        true,
    )?;
    cli(
        &[
            "sft",
            "--conversation",
            string(&data),
            "--resume",
            string(&split),
            "--steps",
            "5",
            "--checkpoint",
            string(&split),
        ],
        true,
    )?;
    same_tensors(&full, &split)?;
    let before = std::fs::read(split.join("CURRENT"))?;
    cli(
        &[
            "sft",
            "--conversation",
            string(&data),
            "--resume",
            string(&split),
            "--init-from",
            string(&base),
            "--checkpoint",
            string(&split),
        ],
        false,
    )?;
    cli(
        &[
            "sft",
            "--conversation",
            string(&data),
            "--resume",
            string(&split),
            "--batch-size",
            "3",
            "--checkpoint",
            string(&split),
        ],
        false,
    )?;
    std::fs::write(&data, format!("{line}\n"))?;
    cli(
        &[
            "sft",
            "--conversation",
            string(&data),
            "--resume",
            string(&split),
            "--checkpoint",
            string(&split),
        ],
        false,
    )?;
    assert_eq!(std::fs::read(split.join("CURRENT"))?, before);
    Ok(())
}

#[test]
fn killed_training_can_resume_last_published_checkpoint() -> Result<()> {
    use std::{
        process::{Command, Stdio},
        time::{Duration, Instant},
    };
    let scratch = Scratch::new();
    let output = scratch.path("interrupted");
    let resumed = scratch.path("resumed");
    let mut child = Command::new(env!("CARGO_BIN_EXE_nanochat-rs"))
        .args([
            "train",
            "--depth",
            "1",
            "--seq-len",
            "8",
            "--batch-size",
            "2",
            "--text",
            "abcdefghijk",
            "--steps",
            "10000",
            "--checkpoint-every",
            "1",
            "--checkpoint",
            string(&output),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let start = Instant::now();
    loop {
        if output.join("CURRENT").exists() {
            break;
        }
        if let Some(status) = child.try_wait()? {
            anyhow::bail!("training exited before checkpoint: {status}");
        }
        if start.elapsed() > Duration::from_secs(20) {
            child.kill()?;
            child.wait()?;
            anyhow::bail!("checkpoint timeout");
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    child.kill()?;
    child.wait()?;
    let pinned = checkpoint::resolve(&output)?;
    let step = checkpoint::load_step(&pinned)?;
    assert!(step > 0 && step < 10000);
    let total = (step + 1).to_string();
    cli(
        &[
            "train",
            "--resume",
            string(&output),
            "--text",
            "abcdefghijk",
            "--steps",
            &total,
            "--checkpoint",
            string(&resumed),
        ],
        true,
    )?;
    assert_eq!(checkpoint::load_step(&resumed)?, step + 1);
    // A second continuation from the pinned generation must be identical.
    let reference = scratch.path("reference");
    cli(
        &[
            "train",
            "--resume",
            string(&pinned),
            "--text",
            "abcdefghijk",
            "--steps",
            &total,
            "--checkpoint",
            string(&reference),
        ],
        true,
    )?;
    same_tensors(&reference, &resumed)?;
    Ok(())
}
