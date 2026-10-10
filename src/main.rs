use anyhow::Result;
use candle_core::DType;
use candle_core::Device;
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use nanochat_rs::{
    bpe::BpeTokenizer,
    checkpoint,
    config::{DeviceKind, GptConfig, TrainConfig},
    inference::{generate_tokens, GenerationOptions},
    model::Gpt,
    tokenizer::{Conversation, Message, ModelTokenizer, Tokenizer},
};
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Tokenizer {
        #[command(subcommand)]
        command: TokenizerCommand,
    },
    Train {
        #[arg(long, default_value_t = 4)]
        depth: usize,
        #[arg(long, default_value_t = 128)]
        seq_len: usize,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long, default_value = "checkpoints/d4")]
        checkpoint: String,
        #[arg(
            long,
            default_value = "hello world hello rust hello nanochat rust is fast rust is safe"
        )]
        text: String,
        #[arg(long)]
        data: Option<String>,
        #[arg(long)]
        resume: Option<String>,
        #[arg(long)]
        tokenizer: Option<String>,
    },
    Sft {
        #[arg(long)]
        conversation: String,
        #[arg(long, default_value_t = 4)]
        depth: usize,
        #[arg(long, default_value_t = 128)]
        seq_len: usize,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long, default_value = "checkpoints/sft")]
        checkpoint: String,
        #[arg(long)]
        tokenizer: Option<String>,
        /// Initialize from pretrained weights with a fresh optimizer.
        #[arg(long)]
        init_from: Option<String>,
    },
    Chat {
        #[arg(long)]
        checkpoint: String,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long, default_value_t = 0.9)]
        top_p: f64,
        /// Render user/assistant roles and retain conversation history.
        #[arg(long)]
        conversation: bool,
    },
    Eval {
        #[arg(long)]
        checkpoint: String,
        #[arg(long, default_value_t = 10)]
        steps: usize,
        #[arg(
            long,
            default_value = "hello world hello rust hello nanochat rust is fast rust is safe"
        )]
        text: String,
    },
    Bench {
        #[arg(long, default_value_t = 4)]
        depth: usize,
        #[arg(long, default_value_t = 128)]
        seq_len: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 10)]
        steps: usize,
        #[arg(long, default_value = "cpu")]
        device: String,
    },
}
#[derive(Subcommand)]
enum ModelCommand {
    Info,
}
#[derive(Subcommand)]
enum TokenizerCommand {
    Inspect,
    Train {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
        #[arg(long, default_value_t = 1024)]
        vocab_size: usize,
    },
}
fn resolve_device(kind: &str) -> Result<Device> {
    match DeviceKind::parse(kind)? {
        DeviceKind::Cpu => Ok(Device::Cpu),
        DeviceKind::Metal => match Device::new_metal(0) {
            Ok(device) => Ok(device),
            Err(error) => {
                eprintln!("warning: Metal unavailable ({error}); falling back to CPU");
                Ok(Device::Cpu)
            }
        },
        DeviceKind::Auto => match Device::new_metal(0) {
            Ok(device) => Ok(device),
            Err(_) => {
                eprintln!("warning: Metal unavailable; using CPU");
                Ok(Device::Cpu)
            }
        },
    }
}
fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Model {
            command: ModelCommand::Info,
        } => {
            let c = GptConfig::default();
            println!(
                "nanochat-rs tiny GPT\ndepth={} dim={} heads={} vocab={} context={}",
                c.depth, c.model_dim, c.num_heads, c.vocab_size, c.context_length
            );
        }
        Command::Tokenizer {
            command: TokenizerCommand::Inspect,
        } => {
            let t = Tokenizer::default();
            println!(
                "byte tokenizer: vocab={} bos={} eos={}",
                t.vocab_size, t.bos_id, t.eos_id
            );
        }
        Command::Tokenizer {
            command:
                TokenizerCommand::Train {
                    input,
                    output,
                    vocab_size,
                },
        } => {
            let text = std::fs::read_to_string(&input)?;
            let tokenizer = BpeTokenizer::train(&text, vocab_size)?;
            tokenizer.save(&output)?;
            println!(
                "trained BPE tokenizer: vocab={} output={output}",
                tokenizer.vocab_size()
            );
        }
        Command::Train {
            depth,
            seq_len,
            steps,
            batch_size,
            device,
            checkpoint,
            tokenizer,
            text,
            data,
            resume,
        } => {
            let d = resolve_device(&device)?;
            let selected = checkpoint::select_tokenizer(resume.as_deref(), tokenizer.as_deref())?;
            let c = if let Some(dir) = resume.as_deref() {
                checkpoint::load_config(dir)?
            } else {
                GptConfig::from_depth(depth, selected.vocab_size(), seq_len)
            };
            let checkpoint = if matches!(&selected, ModelTokenizer::Bpe(_))
                && checkpoint == "checkpoints/d4"
                && resume.is_none()
            {
                "checkpoints/d4-bpe".to_string()
            } else {
                checkpoint
            };
            let t = TrainConfig {
                steps,
                batch_size,
                sequence_length: seq_len,
                checkpoint,
                ..Default::default()
            };
            println!(
                "Backend: {}",
                if matches!(d, Device::Cpu) {
                    "CPU"
                } else {
                    "Metal/MPS"
                }
            );
            let is_file = data.is_some();
            let source = data.as_deref().unwrap_or(&text);
            nanochat_rs::trainer::train_source(
                c,
                t,
                source,
                is_file,
                selected,
                &d,
                resume.as_deref(),
            )?;
        }
        Command::Sft {
            conversation,
            depth,
            seq_len,
            steps,
            batch_size,
            device,
            checkpoint,
            tokenizer,
            init_from,
        } => {
            let d = resolve_device(&device)?;
            let selected =
                checkpoint::select_tokenizer(init_from.as_deref(), tokenizer.as_deref())?;
            let c = if let Some(dir) = init_from.as_deref() {
                checkpoint::load_config(dir)?
            } else {
                GptConfig::from_depth(depth, selected.vocab_size(), seq_len)
            };
            let t = TrainConfig {
                steps,
                batch_size,
                sequence_length: seq_len,
                checkpoint,
                ..Default::default()
            };
            println!(
                "Backend: {}",
                if matches!(d, Device::Cpu) {
                    "CPU"
                } else {
                    "Metal/MPS"
                }
            );
            nanochat_rs::sft::train_from_checkpoint(
                c,
                t,
                &conversation,
                tokenizer.as_deref(),
                &d,
                init_from.as_deref(),
            )?;
        }
        Command::Chat {
            checkpoint: dir,
            max_new_tokens,
            temperature,
            top_p,
            conversation,
        } => {
            let d = Device::Cpu;
            let (model, _vars, _) = nanochat_rs::eval::load_model(&dir, &d)?;
            let tokenizer = checkpoint::load_tokenizer(&dir)?;
            let mut history = Conversation {
                messages: Vec::new(),
            };
            println!("nanochat-rs chat (Ctrl-D to exit)");
            use std::io::{self, Write};
            loop {
                print!("user> ");
                io::stdout().flush()?;
                let mut p = String::new();
                if io::stdin().read_line(&mut p)? == 0 {
                    break;
                }
                let prompt = if conversation {
                    history.messages.push(Message {
                        role: "user".into(),
                        content: p.trim().into(),
                    });
                    tokenizer.conversation_prompt(&history)?
                } else {
                    let mut ids = vec![tokenizer.special_id("<|bos|>")?];
                    ids.extend(tokenizer.encode(p.trim()));
                    ids
                };
                let generated = generate_tokens(
                    &model,
                    &tokenizer,
                    &d,
                    &prompt,
                    GenerationOptions {
                        max_new_tokens,
                        temperature,
                        top_k: 50,
                        top_p,
                        seed: 42,
                    },
                )?;
                let response = tokenizer.decode(&generated);
                if conversation {
                    history.messages.push(Message {
                        role: "assistant".into(),
                        content: response.clone(),
                    });
                }
                println!("assistant> {response}");
            }
        }
        Command::Eval {
            checkpoint,
            steps,
            text,
        } => {
            let device = Device::Cpu;
            let loss = nanochat_rs::eval::validation_loss(&checkpoint, &text, steps, &device)?;
            println!("Checkpoint: {checkpoint}\nDevice: CPU\nValidation loss: {loss:.6}");
        }
        Command::Bench {
            depth,
            seq_len,
            batch_size,
            steps,
            device,
        } => {
            let device = resolve_device(&device)?;
            let config = GptConfig::from_depth(depth, 266, seq_len);
            let vars = VarMap::new();
            let model = Gpt::new(
                config.clone(),
                VarBuilder::from_varmap(&vars, DType::F32, &device),
            )?;
            let ids = candle_core::Tensor::zeros(
                (batch_size, seq_len),
                candle_core::DType::U32,
                &device,
            )?;
            let start = std::time::Instant::now();
            for _ in 0..steps.max(1) {
                let _ = model.forward(&ids, None)?;
            }
            let elapsed = start.elapsed().as_secs_f64();
            let parameters: usize = vars.all_vars().iter().map(|v| v.elem_count()).sum();
            let tokens = (steps.max(1) * batch_size * seq_len) as f64;
            println!("Model: d{depth}\nParameters: {parameters}\nDevice: {}\nDType: f32\nContext: {seq_len}\nBatch: {batch_size}\nStep time: {:.3}s\nTokens/sec: {:.1}", if matches!(device, Device::Cpu) { "CPU" } else { "Metal/MPS" }, elapsed / steps.max(1) as f64, tokens / elapsed);
        }
    }
    Ok(())
}
