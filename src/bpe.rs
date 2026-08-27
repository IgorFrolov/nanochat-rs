use crate::tokenizer::{Conversation, RenderedConversation};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BpeTokenizer {
    pub vocab: Vec<Vec<u8>>,
    pub merges: Vec<(u32, u32)>,
    pub special_tokens: Vec<String>,
}

impl BpeTokenizer {
    pub fn train(text: &str, vocab_size: usize) -> Result<Self> {
        if vocab_size < 256 {
            anyhow::bail!("BPE vocab size must be at least 256")
        }
        let mut vocab: Vec<Vec<u8>> = (0..=255).map(|id| vec![id as u8]).collect();
        let mut sequences: Vec<Vec<u32>> = text
            .lines()
            .map(|line| line.bytes().map(u32::from).collect())
            .collect();
        let merge_count = vocab_size - 256;
        let mut merges = Vec::with_capacity(merge_count);
        for _ in 0..merge_count {
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for sequence in &sequences {
                for pair in sequence.windows(2) {
                    *counts.entry((pair[0], pair[1])).or_default() += 1;
                }
            }
            let Some((&pair, &count)) = counts
                .iter()
                .max_by(|(a, x), (b, y)| x.cmp(y).then_with(|| b.cmp(a)))
            else {
                break;
            };
            if count < 2 {
                break;
            }
            let id = vocab.len() as u32;
            let mut merged = vocab[pair.0 as usize].clone();
            merged.extend(&vocab[pair.1 as usize]);
            vocab.push(merged);
            merges.push(pair);
            for sequence in &mut sequences {
                let mut replaced = Vec::with_capacity(sequence.len());
                let mut index = 0;
                while index < sequence.len() {
                    if index + 1 < sequence.len() && (sequence[index], sequence[index + 1]) == pair
                    {
                        replaced.push(id);
                        index += 2;
                    } else {
                        replaced.push(sequence[index]);
                        index += 1;
                    }
                }
                *sequence = replaced;
            }
        }
        Ok(Self {
            vocab,
            merges,
            special_tokens: [
                "<|bos|>",
                "<|user_start|>",
                "<|user_end|>",
                "<|assistant_start|>",
                "<|assistant_end|>",
                "<|python_start|>",
                "<|python_end|>",
                "<|output_start|>",
                "<|output_end|>",
            ]
            .iter()
            .map(|x| (*x).into())
            .collect(),
        })
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + self.special_tokens.len()
    }
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text.bytes().map(u32::from).collect();
        for (offset, &(left, right)) in self.merges.iter().enumerate() {
            let merged = 256 + offset as u32;
            let mut next = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == left && ids[i + 1] == right {
                    next.push(merged);
                    i += 2;
                } else {
                    next.push(ids[i]);
                    i += 1;
                }
            }
            ids = next;
        }
        ids
    }
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter_map(|id| self.vocab.get(*id as usize))
            .flatten()
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
    pub fn special_id(&self, name: &str) -> Result<u32> {
        self.special_tokens
            .iter()
            .position(|token| token == name)
            .map(|index| self.vocab.len() as u32 + index as u32)
            .context("unknown BPE special token")
    }
    pub fn render_conversation(
        &self,
        conversation: &Conversation,
        max_tokens: usize,
    ) -> Result<RenderedConversation> {
        if conversation.messages.is_empty() {
            anyhow::bail!("conversation must contain at least one message");
        }
        let mut ids = vec![self.special_id("<|bos|>")?];
        let mut loss_mask = vec![false];
        for (index, message) in conversation.messages.iter().enumerate() {
            let expected = if index % 2 == 0 { "user" } else { "assistant" };
            if message.role != expected {
                anyhow::bail!(
                    "message {index} has role '{}', expected '{expected}'",
                    message.role
                );
            }
            let assistant = message.role == "assistant";
            ids.push(self.special_id(if assistant {
                "<|assistant_start|>"
            } else {
                "<|user_start|>"
            })?);
            loss_mask.push(false);
            let content = self.encode(&message.content);
            ids.extend(&content);
            loss_mask.extend(std::iter::repeat_n(assistant, content.len()));
            ids.push(self.special_id(if assistant {
                "<|assistant_end|>"
            } else {
                "<|user_end|>"
            })?);
            loss_mask.push(assistant);
        }
        ids.truncate(max_tokens);
        loss_mask.truncate(max_tokens);
        Ok(RenderedConversation { ids, loss_mask })
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_vec_pretty(self)?).context("saving BPE tokenizer")
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_slice(
            &fs::read(path).context("reading BPE tokenizer")?,
        )?)
    }
}
