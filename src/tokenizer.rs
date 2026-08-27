use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tokenizer {
    pub bos_id: u32,
    pub eos_id: u32,
    pub vocab_size: usize,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self {
            bos_id: 256,
            eos_id: 265,
            vocab_size: 266,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Conversation {
    pub messages: Vec<Message>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedConversation {
    pub ids: Vec<u32>,
    pub loss_mask: Vec<bool>,
}

impl Tokenizer {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|&b| b as u32).collect()
    }
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(
            &ids.iter()
                .filter_map(|&id| (id < 256).then_some(id as u8))
                .collect::<Vec<_>>(),
        )
        .into_owned()
    }
    pub fn encode_with_bos(&self, text: &str) -> Vec<u32> {
        let mut x = vec![self.bos_id];
        x.extend(self.encode(text));
        x
    }
    pub fn special_id(&self, name: &str) -> Result<u32> {
        let id = match name {
            "<|bos|>" => self.bos_id,
            "<|user_start|>" => 257,
            "<|user_end|>" => 258,
            "<|assistant_start|>" => 259,
            "<|assistant_end|>" => 260,
            "<|python_start|>" => 261,
            "<|python_end|>" => 262,
            "<|output_start|>" => 263,
            "<|output_end|>" => 264,
            _ => bail!("unknown special token '{name}'"),
        };
        Ok(id)
    }
    pub fn render_conversation(
        &self,
        conversation: &Conversation,
        max_tokens: usize,
    ) -> Result<RenderedConversation> {
        if conversation.messages.is_empty() {
            bail!("conversation must contain at least one message");
        }
        let mut ids = vec![self.bos_id];
        let mut loss_mask = vec![false];
        for (index, message) in conversation.messages.iter().enumerate() {
            let expected = if index % 2 == 0 { "user" } else { "assistant" };
            if message.role != expected {
                bail!(
                    "message {index} has role '{}', expected '{expected}'",
                    message.role
                );
            }
            let (start, end, supervised) = if message.role == "user" {
                (
                    self.special_id("<|user_start|>")?,
                    self.special_id("<|user_end|>")?,
                    false,
                )
            } else {
                (
                    self.special_id("<|assistant_start|>")?,
                    self.special_id("<|assistant_end|>")?,
                    true,
                )
            };
            ids.push(start);
            loss_mask.push(false);
            let content = self.encode(&message.content);
            ids.extend(&content);
            loss_mask.extend(std::iter::repeat_n(supervised, content.len()));
            ids.push(end);
            loss_mask.push(supervised);
        }
        ids.truncate(max_tokens);
        loss_mask.truncate(max_tokens);
        Ok(RenderedConversation { ids, loss_mask })
    }
    pub fn encode_id(&self, id: u32) -> Result<u32> {
        if id < self.vocab_size as u32 {
            Ok(id)
        } else {
            bail!("token id {id} exceeds vocabulary {}", self.vocab_size)
        }
    }
}
