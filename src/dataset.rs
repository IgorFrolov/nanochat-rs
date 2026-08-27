use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
};

/// Sequential, bounded-memory token source. Text files are read line by line;
/// `.u32` files contain little-endian uint32 token ids.
pub struct StreamingDataset {
    reader: BufReader<File>,
    path: PathBuf,
    kind: DatasetKind,
    tokenizer: Tokenizer,
    pending: Vec<u32>,
}

enum DatasetKind {
    Text,
    U32,
}

impl StreamingDataset {
    pub fn open(path: impl AsRef<Path>, tokenizer: Tokenizer) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("opening dataset {}", path.display()))?;
        let kind = if path.extension().and_then(|x| x.to_str()) == Some("u32") {
            DatasetKind::U32
        } else {
            DatasetKind::Text
        };
        Ok(Self {
            reader: BufReader::new(file),
            path: path.to_path_buf(),
            kind,
            tokenizer,
            pending: Vec::new(),
        })
    }

    fn fill(&mut self) -> Result<bool> {
        match self.kind {
            DatasetKind::Text => {
                let mut line = String::new();
                if self.reader.read_line(&mut line)? == 0 {
                    return Ok(false);
                }
                self.pending.extend(self.tokenizer.encode(&line));
            }
            DatasetKind::U32 => {
                let mut bytes = [0u8; 4];
                if self.reader.read(&mut bytes[..1])? == 0 {
                    return Ok(false);
                }
                self.reader
                    .read_exact(&mut bytes[1..])
                    .context("u32 dataset ended with a partial token")?;
                for _ in 0..1023 {
                    let id = u32::from_le_bytes(bytes);
                    self.pending.push(self.tokenizer.encode_id(id)?);
                    if self.reader.read(&mut bytes[..1])? == 0 {
                        break;
                    }
                    self.reader
                        .read_exact(&mut bytes[1..])
                        .context("u32 dataset ended with a partial token")?;
                }
            }
        }
        Ok(true)
    }

    pub fn next_window(&mut self, sequence_length: usize) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
        while self.pending.len() < sequence_length + 1 {
            if !self.fill()? {
                self.reader = BufReader::new(File::open(&self.path)?);
                if self.pending.is_empty() && !self.fill()? {
                    return Ok(None);
                }
            }
        }
        let input = self.pending[..sequence_length].to_vec();
        let target = self.pending[1..sequence_length + 1].to_vec();
        self.pending.drain(..sequence_length);
        Ok(Some((input, target)))
    }
}
