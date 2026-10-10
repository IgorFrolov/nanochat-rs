use crate::tokenizer::{ModelTokenizer, Tokenizer};
use anyhow::{Context, Result};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

/// Sequential, bounded-memory token source. Text files are read line by line;
/// `.u32` files contain little-endian uint32 token ids.
pub struct StreamingDataset {
    reader: BufReader<File>,
    path: PathBuf,
    kind: DatasetKind,
    tokenizer: ModelTokenizer,
    pending: Vec<u32>,
}

enum DatasetKind {
    Text,
    U32,
}

impl StreamingDataset {
    pub fn open(path: impl AsRef<Path>, tokenizer: Tokenizer) -> Result<Self> {
        Self::open_with_tokenizer(path, ModelTokenizer::Byte(tokenizer))
    }

    pub fn open_with_tokenizer(path: impl AsRef<Path>, tokenizer: ModelTokenizer) -> Result<Self> {
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
                for index in 0..1024 {
                    let mut bytes = [0u8; 4];
                    if self.reader.read(&mut bytes[..1])? == 0 {
                        return Ok(index > 0);
                    }
                    self.reader
                        .read_exact(&mut bytes[1..])
                        .context("u32 dataset ended with a partial token")?;
                    let id = u32::from_le_bytes(bytes);
                    anyhow::ensure!(
                        (id as usize) < self.tokenizer.vocab_size(),
                        "token id {id} exceeds vocabulary"
                    );
                    self.pending.push(id);
                }
            }
        }
        Ok(true)
    }

    pub fn next_window(&mut self, sequence_length: usize) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
        anyhow::ensure!(sequence_length > 0, "sequence length must be nonzero");
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StreamingState {
    pub offset: u64,
    pub pending: Vec<u32>,
}
impl StreamingDataset {
    pub fn state(&mut self) -> Result<StreamingState> {
        Ok(StreamingState {
            offset: self.reader.stream_position()?,
            pending: self.pending.clone(),
        })
    }
    pub fn restore(&mut self, state: &StreamingState) -> Result<()> {
        anyhow::ensure!(
            state.offset <= self.reader.get_ref().metadata()?.len(),
            "dataset cursor exceeds file size"
        );
        anyhow::ensure!(
            state
                .pending
                .iter()
                .all(|&id| (id as usize) < self.tokenizer.vocab_size()),
            "invalid buffered token"
        );
        self.reader.seek(SeekFrom::Start(state.offset))?;
        self.pending = state.pending.clone();
        Ok(())
    }
}
