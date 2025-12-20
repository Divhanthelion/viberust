//! Tokenizer wrapper for text encoding/decoding

use anyhow::Result;
use tokenizers::Tokenizer as HfTokenizer;

/// Tokenizer wrapper
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    /// Load tokenizer from file
    pub fn from_file(path: &str) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { inner })
    }

    /// Create a simple fallback tokenizer for testing
    pub fn simple() -> Self {
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        
        let mut tokenizer = HfTokenizer::new(BPE::default());
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
        
        Self { inner: tokenizer }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get special token IDs
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner.token_to_id("<bos>").or_else(|| self.inner.token_to_id("<s>"))
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner.token_to_id("<eos>").or_else(|| self.inner.token_to_id("</s>"))
    }

    pub fn pad_token_id(&self) -> Option<u32> {
        self.inner.token_to_id("<pad>")
    }
}
