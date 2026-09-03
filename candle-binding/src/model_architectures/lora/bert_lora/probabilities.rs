//! Single-text classification for `HighPerformanceBertClassifier`, with and
//! without the full softmax probability distribution.
//!
//! Kept in its own file rather than growing `bert_lora.rs` further, since that
//! file is already over the repo's structure-check size limits.

use super::HighPerformanceBertClassifier;
use anyhow::{Error as E, Result};
use candle_core::{IndexOp, Tensor};
use candle_nn::Module;

impl HighPerformanceBertClassifier {
    /// Single text classification (following old architecture pattern exactly)
    pub fn classify_text(&self, text: &str) -> Result<(usize, f32)> {
        let (predicted_class, confidence, _) = self.classify_text_internal(text)?;
        Ok((predicted_class, confidence))
    }

    /// Classify text and return the top-1 prediction together with the full
    /// softmax probability distribution across all classes. This lets callers
    /// read the probability of a specific class directly instead of only the
    /// confidence of whichever class wins argmax.
    pub fn classify_text_with_probabilities(&self, text: &str) -> Result<(usize, f32, Vec<f32>)> {
        self.classify_text_internal(text)
    }

    fn classify_text_internal(&self, text: &str) -> Result<(usize, f32, Vec<f32>)> {
        // Tokenize following old architecture pattern
        let encoding = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let token_ids = encoding.get_ids();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        // Create tensors following old architecture pattern
        let token_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let attention_mask = Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;

        // Forward pass through BERT - following old architecture pattern exactly
        let sequence_output =
            self.bert
                .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Apply BERT pooler: CLS token -> linear -> tanh (old architecture pattern)
        let cls_token = sequence_output.i((.., 0))?; // Take CLS token
        let pooled_output = self.pooler.forward(&cls_token)?;
        let pooled_output = pooled_output.tanh()?; // Apply tanh activation

        // Apply classifier
        let logits = self.classifier.forward(&pooled_output)?;

        // Apply softmax to get probabilities (old architecture pattern)
        let probabilities = candle_nn::ops::softmax(&logits, 1)?;
        let probabilities = probabilities.squeeze(0)?;

        // Get predicted class and confidence
        let probabilities_vec = probabilities.to_vec1::<f32>()?;
        let (predicted_class, &confidence) = probabilities_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Ok((predicted_class, confidence, probabilities_vec))
    }
}
