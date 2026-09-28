//! Offset-preserving entry points for owned merged BERT task instances.
use super::{HighPerformanceBertClassifier, HighPerformanceBertTokenClassifier};
use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::Module;

impl HighPerformanceBertClassifier {
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl HighPerformanceBertTokenClassifier {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn classify_tokens_with_offsets(
        &self,
        text: &str,
    ) -> Result<Vec<crate::core::tokenization::TokenPrediction>> {
        let encoding = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let mask = Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;
        let states = self.bert.forward(&ids, &ids.zeros_like()?, Some(&mask))?;
        let probabilities = candle_nn::ops::softmax(&self.classifier.forward(&states)?, 2)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
        Ok(probabilities
            .iter()
            .zip(encoding.get_tokens())
            .zip(encoding.get_offsets())
            .map(|((probs, token), &(start, end))| {
                let (class, confidence) = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, p)| (i, *p))
                    .unwrap_or((0, 0.0));
                (token.clone(), class, confidence, start, end)
            })
            .collect())
    }
}
