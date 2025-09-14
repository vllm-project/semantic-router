// Official Candle BERT implementation based on Candle examples
// Reference: https://github.com/huggingface/candle/blob/main/candle-examples/examples/bert/main.rs

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;
use tokenizers::Tokenizer;

/// BERT classifier following Candle's official pattern
pub struct CandleBertClassifier {
    bert: BertModel,
    pooler: Linear, // BERT pooler layer (CLS token -> pooled output)
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleBertClassifier {
    /// Shared helper method for efficient batch tensor creation
    fn create_batch_tensors(
        &self,
        texts: &[&str],
    ) -> Result<(Tensor, Tensor, Tensor, Vec<tokenizers::Encoding>)> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(E::msg)?;

        let batch_size = texts.len();
        let max_len = encodings
            .iter()
            .map(|enc| enc.get_ids().len())
            .max()
            .unwrap_or(0);

        let total_elements = batch_size * max_len;
        let mut all_token_ids = Vec::with_capacity(total_elements);
        let mut all_attention_masks = Vec::with_capacity(total_elements);

        for encoding in &encodings {
            let token_ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();

            all_token_ids.extend_from_slice(token_ids);
            all_attention_masks.extend_from_slice(attention_mask);

            let padding_needed = max_len - token_ids.len();
            all_token_ids.extend(std::iter::repeat(0).take(padding_needed));
            all_attention_masks.extend(std::iter::repeat(0).take(padding_needed));
        }

        let token_ids =
            Tensor::new(all_token_ids.as_slice(), &self.device)?.reshape(&[batch_size, max_len])?;
        let attention_mask = Tensor::new(all_attention_masks.as_slice(), &self.device)?
            .reshape(&[batch_size, max_len])?;
        let token_type_ids = Tensor::zeros(&[batch_size, max_len], DType::U32, &self.device)?;

        Ok((token_ids, attention_mask, token_type_ids, encodings))
    }

    pub fn new(model_path: &str, num_classes: usize, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Load config
        let config_path = Path::new(model_path).join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| E::msg(format!("Failed to read config.json: {}", e)))?;

        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| E::msg(format!("Failed to parse config.json: {}", e)))?;

        // Load tokenizer
        let tokenizer_path = Path::new(model_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| E::msg(format!("Failed to load tokenizer: {}", e)))?;

        // Load model weights
        let weights_path = if Path::new(model_path).join("model.safetensors").exists() {
            Path::new(model_path).join("model.safetensors")
        } else if Path::new(model_path).join("pytorch_model.bin").exists() {
            Path::new(model_path).join("pytorch_model.bin")
        } else {
            return Err(E::msg("No model weights found"));
        };

        let use_pth = weights_path.extension().and_then(|s| s.to_str()) == Some("bin");

        // Create VarBuilder following Candle's official pattern
        let vb = if use_pth {
            VarBuilder::from_pth(&weights_path, DType::F32, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? }
        };

        // Load BERT model using Candle's official method
        // Support both BERT and RoBERTa naming conventions
        let (bert, pooler, classifier) = {
            // Try RoBERTa first, then fall back to BERT
            match BertModel::load(vb.pp("roberta"), &config) {
                Ok(bert) => {
                    // RoBERTa uses classifier.dense as pooler + classifier.out_proj as final classifier
                    let pooler = candle_nn::linear(
                        config.hidden_size,
                        config.hidden_size,
                        vb.pp("classifier").pp("dense"),
                    )?;
                    let classifier = candle_nn::linear(
                        config.hidden_size,
                        num_classes,
                        vb.pp("classifier").pp("out_proj"),
                    )?;
                    (bert, pooler, classifier)
                }
                Err(_) => {
                    // Fall back to BERT
                    let bert = BertModel::load(vb.pp("bert"), &config)?;
                    let pooler = candle_nn::linear(
                        config.hidden_size,
                        config.hidden_size,
                        vb.pp("bert").pp("pooler").pp("dense"),
                    )?;
                    let classifier =
                        candle_nn::linear(config.hidden_size, num_classes, vb.pp("classifier"))?;
                    (bert, pooler, classifier)
                }
            }
        };

        Ok(Self {
            bert,
            pooler,
            classifier,
            tokenizer,
            device,
        })
    }

    pub fn classify_text(&self, text: &str) -> Result<(usize, f32)> {
        // Tokenize following Candle's pattern
        let encoding = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();

        // Create tensors following Candle's pattern
        let token_ids = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let attention_mask = Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;

        // Forward pass through BERT - following official Candle BERT usage
        let sequence_output =
            self.bert
                .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Apply BERT pooler: CLS token -> linear -> tanh (standard BERT pooling)
        let cls_token = sequence_output.i((.., 0))?; // Take CLS token
        let pooled_output = self.pooler.forward(&cls_token)?;
        let pooled_output = pooled_output.tanh()?; // Apply tanh activation

        // Apply classifier
        let logits = self.classifier.forward(&pooled_output)?;

        // Apply softmax to get probabilities
        let probabilities = candle_nn::ops::softmax(&logits, 1)?;
        let probabilities = probabilities.squeeze(0)?;

        // Get predicted class and confidence
        let probabilities_vec = probabilities.to_vec1::<f32>()?;
        let (predicted_class, &confidence) = probabilities_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Ok((predicted_class, confidence))
    }

    /// True batch processing for multiple texts - significant performance improvement
    pub fn classify_batch(&self, texts: &[&str]) -> Result<Vec<(usize, f32)>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // OPTIMIZATION: Use shared tensor creation method
        let (token_ids, attention_mask, token_type_ids, _encodings) =
            self.create_batch_tensors(texts)?;

        // Batch BERT forward pass
        let sequence_output =
            self.bert
                .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // OPTIMIZATION: Use proper CLS token pooling instead of mean pooling
        let cls_tokens = sequence_output.i((.., 0))?; // Extract CLS tokens for all samples
        let pooled_output = self.pooler.forward(&cls_tokens)?;
        let pooled_output = pooled_output.tanh()?;

        let logits = self.classifier.forward(&pooled_output)?;
        let probabilities = candle_nn::ops::softmax(&logits, 1)?;

        // OPTIMIZATION: Batch result extraction
        let probs_data = probabilities.to_vec2::<f32>()?;
        let mut results = Vec::with_capacity(texts.len());

        for row in probs_data {
            let (predicted_class, confidence) = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, &conf)| (idx, conf))
                .unwrap_or((0, 0.0));

            results.push((predicted_class, confidence));
        }

        Ok(results)
    }
}

/// BERT token classifier for PII detection
pub struct CandleBertTokenClassifier {
    bert: BertModel,
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleBertTokenClassifier {
    /// Shared helper method for efficient batch tensor creation
    fn create_batch_tensors(
        &self,
        texts: &[&str],
    ) -> Result<(Tensor, Tensor, Tensor, Vec<tokenizers::Encoding>)> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(E::msg)?;

        let batch_size = texts.len();
        let max_len = encodings
            .iter()
            .map(|enc| enc.get_ids().len())
            .max()
            .unwrap_or(0);

        let total_elements = batch_size * max_len;
        let mut all_token_ids = Vec::with_capacity(total_elements);
        let mut all_attention_masks = Vec::with_capacity(total_elements);

        for encoding in &encodings {
            let token_ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();

            all_token_ids.extend_from_slice(token_ids);
            all_attention_masks.extend_from_slice(attention_mask);

            let padding_needed = max_len - token_ids.len();
            all_token_ids.extend(std::iter::repeat(0).take(padding_needed));
            all_attention_masks.extend(std::iter::repeat(0).take(padding_needed));
        }

        let token_ids =
            Tensor::new(all_token_ids.as_slice(), &self.device)?.reshape(&[batch_size, max_len])?;
        let attention_mask = Tensor::new(all_attention_masks.as_slice(), &self.device)?
            .reshape(&[batch_size, max_len])?;
        let token_type_ids = Tensor::zeros(&[batch_size, max_len], DType::U32, &self.device)?;

        Ok((token_ids, attention_mask, token_type_ids, encodings))
    }

    pub fn new(model_path: &str, num_classes: usize, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Load config
        let config_path = Path::new(model_path).join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;

        // Load tokenizer
        let tokenizer_path = Path::new(model_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        // Load weights
        let weights_path = if Path::new(model_path).join("model.safetensors").exists() {
            Path::new(model_path).join("model.safetensors")
        } else {
            Path::new(model_path).join("pytorch_model.bin")
        };

        let use_pth = weights_path.extension().and_then(|s| s.to_str()) == Some("bin");

        let vb = if use_pth {
            VarBuilder::from_pth(&weights_path, DType::F32, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? }
        };

        // Load BERT and token classifier - support both BERT and RoBERTa
        let (bert, classifier) = {
            // Try RoBERTa first, then fall back to BERT
            match BertModel::load(vb.pp("roberta"), &config) {
                Ok(bert) => {
                    println!("Detected RoBERTa token classifier - using RoBERTa naming");
                    let classifier =
                        candle_nn::linear(config.hidden_size, num_classes, vb.pp("classifier"))?;
                    (bert, classifier)
                }
                Err(_) => {
                    // Fall back to BERT
                    println!("Detected BERT token classifier - using BERT naming");
                    let bert = BertModel::load(vb.pp("bert"), &config)?;
                    let classifier =
                        candle_nn::linear(config.hidden_size, num_classes, vb.pp("classifier"))?;
                    (bert, classifier)
                }
            }
        };

        Ok(Self {
            bert,
            classifier,
            tokenizer,
            device,
        })
    }

    /// Helper method to extract entities from probabilities
    fn extract_entities_from_probs(
        &self,
        probs: &Tensor,
        tokens: &[String],
        offsets: &[(usize, usize)],
    ) -> Result<Vec<(String, usize, f32)>> {
        let probs_vec = probs.to_vec2::<f32>()?;
        let mut results = Vec::new();

        for (token_idx, (token, token_probs)) in tokens.iter().zip(probs_vec.iter()).enumerate() {
            if token_idx >= offsets.len() {
                break;
            }

            let (predicted_class, &confidence) = token_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &0.0));

            // Skip padding tokens and special tokens
            if token.starts_with("[PAD]")
                || token.starts_with("[CLS]")
                || token.starts_with("[SEP]")
            {
                continue;
            }

            results.push((token.clone(), predicted_class, confidence));
        }

        Ok(results)
    }

    /// True batch processing for token classification - significant performance improvement
    pub fn classify_tokens_batch(&self, texts: &[&str]) -> Result<Vec<Vec<(String, usize, f32)>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // OPTIMIZATION: Use shared tensor creation method
        let (token_ids, attention_mask, token_type_ids, encodings) =
            self.create_batch_tensors(texts)?;

        // Batch BERT forward pass
        let sequence_output =
            self.bert
                .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Batch token classification
        let logits = self.classifier.forward(&sequence_output)?; // (batch_size, seq_len, num_labels)
        let probabilities = candle_nn::ops::softmax(&logits, 2)?;

        // OPTIMIZATION: More efficient result extraction
        let mut batch_results = Vec::with_capacity(texts.len());
        for i in 0..texts.len() {
            let encoding = &encodings[i];
            let tokens = encoding.get_tokens();
            let offsets = encoding.get_offsets();

            let text_probs = probabilities.get(i)?; // (seq_len, num_labels)
            let text_results = self.extract_entities_from_probs(&text_probs, tokens, offsets)?;
            batch_results.push(text_results);
        }

        Ok(batch_results)
    }

    /// Single text token classification with span information (for backward compatibility)
    pub fn classify_tokens_with_spans(
        &self,
        text: &str,
    ) -> Result<Vec<(String, usize, f32, usize, usize)>> {
        // Use batch processing for single text
        let batch_results = self.classify_tokens_batch(&[text])?;
        if batch_results.is_empty() {
            return Ok(Vec::new());
        }

        // Get tokenization info for spans
        let encoding = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let offsets = encoding.get_offsets();

        let mut results = Vec::new();
        for (i, (token, class_id, confidence)) in batch_results[0].iter().enumerate() {
            if i < offsets.len() {
                let (start_char, end_char) = offsets[i];
                results.push((token.clone(), *class_id, *confidence, start_char, end_char));
            }
        }

        Ok(results)
    }

    /// Single text token classification (for backward compatibility)
    pub fn classify_tokens(&self, text: &str) -> Result<Vec<(String, usize, f32)>> {
        // Use batch processing for single text
        let batch_results = self.classify_tokens_batch(&[text])?;
        if batch_results.is_empty() {
            return Ok(Vec::new());
        }

        Ok(batch_results.into_iter().next().unwrap())
    }
}
