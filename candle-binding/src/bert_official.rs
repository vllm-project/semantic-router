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
}

/// BERT token classifier for PII detection
pub struct CandleBertTokenClassifier {
    bert: BertModel,
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleBertTokenClassifier {
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

    pub fn classify_tokens(&self, text: &str) -> Result<Vec<(String, usize, f32)>> {
        // Tokenize
        let encoding = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();
        let tokens = encoding.get_tokens();

        // Create tensors
        let token_ids = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let attention_mask = Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;

        // Forward pass
        let sequence_output =
            self.bert
                .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Apply token classifier to each token
        let logits = self.classifier.forward(&sequence_output)?;

        // Get predictions for each token
        let probabilities = candle_nn::ops::softmax(&logits, 2)?;
        let probabilities = probabilities.squeeze(0)?;
        let probabilities_vec = probabilities.to_vec2::<f32>()?;

        let mut results = Vec::new();
        for (token, probs) in tokens.iter().zip(probabilities_vec.iter()) {
            let (predicted_class, &confidence) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            results.push((token.clone(), predicted_class, confidence));
        }

        Ok(results)
    }

    pub fn classify_tokens_with_spans(
        &self,
        text: &str,
    ) -> Result<Vec<(String, usize, f32, usize, usize)>> {
        // Tokenize with offset mapping
        let encoding = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();
        let tokens = encoding.get_tokens();
        let offsets = encoding.get_offsets();

        // Create tensors
        let token_ids = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let attention_mask = Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;

        // Forward pass
        let sequence_output =
            self.bert
                .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Apply token classifier to each token
        let logits = self.classifier.forward(&sequence_output)?;

        // Get predictions for each token
        let probabilities = candle_nn::ops::softmax(&logits, 2)?;
        let probabilities = probabilities.squeeze(0)?;
        let probabilities_vec = probabilities.to_vec2::<f32>()?;

        let mut results = Vec::new();
        for ((token, offset), probs) in tokens
            .iter()
            .zip(offsets.iter())
            .zip(probabilities_vec.iter())
        {
            let (predicted_class, &confidence) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            results.push((
                token.clone(),
                predicted_class,
                confidence,
                offset.0,
                offset.1,
            ));
        }

        Ok(results)
    }
}
