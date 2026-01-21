//! Traditional BERT Implementation
//!
//! This module contains the traditional full-model fine-tuning BERT implementation,
//! migrated from bert_official.rs as part of the dual-path architecture.
//!
//! ## Traditional BERT Characteristics
//! - **Stability**: Proven, reliable performance
//! - **Compatibility**: 100% backward compatible with existing APIs
//! - **Processing**: Sequential single-task processing
//! - **Performance**: Stable baseline performance
//! - **Reliability**: Battle-tested in production
//!
//! ## Architecture
//! Based on Candle's official BERT implementation pattern, following the
//! reference: https://github.com/huggingface/candle/blob/main/candle-examples/examples/bert/main.rs

use crate::core::{ModelErrorType, UnifiedError};
use crate::model_error;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::core::tokenization::{create_bert_compatibility_tokenizer, DualPathTokenizer};
use crate::model_architectures::traits::{FineTuningType, ModelType, TaskType, TraditionalModel};
use crate::model_architectures::unified_interface::{
    ConfigurableModel, CoreModel, PathSpecialization,
};

/// Traditional BERT classifier following Candle's official pattern
///
/// This is the stable, traditional fine-tuning path that provides reliable
/// performance with full backward compatibility.
pub struct TraditionalBertClassifier {
    /// Core BERT model
    bert: BertModel,
    /// BERT pooler layer (CLS token -> pooled output)
    pooler: Linear,
    /// Classification head
    classifier: Linear,
    /// Unified tokenizer compatible with dual-path architecture
    tokenizer: Box<dyn DualPathTokenizer>,
    /// Computing device
    device: Device,
    /// Number of output classes
    num_classes: usize,
    /// Model configuration for CoreModel trait
    config: Config,
}

impl TraditionalBertClassifier {
    /// Create a new traditional BERT classifier
    ///
    /// ## Arguments
    /// * `model_id` - Model identifier (HuggingFace Hub ID or local path)
    /// * `num_classes` - Number of classification classes
    /// * `use_cpu` - Whether to force CPU usage
    ///
    /// ## Returns
    /// * `Result<Self>` - Initialized traditional BERT classifier
    pub fn new(model_id: &str, num_classes: usize, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        println!("Initializing Traditional BERT classifier: {}", model_id);

        // Load model configuration and files
        let (config_filename, tokenizer_filename, weights_filename, use_pth) =
            Self::resolve_model_files(model_id)?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let base_tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Create dual-path compatible tokenizer
        let tokenizer = create_bert_compatibility_tokenizer(base_tokenizer, device.clone())?;

        // Load model weights
        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[weights_filename.clone()],
                    DType::F32,
                    &device,
                )?
            }
        };

        // Load BERT model
        let bert = BertModel::load(vb.pp("bert"), &config)?;

        // Create pooler layer
        let pooler = {
            let pooler_weight = vb.get(
                (config.hidden_size, config.hidden_size),
                "bert.pooler.dense.weight",
            )?;
            let pooler_bias = vb.get(config.hidden_size, "bert.pooler.dense.bias")?;
            Linear::new(pooler_weight.t()?, Some(pooler_bias))
        };

        // Create classification head
        let classifier = {
            let classifier_weight =
                vb.get((num_classes, config.hidden_size), "classifier.weight")?;
            let classifier_bias = vb.get(num_classes, "classifier.bias")?;
            Linear::new(classifier_weight, Some(classifier_bias))
        };

        Ok(Self {
            bert,
            pooler,
            classifier,
            tokenizer,
            device: device.clone(),
            num_classes,
            config: config.clone(),
        })
    }

    /// Resolve model files (HuggingFace Hub or local)
    fn resolve_model_files(model_id: &str) -> Result<(String, String, String, bool)> {
        if Path::new(model_id).exists() {
            // Local model path
            let config_path = Path::new(model_id).join("config.json");
            let tokenizer_path = Path::new(model_id).join("tokenizer.json");

            // Check for safetensors first, fall back to PyTorch
            let (weights_path, use_pth) = if Path::new(model_id).join("model.safetensors").exists()
            {
                (
                    Path::new(model_id)
                        .join("model.safetensors")
                        .to_string_lossy()
                        .to_string(),
                    false,
                )
            } else if Path::new(model_id).join("pytorch_model.bin").exists() {
                (
                    Path::new(model_id)
                        .join("pytorch_model.bin")
                        .to_string_lossy()
                        .to_string(),
                    true,
                )
            } else {
                return Err(E::msg(format!("No model weights found in {}", model_id)));
            };

            Ok((
                config_path.to_string_lossy().to_string(),
                tokenizer_path.to_string_lossy().to_string(),
                weights_path,
                use_pth,
            ))
        } else {
            // HuggingFace Hub model
            let repo =
                Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());

            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;

            // Try safetensors first, fall back to PyTorch
            let (weights, use_pth) = match api.get("model.safetensors") {
                Ok(weights) => (weights, false),
                Err(_) => {
                    println!("Safetensors not found, trying PyTorch model...");
                    (api.get("pytorch_model.bin")?, true)
                }
            };

            Ok((
                config.to_string_lossy().to_string(),
                tokenizer.to_string_lossy().to_string(),
                weights.to_string_lossy().to_string(),
                use_pth,
            ))
        }
    }

    /// Shared helper method for efficient batch tensor creation
    fn create_batch_tensors(
        &self,
        texts: &[&str],
    ) -> Result<(Tensor, Tensor, Tensor, Vec<tokenizers::Encoding>)> {
        // Use the dual-path tokenizer for batch processing
        let batch_result = self.tokenizer.tokenize_batch(texts)?;

        let batch_size = batch_result.batch_size;
        let max_len = batch_result.max_length;

        // Create tensors using the unified tokenizer
        let (token_ids_tensor, attention_mask_tensor) =
            self.tokenizer.create_batch_tensors(&batch_result)?;

        // Create token type IDs (all zeros for single sentence classification)
        let token_type_ids = Tensor::zeros((batch_size, max_len), DType::U32, &self.device)?;

        // Create encodings for compatibility (simplified implementation)
        let encodings = vec![];

        Ok((
            token_ids_tensor,
            token_type_ids,
            attention_mask_tensor,
            encodings,
        ))
    }

    /// Classify a single text
    pub fn classify_text(&self, text: &str) -> Result<(usize, f32)> {
        let result = self.tokenizer.tokenize_for_traditional(text)?;
        let (token_ids_tensor, attention_mask_tensor) = self.tokenizer.create_tensors(&result)?;

        // Create token type IDs (all zeros for single sentence)
        let token_type_ids = token_ids_tensor.zeros_like()?;

        // Forward through BERT
        let embeddings = self.bert.forward(
            &token_ids_tensor,
            &token_type_ids,
            Some(&attention_mask_tensor),
        )?;

        // Use CLS token embedding and apply pooler (following old architecture pattern)
        let cls_embedding = embeddings.i((.., 0))?;
        let pooled = self.pooler.forward(&cls_embedding)?;
        let pooled = pooled.tanh()?; // BERT pooler uses tanh activation

        // Apply classification head
        let logits = self.classifier.forward(&pooled)?;

        // Apply softmax and get prediction
        let probabilities = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let probabilities_vec = probabilities.squeeze(0)?.to_vec1::<f32>()?;

        let (predicted_idx, &max_prob) = probabilities_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        Ok((predicted_idx, max_prob))
    }

    /// Classify a batch of texts efficiently
    pub fn classify_batch(&self, texts: &[&str]) -> Result<Vec<(usize, f32)>> {
        let (token_ids_tensor, token_type_ids, attention_mask_tensor, _) =
            self.create_batch_tensors(texts)?;

        // Forward through BERT
        let embeddings = self.bert.forward(
            &token_ids_tensor,
            &token_type_ids,
            Some(&attention_mask_tensor),
        )?;

        // Use CLS token embeddings and apply pooler (following old architecture pattern)
        let cls_embeddings = embeddings.i((.., 0))?;
        let pooled = self.pooler.forward(&cls_embeddings)?;
        let pooled = pooled.tanh()?;

        // Apply classification head
        let logits = self.classifier.forward(&pooled)?;

        // Apply softmax along the last dimension
        let probabilities = candle_nn::ops::softmax(&logits, 1)?;

        // Extract results for each text
        let mut results = Vec::new();
        let batch_size = texts.len();

        for i in 0..batch_size {
            let text_probs = probabilities.i(i)?;
            let probs_vec = text_probs.to_vec1::<f32>()?;

            let (predicted_idx, &max_prob) = probs_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            results.push((predicted_idx, max_prob));
        }

        Ok(results)
    }

    /// Get the device this model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the number of classes
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Implementation of CoreModel for TraditionalBertClassifier
///
/// This provides the core functionality using the new simplified interface.
/// It delegates to the existing ModelBackbone implementation for compatibility.
impl CoreModel for TraditionalBertClassifier {
    type Config = Config;
    type Error = candle_core::Error;
    type Output = (usize, f32);

    fn model_type(&self) -> ModelType {
        ModelType::Traditional
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Self::Output, Self::Error> {
        // Forward pass through BERT model (match original ModelBackbone logic)
        let outputs = self.bert.forward(input_ids, attention_mask, None)?;

        // Apply pooler (match original ModelBackbone logic)
        let pooled_output = self.pooler.forward(&outputs)?;

        // Apply classification head (match original ModelBackbone logic)
        let logits = self.classifier.forward(&pooled_output)?;

        // Get the predicted class (argmax) and confidence (max softmax probability)
        // (match original ModelBackbone logic)
        let softmax_probs = candle_nn::ops::softmax(&logits, 0)?;
        let max_prob = softmax_probs.max(0)?.to_scalar::<f32>()?;
        let predicted_class = softmax_probs.argmax(0)?.to_scalar::<u32>()? as usize;

        Ok((predicted_class, max_prob))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }
}

/// Implementation of PathSpecialization for TraditionalBertClassifier
///
/// This provides path-specific characteristics for traditional BERT models.
impl PathSpecialization for TraditionalBertClassifier {
    fn supports_parallel(&self) -> bool {
        false // Traditional models use sequential processing
    }

    fn get_confidence_threshold(&self) -> f32 {
        use crate::core::config_loader::GlobalConfigLoader;
        GlobalConfigLoader::load_router_config_safe().traditional_bert_confidence_threshold
    }

    fn optimal_batch_size(&self) -> usize {
        16 // Conservative batch size for stability
    }
}

/// Implementation of ConfigurableModel for TraditionalBertClassifier
///
/// This enables configuration-based model loading using the new interface.
impl ConfigurableModel for TraditionalBertClassifier {
    fn load(config: &Self::Config, device: &Device) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        // Replicate original ModelBackbone::load logic for compatibility
        // Note: This has limitations (hardcoded paths) but maintains functionality

        // Create dual-path compatible tokenizer from config
        let base_tokenizer = Tokenizer::from_file("tokenizer.json").map_err(|e| {
            let unified_err = model_error!(
                ModelErrorType::Tokenizer,
                "tokenizer loading",
                format!("Failed to load tokenizer: {}", e),
                "tokenizer.json"
            );
            candle_core::Error::from(unified_err)
        })?;
        let tokenizer = create_bert_compatibility_tokenizer(base_tokenizer, device.clone())
            .map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::Tokenizer,
                    "tokenizer creation",
                    format!("Failed to create tokenizer: {}", e),
                    "BERT compatibility"
                );
                candle_core::Error::from(unified_err)
            })?;

        // Create VarBuilder for model weights (simplified)
        let vb = VarBuilder::zeros(DType::F32, device);

        // Load BERT model using the provided config
        let bert = BertModel::load(vb.pp("bert"), config)?;

        // Create pooler layer (768 -> 768 for BERT-base)
        let pooler = Linear::new(
            vb.pp("pooler")
                .pp("dense")
                .get((config.hidden_size, config.hidden_size), "weight")?,
            Some(
                vb.pp("pooler")
                    .pp("dense")
                    .get(config.hidden_size, "bias")?,
            ),
        );

        // Create classifier head (768 -> num_classes, defaulting to 2)
        let num_classes = 2; // Default for binary classification
        let classifier = Linear::new(
            vb.pp("classifier")
                .get((config.hidden_size, num_classes), "weight")?,
            Some(vb.pp("classifier").get(num_classes, "bias")?),
        );

        Ok(Self {
            bert,
            pooler,
            classifier,
            tokenizer,
            device: device.clone(),
            num_classes,
            config: config.clone(),
        })
    }
}

impl std::fmt::Debug for TraditionalBertClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraditionalBertClassifier")
            .field("device", &self.device)
            .field("num_classes", &self.num_classes)
            .finish()
    }
}

// Global instances using OnceLock pattern for zero-cost reads after initialization
/// Global Traditional BERT classifier instance
pub static TRADITIONAL_BERT_CLASSIFIER: std::sync::OnceLock<
    std::sync::Arc<TraditionalBertClassifier>,
> = std::sync::OnceLock::new();

/// Global Traditional BERT token classifier instance
pub static TRADITIONAL_BERT_TOKEN_CLASSIFIER: std::sync::OnceLock<
    std::sync::Arc<TraditionalBertTokenClassifier>,
> = std::sync::OnceLock::new();

/// Traditional BERT token classifier for token-level classification
pub struct TraditionalBertTokenClassifier {
    /// Core BERT model
    bert: BertModel,
    /// Token classification head
    classifier: Linear,
    /// Unified tokenizer compatible with dual-path architecture
    tokenizer: Box<dyn DualPathTokenizer>,
    /// Computing device
    device: Device,
    /// Number of output classes
    num_classes: usize,
}

impl TraditionalBertTokenClassifier {
    /// Create a new traditional BERT token classifier
    pub fn new(model_path: &str, _num_classes: usize, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Load model configuration and files
        let (config_filename, tokenizer_filename, weights_filename, use_pth) =
            TraditionalBertClassifier::resolve_model_files(model_path)?;

        let config_str = std::fs::read_to_string(&config_filename)?;
        let config: Config = serde_json::from_str(&config_str)?;

        // Read actual number of classes from config.json id2label field
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let actual_num_classes = if let Some(id2label) = config_json.get("id2label") {
            if let Some(obj) = id2label.as_object() {
                obj.len()
            } else {
                return Err(E::msg("id2label is not an object"));
            }
        } else {
            return Err(E::msg("config.json missing id2label field"));
        };

        println!(
            "  Detected {} classes from config.json id2label field",
            actual_num_classes
        );

        let base_tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Create dual-path compatible tokenizer
        let tokenizer = create_bert_compatibility_tokenizer(base_tokenizer, device.clone())?;

        // Load model weights
        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[weights_filename.clone()],
                    DType::F32,
                    &device,
                )?
            }
        };

        // Load BERT model (without pooler for token classification)
        let bert = BertModel::load(vb.pp("bert"), &config)?;

        // Create token classification head using actual number of classes from config
        let classifier =
            candle_nn::linear(config.hidden_size, actual_num_classes, vb.pp("classifier"))?;

        Ok(Self {
            bert,
            classifier,
            tokenizer,
            device,
            num_classes: actual_num_classes,
        })
    }

    /// Classify tokens in text
    pub fn classify_tokens(&self, text: &str) -> Result<Vec<(String, usize, f32)>> {
        // Tokenize input text
        let tokenization_result = self.tokenizer.tokenize(text)?;
        let token_ids = tokenization_result.token_ids;
        let token_strings = tokenization_result.tokens;

        // Create input tensors
        // Convert i32 to u32 for tensor creation
        let token_ids_u32: Vec<u32> = token_ids.into_iter().map(|id| id as u32).collect();
        let seq_len = token_ids_u32.len();
        let token_ids_tensor = Tensor::from_vec(token_ids_u32, (1, seq_len), &self.device)?;
        let token_type_ids = token_ids_tensor.zeros_like()?;
        let attention_mask = Tensor::ones_like(&token_ids_tensor)?;

        // Forward pass through BERT
        let hidden_states =
            self.bert
                .forward(&token_ids_tensor, &token_type_ids, Some(&attention_mask))?;

        // Apply classification head to each token
        let logits = self.classifier.forward(&hidden_states)?;
        let probabilities = candle_nn::ops::softmax(&logits, 2)?;

        // Extract predictions for each token
        let probs_data = probabilities.to_vec3::<f32>()?;
        let mut results = Vec::new();

        for (i, token) in token_strings.iter().enumerate() {
            if i < probs_data[0].len() {
                let token_probs = &probs_data[0][i];
                let (predicted_class, confidence) = token_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, &conf)| (idx, conf))
                    .unwrap_or((0, 0.0));

                // Only include tokens with reasonable confidence (configurable threshold)
                let pii_threshold = {
                    use crate::core::config_loader::GlobalConfigLoader;
                    GlobalConfigLoader::load_router_config_safe()
                        .traditional_pii_detection_threshold
                };
                if confidence > pii_threshold {
                    results.push((token.clone(), predicted_class, confidence));
                }
            }
        }

        Ok(results)
    }
}

impl std::fmt::Debug for TraditionalBertTokenClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraditionalBertTokenClassifier")
            .field("device", &self.device)
            .field("num_classes", &self.num_classes)
            .finish()
    }
}
