//! Traditional ModernBERT Implementation - Dual Path Architecture
//!
//! This module provides the traditional fine-tuning ModernBERT implementation
//! that preserves all bug fixes from FixedModernBertClassifier.

use crate::core::{config_errors, processing_errors, ModelErrorType, UnifiedError};
use crate::model_error;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert::{
    ClassifierConfig, ClassifierPooling, Config, ModernBert,
};
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

use crate::core::tokenization::DualPathTokenizer;
use crate::model_architectures::traits::*;
use crate::model_architectures::unified_interface::{
    ConfigurableModel, CoreModel, PathSpecialization,
};

/// Traditional ModernBERT sequence classifier
pub struct TraditionalModernBertClassifier {
    model: ModernBert,
    head: Option<FixedModernBertHead>,
    classifier: FixedModernBertClassifier,
    classifier_pooling: ClassifierPooling,
    tokenizer: Box<dyn DualPathTokenizer>,
    device: Device,
    config: Config,
    num_classes: usize,
}

/// Traditional ModernBERT token classifier
pub struct TraditionalModernBertTokenClassifier {
    model: ModernBert,
    head: Option<FixedModernBertHead>,
    classifier: FixedModernBertTokenClassifier,
    tokenizer: Box<dyn DualPathTokenizer>,
    device: Device,
    config: Config,
    num_classes: usize,
    model_path: String,
}

// Global static instances for FFI compatibility
lazy_static! {
    pub static ref TRADITIONAL_MODERNBERT_CLASSIFIER: Arc<Mutex<Option<TraditionalModernBertClassifier>>> =
        Arc::new(Mutex::new(None));
    pub static ref TRADITIONAL_MODERNBERT_PII_CLASSIFIER: Arc<Mutex<Option<TraditionalModernBertClassifier>>> =
        Arc::new(Mutex::new(None));
    pub static ref TRADITIONAL_MODERNBERT_JAILBREAK_CLASSIFIER: Arc<Mutex<Option<TraditionalModernBertClassifier>>> =
        Arc::new(Mutex::new(None));
    pub static ref TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER: Arc<Mutex<Option<TraditionalModernBertTokenClassifier>>> =
        Arc::new(Mutex::new(None));
}

// Real classifier implementations
#[derive(Clone)]
pub struct FixedModernBertHead {
    dense: candle_nn::Linear,
    layer_norm: candle_nn::LayerNorm,
}

#[derive(Clone)]
pub struct FixedModernBertClassifier {
    classifier: candle_nn::Linear,
}

#[derive(Clone)]
pub struct FixedModernBertTokenClassifier {
    classifier: candle_nn::Linear,
}

impl FixedModernBertHead {
    pub fn load(vb: candle_nn::VarBuilder, config: &Config) -> Result<Self, candle_core::Error> {
        // Following old architecture pattern - no bias for dense layer
        let dense = candle_nn::Linear::new(
            vb.get((config.hidden_size, config.hidden_size), "dense.weight")?,
            None, // No bias in this model
        );

        // Load layer norm - following old architecture pattern
        let layer_norm = candle_nn::LayerNorm::new(
            vb.get((config.hidden_size,), "norm.weight")?,
            // Create a zero bias tensor since LayerNorm::new requires it but the model doesn't have one
            candle_core::Tensor::zeros((config.hidden_size,), DType::F32, vb.device())?,
            1e-12,
        );

        Ok(Self { dense, layer_norm })
    }
}

impl candle_nn::Module for FixedModernBertHead {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = xs.apply(&self.dense)?;
        let xs = xs.gelu()?; // GELU activation
        xs.apply(&self.layer_norm)
    }
}

/// Implementation of CoreModel for TraditionalModernBertClassifier
impl CoreModel for TraditionalModernBertClassifier {
    type Config = String;
    type Error = candle_core::Error;
    type Output = (usize, f32);

    fn model_type(&self) -> ModelType {
        ModelType::Traditional
    }

    fn forward(
        &self,
        _input_ids: &Tensor,
        _attention_mask: &Tensor,
    ) -> Result<Self::Output, Self::Error> {
        // Placeholder implementation (match original ModelBackbone logic)
        let default_confidence = {
            use crate::core::config_loader::GlobalConfigLoader;
            GlobalConfigLoader::load_router_config_safe()
                .traditional_modernbert_confidence_threshold
        };
        Ok((0, default_confidence))
    }

    fn get_config(&self) -> &Self::Config {
        // CoreModel requires get_config but original ModelBackbone didn't have it
        // Since Config type is String but struct stores Config, we use lazy_static for String
        use std::sync::OnceLock;
        static DEFAULT_CONFIG: OnceLock<String> = OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| "modernbert-base".to_string())
    }
}

/// Implementation of PathSpecialization for TraditionalModernBertClassifier
impl PathSpecialization for TraditionalModernBertClassifier {
    fn supports_parallel(&self) -> bool {
        false // Match original ModelBackbone value
    }

    fn get_confidence_threshold(&self) -> f32 {
        use crate::core::config_loader::GlobalConfigLoader;
        GlobalConfigLoader::load_router_config_safe().traditional_modernbert_confidence_threshold
    }

    fn optimal_batch_size(&self) -> usize {
        16 // Conservative batch size for stability
    }
}

/// Implementation of ConfigurableModel for TraditionalModernBertClassifier
impl ConfigurableModel for TraditionalModernBertClassifier {
    fn load(_config: &Self::Config, _device: &Device) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        // Placeholder implementation (match original ModelBackbone logic)
        let unified_err = model_error!(
            ModelErrorType::ModernBERT,
            "trait implementation",
            "Not implemented yet - use TraditionalModernBertClassifier::new() instead",
            "TraditionalModel trait"
        );
        Err(candle_core::Error::from(unified_err))
    }
}

/// Implementation of CoreModel for TraditionalModernBertTokenClassifier
impl CoreModel for TraditionalModernBertTokenClassifier {
    type Config = String;
    type Error = candle_core::Error;
    type Output = Vec<(String, usize, f32)>;

    fn model_type(&self) -> ModelType {
        ModelType::Traditional
    }

    fn forward(
        &self,
        _input_ids: &Tensor,
        _attention_mask: &Tensor,
    ) -> Result<Self::Output, Self::Error> {
        // Placeholder implementation (match original ModelBackbone logic)
        let token_threshold = {
            use crate::core::config_loader::GlobalConfigLoader;
            GlobalConfigLoader::load_router_config_safe().traditional_token_classification_threshold
        };
        Ok(vec![("O".to_string(), 0, token_threshold)])
    }

    fn get_config(&self) -> &Self::Config {
        // CoreModel requires get_config but original ModelBackbone didn't have it
        // Since Config type is String but struct stores Config, we use lazy_static for String
        use std::sync::OnceLock;
        static DEFAULT_CONFIG: OnceLock<String> = OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| "modernbert-base-token".to_string())
    }
}

/// Implementation of PathSpecialization for TraditionalModernBertTokenClassifier
impl PathSpecialization for TraditionalModernBertTokenClassifier {
    fn supports_parallel(&self) -> bool {
        false // Match original ModelBackbone value
    }

    fn get_confidence_threshold(&self) -> f32 {
        use crate::core::config_loader::GlobalConfigLoader;
        GlobalConfigLoader::load_router_config_safe().traditional_modernbert_confidence_threshold
    }

    fn optimal_batch_size(&self) -> usize {
        16 // Conservative batch size for stability
    }
}

/// Implementation of ConfigurableModel for TraditionalModernBertTokenClassifier
impl ConfigurableModel for TraditionalModernBertTokenClassifier {
    fn load(_config: &Self::Config, _device: &Device) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        // Placeholder implementation (match original ModelBackbone logic)
        let unified_err = model_error!(
            ModelErrorType::ModernBERT,
            "trait implementation",
            "Not implemented yet - use TraditionalModernBertClassifier::new() instead",
            "TokenClassifier trait"
        );
        Err(candle_core::Error::from(unified_err))
    }
}

impl FixedModernBertClassifier {
    pub fn load(vb: candle_nn::VarBuilder, config: &Config) -> Result<Self, candle_core::Error> {
        // Try to get num_classes from classifier_config, fallback to 2
        let num_classes = if let Some(ref cc) = config.classifier_config {
            cc.id2label.len()
        } else {
            2
        };

        let classifier = candle_nn::linear(config.hidden_size, num_classes, vb.pp("classifier"))?;

        Ok(Self { classifier })
    }

    pub fn load_with_classes(
        vb: candle_nn::VarBuilder,
        config: &Config,
        num_classes: usize,
    ) -> Result<Self, candle_core::Error> {
        // Load pre-trained classifier weights (match old architecture)
        let weight = vb.get((num_classes, config.hidden_size), "weight")?;
        let bias = vb.get((num_classes,), "bias")?;
        let classifier = candle_nn::Linear::new(weight, Some(bias));

        Ok(Self { classifier })
    }
}

impl candle_nn::Module for FixedModernBertClassifier {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Apply linear classifier to get logits
        let logits = xs.apply(&self.classifier)?;
        // Apply softmax to get probabilities (match old architecture)
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
    }
}

impl FixedModernBertTokenClassifier {
    pub fn load(vb: candle_nn::VarBuilder, config: &Config) -> Result<Self, candle_core::Error> {
        // Following old architecture pattern - get num_classes from classifier_config
        let num_classes = config
            .classifier_config
            .as_ref()
            .map(|cc| cc.id2label.len())
            .unwrap_or(2);

        Self::load_with_classes(vb, config, num_classes)
    }

    pub fn load_with_classes(
        vb: candle_nn::VarBuilder,
        config: &Config,
        num_classes: usize,
    ) -> Result<Self, candle_core::Error> {
        // Following old architecture pattern - manually load weight and bias
        let classifier = candle_nn::Linear::new(
            vb.get((num_classes, config.hidden_size), "classifier.weight")?,
            Some(vb.get((num_classes,), "classifier.bias")?),
        );

        Ok(Self { classifier })
    }
}

impl candle_nn::Module for FixedModernBertTokenClassifier {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // For token classification, return logits for each token
        xs.apply(&self.classifier)
    }
}

// Manual Debug implementations (external types don't implement Debug)
impl std::fmt::Debug for TraditionalModernBertClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraditionalModernBertClassifier")
            .field("classifier_pooling", &self.classifier_pooling)
            .field("device", &self.device)
            .field("num_classes", &self.num_classes)
            .finish()
    }
}

impl std::fmt::Debug for TraditionalModernBertTokenClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraditionalModernBertTokenClassifier")
            .field("device", &self.device)
            .field("num_classes", &self.num_classes)
            .finish()
    }
}

impl TraditionalModernBertClassifier {
    /// Load ModernBERT number of classes using unified config loader
    fn load_modernbert_num_classes(model_path: &str) -> Result<usize, candle_core::Error> {
        use crate::core::config_loader;

        match config_loader::load_modernbert_num_classes(model_path) {
            Ok(result) => Ok(result),
            Err(unified_err) => Err(candle_core::Error::from(unified_err)),
        }
    }

    pub fn load_from_directory(
        model_path: &str,
        use_cpu: bool,
    ) -> Result<Self, candle_core::Error> {
        // 1. Determine device
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };
        // 2. Load config.json
        let config_path = format!("{}/config.json", model_path);
        let config_str = std::fs::read_to_string(&config_path).map_err(|_e| {
            let unified_err = config_errors::file_not_found(&config_path);
            candle_core::Error::from(unified_err)
        })?;

        let config: Config = serde_json::from_str(&config_str).map_err(|e| {
            let unified_err = config_errors::invalid_json(&config_path, &e.to_string());
            candle_core::Error::from(unified_err)
        })?;

        // 3. Dynamic class detection from id2label using unified config loader
        let num_classes = Self::load_modernbert_num_classes(model_path)?;

        // 4. Load tokenizer.json
        let tokenizer_path = format!("{}/tokenizer.json", model_path);
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            let unified_err = model_error!(
                ModelErrorType::Tokenizer,
                "tokenizer loading",
                format!("Failed to load tokenizer from {}: {}", tokenizer_path, e),
                &tokenizer_path
            );
            candle_core::Error::from(unified_err)
        })?;

        // Configure padding for batch processing
        if let Some(pad_token) = tokenizer.get_padding() {
            let mut padding_params = pad_token.clone();
            padding_params.strategy = tokenizers::PaddingStrategy::BatchLongest;
            tokenizer.with_padding(Some(padding_params));
        }
        // 5. Load model weights (model.safetensors)
        let weights_path = format!("{}/model.safetensors", model_path);
        if !std::path::Path::new(&weights_path).exists() {
            let unified_err = config_errors::file_not_found(&weights_path);
            return Err(candle_core::Error::from(unified_err));
        }

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DType::F32, &device)
                .map_err(|e| {
                    let unified_err = model_error!(
                        ModelErrorType::ModernBERT,
                        "weights loading",
                        format!("Failed to load weights from {}: {}", weights_path, e),
                        &weights_path
                    );
                    candle_core::Error::from(unified_err)
                })?
        };

        // 6. Create ModernBERT model - try both with and without prefix
        // Use the same logic as old architecture: try standard first, then _orig_mod
        let (model, model_vb) = if let Ok(model) = ModernBert::load(vb.clone(), &config) {
            // Standard loading succeeded, use vb.clone() for head and classifier
            (model, vb.clone())
        } else if let Ok(model) = ModernBert::load(vb.pp("_orig_mod"), &config) {
            // _orig_mod loading succeeded, use vb.pp("_orig_mod") for head and classifier
            (model, vb.pp("_orig_mod"))
        } else {
            let unified_err = model_error!(
                ModelErrorType::ModernBERT,
                "model loading",
                "Failed to load ModernBERT model with or without _orig_mod prefix",
                model_path
            );
            return Err(candle_core::Error::from(unified_err));
        };
        // 7. Load optional head layer
        let head = FixedModernBertHead::load(model_vb.pp("head"), &config).ok();

        // 8. Load classifier with dynamic class count
        let classifier = FixedModernBertClassifier::load_with_classes(
            model_vb.pp("classifier"),
            &config,
            num_classes,
        )
        .map_err(|e| {
            let unified_err = model_error!(
                ModelErrorType::Classifier,
                "classifier loading",
                format!("Failed to load classifier: {}", e),
                model_path
            );
            candle_core::Error::from(unified_err)
        })?;

        // 9. Create unified tokenizer wrapper with ModernBERT-specific config
        let tokenizer_config = crate::core::tokenization::TokenizationConfig {
            max_length: 512,
            add_special_tokens: true,
            truncation_strategy: tokenizers::TruncationStrategy::LongestFirst,
            truncation_direction: tokenizers::TruncationDirection::Right,
            pad_token_id: config.pad_token_id,
            pad_token: "[PAD]".to_string(),
            model_type: crate::core::tokenization::ModelType::ModernBERT,
            token_data_type: crate::core::tokenization::TokenDataType::U32,
        };

        let tokenizer_wrapper = Box::new(
            crate::core::tokenization::UnifiedTokenizer::new(
                tokenizer,
                tokenizer_config,
                device.clone(),
            )
            .map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::Tokenizer,
                    "tokenizer wrapper creation",
                    format!("Failed to create tokenizer wrapper: {}", e),
                    model_path
                );
                candle_core::Error::from(unified_err)
            })?,
        ) as Box<dyn DualPathTokenizer>;

        Ok(Self {
            model,
            head,
            classifier,
            classifier_pooling: ClassifierPooling::MEAN, // Use MEAN pooling as per model config
            tokenizer: tokenizer_wrapper,
            device,
            config,
            num_classes,
        })
    }

    /// Classify text using real model inference - REAL IMPLEMENTATION
    pub fn classify_text(&self, text: &str) -> Result<(usize, f32), candle_core::Error> {
        // 1. Tokenize input text
        let tokenization_result = self.tokenizer.tokenize(text).map_err(|e| {
            let unified_err = processing_errors::tensor_operation("tokenization", &e.to_string());
            candle_core::Error::from(unified_err)
        })?;

        // 2. Create input tensors
        let (input_ids, attention_mask) = self
            .tokenizer
            .create_tensors(&tokenization_result)
            .map_err(|e| {
                let unified_err =
                    processing_errors::tensor_operation("tensor creation", &e.to_string());
                candle_core::Error::from(unified_err)
            })?;

        // 3. Forward pass through ModernBERT model
        let model_output = self.model.forward(&input_ids, &attention_mask)?;

        // 4. Apply pooling strategy
        let pooled_output = match self.classifier_pooling {
            ClassifierPooling::CLS => {
                // Use [CLS] token (first token)
                model_output.i((.., 0, ..))?
            }
            ClassifierPooling::MEAN => {
                // Mean pooling over sequence length
                // Ensure attention_mask has the same number of dimensions as model_output
                let model_dims = model_output.dims().len();
                let mut mask_expanded = attention_mask.clone();

                // Add dimensions to match model_output
                while mask_expanded.dims().len() < model_dims {
                    mask_expanded = mask_expanded.unsqueeze(mask_expanded.dims().len())?;
                }

                let mask_expanded = mask_expanded.to_dtype(candle_core::DType::F32)?;
                let masked_output = model_output.broadcast_mul(&mask_expanded)?;
                let sum_output = masked_output.sum(1)?;
                let mask_sum = attention_mask
                    .sum_keepdim(1)?
                    .to_dtype(candle_core::DType::F32)?;
                sum_output.broadcast_div(&mask_sum)?
            }
        };

        // 5. Apply head layer if present
        let classifier_input = if let Some(ref head) = self.head {
            let head_output = head.forward(&pooled_output)?;
            head_output
        } else {
            pooled_output
        };

        // 6. Apply classifier to get probabilities (classifier applies softmax internally)
        let probabilities = self.classifier.forward(&classifier_input)?;

        // 8. Extract prediction (highest probability class)
        let probabilities_vec = probabilities.squeeze(0)?.to_vec1::<f32>()?;

        let mut max_prob = 0.0f32;
        let mut predicted_class = 0usize;

        for (i, &prob) in probabilities_vec.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                predicted_class = i;
            }
        }

        // 9. Get class label if available
        if let Some(class_labels) = self.get_class_labels() {
            if let Some(_label) = class_labels.get(&predicted_class.to_string()) {
                // Label available but not used in current implementation
            }
        }

        Ok((predicted_class, max_prob))
    }

    /// Get class labels mapping
    pub fn get_class_labels(&self) -> Option<&HashMap<String, String>> {
        self.config
            .classifier_config
            .as_ref()
            .map(|cc| &cc.id2label)
    }

    /// Get number of classes
    pub fn get_num_classes(&self) -> usize {
        self.num_classes
    }
}

impl TraditionalModernBertTokenClassifier {
    /// Create a new traditional ModernBERT token classifier
    pub fn new(model_id: &str, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Load model configuration
        let config_path = std::path::Path::new(model_id).join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| E::msg(format!("Failed to read config.json: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| E::msg(format!("Failed to parse config.json: {}", e)))?;

        // Load tokenizer
        let tokenizer_path = std::path::Path::new(model_id).join("tokenizer.json");
        let base_tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| E::msg(format!("Failed to load tokenizer: {}", e)))?;

        // Create dual-path compatible tokenizer
        let tokenizer = crate::core::tokenization::create_modernbert_compatibility_tokenizer(
            base_tokenizer,
            device.clone(),
        )?;

        // Load model weights
        let weights_path = std::path::Path::new(model_id).join("model.safetensors");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };

        // Load ModernBERT model (following old architecture pattern)
        let model = ModernBert::load(vb.clone(), &config)?;

        // Load head (optional) - following old architecture pattern
        let head = match vb.get(
            (config.hidden_size, config.hidden_size),
            "head.dense.weight",
        ) {
            Ok(_) => {
                let head_vb = vb.pp("head");
                Some(FixedModernBertHead::load(head_vb, &config)?)
            }
            Err(_) => {
                println!("  Head not found in model, using None (this is normal for some ModernBERT models)");
                None
            }
        };

        // Get number of classes from config.json id2label field (single source of truth)
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let num_classes = config_json.get("id2label")
            .and_then(|v| v.as_object())
            .map(|obj| obj.len())
            .ok_or_else(|| E::msg("config.json missing valid id2label field - this is required for ModernBERT token classification"))?;

        // Load token classifier with correct number of classes
        let classifier =
            FixedModernBertTokenClassifier::load_with_classes(vb.clone(), &config, num_classes)?;

        Ok(Self {
            model,
            head,
            classifier,
            tokenizer,
            device,
            config,
            num_classes,
            model_path: model_id.to_string(),
        })
    }

    /// Classify tokens in text
    pub fn classify_tokens(&self, text: &str) -> Result<Vec<(String, usize, f32, usize, usize)>> {
        // Tokenize the text
        let tokenization_result = self.tokenizer.tokenize(text)?;

        // Create tensors from tokenization result
        let (input_ids, attention_mask) = self.tokenizer.create_tensors(&tokenization_result)?;

        // Forward pass through ModernBERT (ModernBert::forward takes &Tensor, &Tensor)
        let sequence_output = self.model.forward(&input_ids, &attention_mask)?;

        // Apply head if available
        let hidden_states = if let Some(ref head) = self.head {
            head.forward(&sequence_output)?
        } else {
            sequence_output
        };

        // Apply token classifier
        let logits = self.classifier.forward(&hidden_states)?;

        // Apply softmax to get probabilities
        let probabilities = ops::softmax(&logits, D::Minus1)?;

        // Extract entities from BIO tags (following old architecture pattern)
        let mut results = Vec::new();
        let probs_data = probabilities.squeeze(0)?.to_vec2::<f32>()?;

        // Get predictions for each token
        let logits_squeezed = logits.squeeze(0)?;
        let predictions = logits_squeezed.argmax(D::Minus1)?;
        let predictions_vec = predictions.to_vec1::<u32>()?;

        // Load id2label mapping for BIO tag processing
        let config_path = format!(
            "{}/config.json",
            self.model_path
                .trim_end_matches("/model.safetensors")
                .trim_end_matches("/pytorch_model.bin")
        );
        let id2label = match crate::ffi::classify::load_id2label_from_config(&config_path) {
            Ok(mapping) => mapping,
            Err(_) => {
                // Fallback: return individual token results without BIO processing
                for (token_idx, token_probs) in probs_data.iter().enumerate() {
                    if token_idx < tokenization_result.tokens.len()
                        && token_idx < tokenization_result.offsets.len()
                    {
                        let (predicted_class, &confidence) = token_probs
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .unwrap();

                        let offset = tokenization_result.offsets[token_idx];
                        let token_text = if offset.0 < text.len()
                            && offset.1 <= text.len()
                            && offset.0 < offset.1
                        {
                            text[offset.0..offset.1].to_string()
                        } else {
                            tokenization_result.tokens[token_idx].clone()
                        };

                        results.push((token_text, predicted_class, confidence, offset.0, offset.1));
                    }
                }
                return Ok(results);
            }
        };

        // BIO tag entity extraction (like old architecture)
        #[derive(Debug, Clone)]
        struct TokenEntity {
            entity_type: String,
            start: usize,
            end: usize,
            text: String,
            confidence: f32,
        }

        let mut entities = Vec::new();
        let mut current_entity: Option<TokenEntity> = None;

        for (i, (&pred_id, offset)) in predictions_vec
            .iter()
            .zip(tokenization_result.offsets.iter())
            .enumerate()
        {
            // Skip special tokens (they have offset (0,0))
            if offset.0 == 0 && offset.1 == 0 && i > 0 {
                continue;
            }

            // Get label from prediction ID
            let label = id2label
                .get(&pred_id.to_string())
                .unwrap_or(&"O".to_string())
                .clone();
            let confidence = probs_data[i][pred_id as usize];

            if label.starts_with("B-") {
                // Beginning of new entity
                if let Some(entity) = current_entity.take() {
                    entities.push(entity);
                }

                let entity_type = label[2..].to_string(); // Remove 'B-' prefix
                current_entity = Some(TokenEntity {
                    entity_type,
                    start: offset.0,
                    end: offset.1,
                    text: text[offset.0..offset.1].to_string(),
                    confidence,
                });
            } else if let Some(entity_type) = label.strip_prefix("I-") {
                // Inside current entity
                if let Some(ref mut entity) = current_entity {
                    if entity.entity_type == entity_type {
                        // Extend current entity
                        entity.end = offset.1;
                        entity.text = text[entity.start..entity.end].to_string();
                        // Update confidence with average
                        entity.confidence = (entity.confidence + confidence) / 2.0;
                    } else {
                        // Different entity type, finish current and don't start new
                        entities.push(entity.clone());
                        current_entity = None;
                    }
                } // If no current entity, ignore I- tag
            } else {
                // Outside entity (O tag or different entity type)
                if let Some(entity) = current_entity.take() {
                    entities.push(entity);
                }
            }
        }

        // Add final entity if exists
        if let Some(entity) = current_entity.take() {
            entities.push(entity);
        }

        // Convert entities to results format
        for entity in entities {
            // Find the class index for this entity type
            let class_idx = id2label
                .iter()
                .find(|(_, v)| {
                    v.starts_with(&format!("B-{}", entity.entity_type))
                        || v.starts_with(&format!("I-{}", entity.entity_type))
                })
                .and_then(|(k, _)| k.parse::<usize>().ok())
                .unwrap_or(0);

            results.push((
                entity.text,
                class_idx,
                entity.confidence,
                entity.start,
                entity.end,
            ));
        }

        Ok(results)
    }

    /// Get class labels if available
    pub fn get_class_labels(&self) -> Option<&HashMap<String, String>> {
        None
    }
}
