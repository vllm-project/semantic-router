//! Traditional ModernBERT Implementation - Dual Path Architecture
//!
//! This module provides the traditional fine-tuning ModernBERT implementation
//! that preserves all bug fixes from FixedModernBertClassifier.
//!
//! Supports both standard ModernBERT and mmBERT (multilingual ModernBERT) variants:
//! - ModernBERT: Standard English-focused model
//! - mmBERT: Multilingual model (1800+ languages), 256k vocab, 8192 max length
//!
//! The variant is auto-detected from config.json or can be explicitly specified.

/// Existing callers keep a bounded default; longer inputs require explicit opt-in.
pub(crate) const DEFAULT_CLASSIFICATION_SEQ_LEN: usize = 512;

use crate::core::{
    config_errors, drain_loader_queue, processing_errors, resolve_device, run_on_inference_pool,
    ModelErrorType, UnifiedError,
};
use crate::model_error;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops, LayerNorm, Linear, Module, VarBuilder};
// Use local copy of ModernBERT with Flash Attention support
// Import from parent module's re-exports (more reliable across branches)
use super::{ClassifierConfig, ClassifierPooling, Config, ModernBert};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tokenizers::{PaddingParams, PaddingStrategy, PostProcessor, Tokenizer};

use crate::core::tokenization::DualPathTokenizer;
use crate::model_architectures::traits::*;
use crate::model_architectures::unified_interface::{
    ConfigurableModel, CoreModel, PathSpecialization,
};

/// ModernBERT model variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModernBertVariant {
    /// Standard ModernBERT (English-focused, 512 max length)
    Standard,
    /// mmBERT - Multilingual ModernBERT (1800+ languages, 256k vocab, 8192 max length)
    /// Reference: https://huggingface.co/jhu-clsp/mmBERT-base
    Multilingual,
    /// ModernBERT-base-32k - Extended context ModernBERT (32,768 max length with RoPE)
    /// Reference: https://huggingface.co/llm-semantic-router/modernbert-base-32k
    Extended32K,
    /// Historical mmBERT-32K variant name (32768 max length).
    /// Actual rotary scaling is declared only by config.json.
    /// Reference: https://huggingface.co/llm-semantic-router/mmbert-32k-yarn
    Multilingual32K,
}

impl ModernBertVariant {
    /// Historical variant hint; loaders use config.max_position_embeddings as
    /// the capacity and an explicit request (default 512) as the input budget.
    pub fn max_length(&self) -> usize {
        match self {
            ModernBertVariant::Standard => 512,
            ModernBertVariant::Multilingual => 8192,
            ModernBertVariant::Extended32K => 32768,
            ModernBertVariant::Multilingual32K => 32768,
        }
    }

    /// Get the tokenization strategy for this variant
    pub fn tokenization_strategy(&self) -> crate::core::tokenization::TokenizationStrategy {
        match self {
            ModernBertVariant::Standard => {
                crate::core::tokenization::TokenizationStrategy::ModernBERT
            }
            ModernBertVariant::Multilingual | ModernBertVariant::Multilingual32K => {
                crate::core::tokenization::TokenizationStrategy::MmBERT
            }
            ModernBertVariant::Extended32K => {
                // 32K variant uses ModernBERT tokenization strategy
                crate::core::tokenization::TokenizationStrategy::ModernBERT
            }
        }
    }

    /// Get the pad token string for this variant
    pub fn pad_token(&self) -> &'static str {
        match self {
            ModernBertVariant::Standard => "[PAD]",
            ModernBertVariant::Multilingual => "<pad>",
            ModernBertVariant::Extended32K => "[PAD]",
            ModernBertVariant::Multilingual32K => "<pad>",
        }
    }

    /// Whether this variant has a historical YaRN label. This metadata does not
    /// implement RoPE scaling or override the checkpoint's positional capacity.
    pub fn uses_yarn_scaling(&self) -> bool {
        matches!(
            self,
            ModernBertVariant::Multilingual32K | ModernBertVariant::Extended32K
        )
    }

    /// Get the expected RoPE theta for this variant
    pub fn expected_rope_theta(&self) -> f64 {
        match self {
            ModernBertVariant::Standard => 10000.0,
            ModernBertVariant::Multilingual => 10000.0,
            ModernBertVariant::Extended32K => 10000.0, // Historical default, not a scaling recipe
            ModernBertVariant::Multilingual32K => 160000.0, // Direct-theta legacy default
        }
    }

    /// Detect a display variant from the checkpoint configuration only.
    pub fn detect_from_config(config_path: &str) -> Result<Self, candle_core::Error> {
        let config_str = std::fs::read_to_string(config_path).map_err(|_e| {
            let unified_err = config_errors::file_not_found(config_path);
            candle_core::Error::from(unified_err)
        })?;

        let config_json: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            let unified_err = config_errors::invalid_json(config_path, &e.to_string());
            candle_core::Error::from(unified_err)
        })?;

        let vocab_size = config_json
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let position_embedding_type = config_json
            .get("position_embedding_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let max_position_embeddings = config_json
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(512);

        // mmBERT has vocab_size >= 200000 and uses sans_pos (RoPE)
        if vocab_size >= 200000 && position_embedding_type == "sans_pos" {
            // The declared capacity determines this historical display variant.
            if max_position_embeddings >= 32768 {
                Ok(ModernBertVariant::Multilingual32K)
            } else {
                Ok(ModernBertVariant::Multilingual)
            }
        }
        // ModernBERT-base-32k: max_position_embeddings >= 32768 and uses sans_pos (RoPE)
        // This should be checked after mmBERT to avoid false positives
        else if max_position_embeddings >= 32768 && position_embedding_type == "sans_pos" {
            Ok(ModernBertVariant::Extended32K)
        } else {
            Ok(ModernBertVariant::Standard)
        }
    }
}

/// Traditional ModernBERT sequence classifier
///
/// Supports both standard ModernBERT and mmBERT (multilingual) variants.
/// The variant is auto-detected from config.json or can be explicitly specified.
pub struct TraditionalModernBertClassifier {
    model: Arc<ModernBert>,
    head: Option<FixedModernBertHead>,
    classifier: FixedModernBertClassifier,
    classifier_pooling: ClassifierPooling,
    tokenizer: Box<dyn DualPathTokenizer>,
    device: Device,
    config: Config,
    num_classes: usize,
    variant: ModernBertVariant,
}

/// Traditional ModernBERT token classifier
///
/// Supports both standard ModernBERT and mmBERT (multilingual) variants.
pub struct TraditionalModernBertTokenClassifier {
    model: Arc<ModernBert>,
    head: Option<FixedModernBertHead>,
    classifier: FixedModernBertTokenClassifier,
    tokenizer: Box<dyn DualPathTokenizer>,
    device: Device,
    config: Config,
    num_classes: usize,
    labels: HashMap<String, String>,
    variant: ModernBertVariant,
}

// Type aliases for mmBERT (multilingual ModernBERT) for API clarity
/// mmBERT sequence classifier (alias for TraditionalModernBertClassifier with Multilingual variant)
pub type MmBertClassifier = TraditionalModernBertClassifier;
/// mmBERT token classifier (alias for TraditionalModernBertTokenClassifier with Multilingual variant)
pub type MmBertTokenClassifier = TraditionalModernBertTokenClassifier;
/// mmBERT-32K sequence classifier (alias for TraditionalModernBertClassifier with Multilingual32K variant)
pub type MmBert32KClassifier = TraditionalModernBertClassifier;
/// mmBERT-32K token classifier (alias for TraditionalModernBertTokenClassifier with Multilingual32K variant)
pub type MmBert32KTokenClassifier = TraditionalModernBertTokenClassifier;

// Global static instances using OnceLock pattern for zero-cost reads after initialization
pub static TRADITIONAL_MODERNBERT_CLASSIFIER: OnceLock<Arc<TraditionalModernBertClassifier>> =
    OnceLock::new();
pub static TRADITIONAL_MODERNBERT_PII_CLASSIFIER: OnceLock<Arc<TraditionalModernBertClassifier>> =
    OnceLock::new();
pub static TRADITIONAL_MODERNBERT_JAILBREAK_CLASSIFIER: OnceLock<
    Arc<TraditionalModernBertClassifier>,
> = OnceLock::new();
pub static TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER: OnceLock<
    Arc<TraditionalModernBertTokenClassifier>,
> = OnceLock::new();
// Fact-check classifier using halugate-sentinel model (ModernBERT-based sequence classifier)
// Model outputs: 0=NO_FACT_CHECK_NEEDED, 1=FACT_CHECK_NEEDED
pub static TRADITIONAL_MODERNBERT_FACT_CHECK_CLASSIFIER: OnceLock<
    Arc<TraditionalModernBertClassifier>,
> = OnceLock::new();

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
    fn load_for_artifact(
        vb: candle_nn::VarBuilder,
        config: &Config,
        artifact_config: &str,
    ) -> Result<Option<Self>, candle_core::Error> {
        let raw: serde_json::Value =
            serde_json::from_str(artifact_config).map_err(candle_core::Error::wrap)?;
        let requires_head = raw["architectures"]
            .as_array()
            .is_some_and(|architectures| {
                architectures.iter().any(|name| {
                    matches!(
                        name.as_str(),
                        Some(
                            "ModernBertForSequenceClassification"
                                | "ModernBertForTokenClassification"
                        )
                    )
                })
            });
        // Legacy linear-only artifacts may omit this block. A declared HF task
        // head or a partially present head must load every required tensor.
        if requires_head
            || ["dense.weight", "dense.bias", "norm.weight", "norm.bias"]
                .iter()
                .any(|name| vb.contains_tensor(name))
        {
            Self::load(vb, config).map(Some)
        } else {
            Ok(None)
        }
    }

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
            config.layer_norm_eps,
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
            .field("variant", &self.variant)
            .field("classifier_pooling", &self.classifier_pooling)
            .field("device", &self.device)
            .field("num_classes", &self.num_classes)
            .finish()
    }
}

impl std::fmt::Debug for TraditionalModernBertTokenClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraditionalModernBertTokenClassifier")
            .field("variant", &self.variant)
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

    /// Normalize config JSON to handle different HuggingFace model config formats
    /// Some models (e.g., mmbert-32k-yarn) use top-level global_rope_theta/local_rope_theta
    /// Other models (e.g., feedback-detector) use nested rope_parameters structure
    /// This function extracts rope_theta from rope_parameters if top-level fields are missing
    pub fn normalize_config_json(config_str: &str) -> String {
        let Ok(mut config_json) = serde_json::from_str::<serde_json::Value>(config_str) else {
            return config_str.to_string();
        };

        let obj = match config_json.as_object_mut() {
            Some(obj) => obj,
            None => return config_str.to_string(),
        };

        // If global_rope_theta is missing, try to extract from rope_parameters
        if !obj.contains_key("global_rope_theta") {
            if let Some(rope_params) = obj.get("rope_parameters") {
                // Try to get from full_attention or sliding_attention
                let theta = rope_params
                    .get("full_attention")
                    .and_then(|v| v.get("rope_theta"))
                    .and_then(|v| v.as_f64())
                    .or_else(|| {
                        rope_params
                            .get("sliding_attention")
                            .and_then(|v| v.get("rope_theta"))
                            .and_then(|v| v.as_f64())
                    })
                    .unwrap_or(160000.0); // Default for mmBERT-32K

                obj.insert(
                    "global_rope_theta".to_string(),
                    serde_json::Value::from(theta),
                );
            } else {
                // No rope_parameters either, use default
                obj.insert(
                    "global_rope_theta".to_string(),
                    serde_json::Value::from(160000.0),
                );
            }
        }

        // Local and global layers can have different RoPE parameters.
        if !obj.contains_key("local_rope_theta") {
            let local_theta = obj
                .get("rope_parameters")
                .and_then(|v| v.get("sliding_attention"))
                .and_then(|v| v.get("rope_theta"))
                .and_then(|v| v.as_f64())
                .or_else(|| obj.get("global_rope_theta").and_then(|v| v.as_f64()))
                .unwrap_or(160000.0);
            obj.insert(
                "local_rope_theta".to_string(),
                serde_json::Value::from(local_theta),
            );
        }

        serde_json::to_string(&config_json).unwrap_or_else(|_| config_str.to_string())
    }

    /// A variant or training metadata cannot enlarge the backbone's declared
    /// capacity. The same config sets the tokenizer limit and RoPE cache size.
    pub(super) fn parse_model_config(config_str: &str) -> Result<Config, candle_core::Error> {
        let raw: serde_json::Value =
            serde_json::from_str(config_str).map_err(candle_core::Error::wrap)?;
        if !raw.is_object() {
            candle_core::bail!("ModernBERT config must be an object");
        }
        for field in ["attention_bias", "mlp_bias", "norm_bias", "classifier_bias"] {
            if raw
                .get(field)
                .is_some_and(|value| value.as_bool() != Some(false))
            {
                candle_core::bail!("unsupported ModernBERT {field}; this adapter requires false");
            }
        }
        for field in ["hidden_activation", "classifier_activation"] {
            if raw
                .get(field)
                .is_some_and(|value| value.as_str() != Some("gelu"))
            {
                candle_core::bail!("unsupported ModernBERT {field}; this adapter requires gelu");
            }
        }
        let global_default = raw
            .get("global_rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(160000.0);
        let thetas = crate::model_architectures::modernbert_rope::resolve_thetas(
            &raw,
            [160000.0, global_default],
        )?;
        let mut normalized = raw.clone();
        if let Some(eps) = crate::model_architectures::modernbert_config::norm_eps(&raw)? {
            normalized.as_object_mut().unwrap().remove("norm_eps");
            normalized["layer_norm_eps"] = serde_json::Value::from(eps);
        }
        normalized["global_rope_theta"] = serde_json::Value::from(thetas[0]);
        normalized["local_rope_theta"] = serde_json::Value::from(thetas[1]);
        let config: Config =
            serde_json::from_value(normalized).map_err(candle_core::Error::wrap)?;
        config.resolve_rope()?;
        if config.max_position_embeddings == 0
            || u32::try_from(config.max_position_embeddings).is_err()
        {
            candle_core::bail!("ModernBERT max_position_embeddings must be a positive u32");
        }
        if config.pad_token_id as usize >= config.vocab_size {
            candle_core::bail!("ModernBERT pad_token_id is outside the vocabulary");
        }
        for theta in [config.global_rope_theta, config.local_rope_theta] {
            if !theta.is_finite() || theta <= 0.0 {
                candle_core::bail!("ModernBERT RoPE theta must be finite and positive");
            }
        }
        Ok(config)
    }

    pub(super) fn resolve_sequence_length(
        config: &Config,
        requested: Option<usize>,
    ) -> Result<usize, candle_core::Error> {
        let max_length =
            requested.unwrap_or(DEFAULT_CLASSIFICATION_SEQ_LEN.min(config.max_position_embeddings));
        if max_length == 0 || max_length > config.max_position_embeddings {
            candle_core::bail!(
                "max_sequence_length must be between 1 and {}, got {max_length}",
                config.max_position_embeddings
            );
        }
        Ok(max_length)
    }

    pub(super) fn parse_classifier_pooling(
        config_json: &str,
    ) -> Result<ClassifierPooling, candle_core::Error> {
        let config: serde_json::Value =
            serde_json::from_str(config_json).map_err(candle_core::Error::wrap)?;
        match config.get("classifier_pooling") {
            // Preserve the historical default for merged artifacts that omit it.
            None => Ok(ClassifierPooling::MEAN),
            Some(value) if value.as_str() == Some("mean") => Ok(ClassifierPooling::MEAN),
            Some(value) if value.as_str() == Some("cls") => Ok(ClassifierPooling::CLS),
            Some(_) => candle_core::bail!("classifier_pooling must be mean or cls"),
        }
    }

    pub(super) fn tokenizer_for_config(
        mut tokenizer: Tokenizer,
        config: &Config,
        variant: ModernBertVariant,
        device: Device,
        max_sequence_length: usize,
    ) -> Result<Box<dyn DualPathTokenizer>> {
        let max_sequence_length = Self::resolve_sequence_length(config, Some(max_sequence_length))?;
        let special_tokens = tokenizer
            .get_post_processor()
            .map_or(0, |processor| processor.added_tokens(false));
        // UnifiedTokenizer configures truncation on its next encode. Validate
        // now: tokenizers subtracts the special-token count with usize arithmetic.
        if max_sequence_length < special_tokens {
            anyhow::bail!(
                "max_sequence_length {max_sequence_length} is smaller than the tokenizer's {special_tokens} special tokens"
            );
        }
        let pad_token = tokenizer
            .id_to_token(config.pad_token_id)
            .unwrap_or_else(|| variant.pad_token().to_string());
        // Exported fixed padding can exceed the backbone capacity even after
        // truncation. UnifiedTokenizer supplies batch padding; singles need none.
        tokenizer.with_padding(None);
        let tokenizer_config = crate::core::tokenization::TokenizationConfig {
            max_length: max_sequence_length,
            add_special_tokens: true,
            truncation_strategy: tokenizers::TruncationStrategy::LongestFirst,
            truncation_direction: tokenizers::TruncationDirection::Right,
            pad_token_id: config.pad_token_id,
            pad_token,
            tokenization_strategy: variant.tokenization_strategy(),
            token_data_type: crate::core::tokenization::TokenDataType::U32,
        };
        Ok(Box::new(crate::core::tokenization::UnifiedTokenizer::new(
            tokenizer,
            tokenizer_config,
            device,
        )?))
    }

    /// Load from directory with auto-detected variant (Standard or Multilingual/mmBERT)
    pub fn load_from_directory(
        model_path: &str,
        use_cpu: bool,
    ) -> Result<Self, candle_core::Error> {
        // Auto-detect variant from config.json
        let config_path = format!("{}/config.json", model_path);
        let variant = ModernBertVariant::detect_from_config(&config_path)?;
        Self::load_from_directory_with_variant(model_path, use_cpu, variant)
    }

    /// Load from directory with explicit variant specification
    pub fn load_from_directory_with_variant(
        model_path: &str,
        use_cpu: bool,
        variant: ModernBertVariant,
    ) -> Result<Self, candle_core::Error> {
        Self::load_from_directory_with_limit(model_path, use_cpu, variant, None)
    }

    /// Load a classifier with an explicit input budget bounded by the model config.
    pub fn load_from_directory_with_max_sequence_length(
        model_path: &str,
        use_cpu: bool,
        max_sequence_length: usize,
    ) -> Result<Self, candle_core::Error> {
        let variant = ModernBertVariant::detect_from_config(&format!("{model_path}/config.json"))?;
        Self::load_from_directory_with_variant_and_max_sequence_length(
            model_path,
            use_cpu,
            variant,
            max_sequence_length,
        )
    }

    pub fn load_from_directory_with_variant_and_max_sequence_length(
        model_path: &str,
        use_cpu: bool,
        variant: ModernBertVariant,
        max_sequence_length: usize,
    ) -> Result<Self, candle_core::Error> {
        Self::load_from_directory_with_limit(
            model_path,
            use_cpu,
            variant,
            Some(max_sequence_length),
        )
    }

    fn load_from_directory_with_limit(
        model_path: &str,
        use_cpu: bool,
        variant: ModernBertVariant,
        max_sequence_length: Option<usize>,
    ) -> Result<Self, candle_core::Error> {
        // 1. Determine device
        let device = resolve_device(use_cpu);
        // 2. Load config.json
        let config_path = format!("{}/config.json", model_path);
        let config_str = std::fs::read_to_string(&config_path).map_err(|_e| {
            let unified_err = config_errors::file_not_found(&config_path);
            candle_core::Error::from(unified_err)
        })?;

        let config = Self::parse_model_config(&config_str).map_err(|e| {
            let unified_err = config_errors::invalid_json(&config_path, &e.to_string());
            candle_core::Error::from(unified_err)
        })?;
        let classifier_pooling = Self::parse_classifier_pooling(&config_str)?;

        let max_sequence_length = Self::resolve_sequence_length(&config, max_sequence_length)?;

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
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights_path),
                DType::F32,
                &device,
            )
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
        let head =
            FixedModernBertHead::load_for_artifact(model_vb.pp("head"), &config, &config_str)?;

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

        let tokenizer_wrapper = Self::tokenizer_for_config(
            tokenizer,
            &config,
            variant,
            device.clone(),
            max_sequence_length,
        )
        .map_err(candle_core::Error::wrap)?;

        drain_loader_queue(&device);
        Ok(Self {
            model: Arc::new(model),
            head,
            classifier,
            classifier_pooling,
            tokenizer: tokenizer_wrapper,
            device,
            config,
            num_classes,
            variant,
        })
    }

    /// Load classifier with custom base model from separate paths
    ///
    /// This method allows loading a base model from one path and classifier weights from another path.
    /// This is useful when you have a base model (e.g., Extended32K) and want to use classifier weights
    /// from a different model (e.g., Standard ModernBERT classifier).
    ///
    /// # Arguments
    /// * `base_model_path` - Path to the base model directory (contains config.json, tokenizer.json, model.safetensors)
    /// * `classifier_path` - Path to the classifier model directory (contains config.json with id2label, model.safetensors with classifier weights)
    /// * `variant` - The ModernBERT variant to use (should match the base model)
    /// * `use_cpu` - Whether to use CPU instead of GPU
    ///
    /// # Returns
    /// * `Result<Self>` - The loaded classifier with custom base model
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use candle_semantic_router::model_architectures::traditional::modernbert::{
    ///     ModernBertVariant, TraditionalModernBertClassifier
    /// };
    ///
    /// // Load Extended32K base model with PII classifier weights
    /// let classifier = TraditionalModernBertClassifier::load_with_custom_base_model(
    ///     "/path/to/modernbert-base-32k",           // Base model path
    ///     "/path/to/pii_classifier_modernbert-base", // Classifier weights path
    ///     ModernBertVariant::Extended32K,
    ///     true, // use_cpu
    /// )?;
    ///
    /// // This compatibility loader keeps the default 512-token input budget.
    /// // Use load_with_custom_base_model_and_max_sequence_length to opt in to
    /// // a larger budget supported by the base model's config.json.
    /// let (class_id, confidence) = classifier.classify_text("My email is john@example.com")?;
    /// ```
    pub fn load_with_custom_base_model(
        base_model_path: &str,
        classifier_path: &str,
        variant: ModernBertVariant,
        use_cpu: bool,
    ) -> Result<Self, candle_core::Error> {
        Self::load_with_custom_base_model_and_limit(
            base_model_path,
            classifier_path,
            variant,
            use_cpu,
            None,
        )
    }

    pub fn load_with_custom_base_model_and_max_sequence_length(
        base_model_path: &str,
        classifier_path: &str,
        variant: ModernBertVariant,
        use_cpu: bool,
        max_sequence_length: usize,
    ) -> Result<Self, candle_core::Error> {
        Self::load_with_custom_base_model_and_limit(
            base_model_path,
            classifier_path,
            variant,
            use_cpu,
            Some(max_sequence_length),
        )
    }

    fn load_with_custom_base_model_and_limit(
        base_model_path: &str,
        classifier_path: &str,
        variant: ModernBertVariant,
        use_cpu: bool,
        max_sequence_length: Option<usize>,
    ) -> Result<Self, candle_core::Error> {
        // 1. Determine device
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };

        // 2. Load base model config.json
        let base_config_path = format!("{}/config.json", base_model_path);
        let base_config_str = std::fs::read_to_string(&base_config_path).map_err(|_e| {
            let unified_err = config_errors::file_not_found(&base_config_path);
            candle_core::Error::from(unified_err)
        })?;

        let config = Self::parse_model_config(&base_config_str).map_err(|e| {
            let unified_err = config_errors::invalid_json(&base_config_path, &e.to_string());
            candle_core::Error::from(unified_err)
        })?;

        let max_sequence_length = Self::resolve_sequence_length(&config, max_sequence_length)?;

        // 3. Load number of classes from classifier config.json
        let num_classes = Self::load_modernbert_num_classes(classifier_path)?;
        let classifier_config_path = format!("{}/config.json", classifier_path);
        let classifier_config_str =
            std::fs::read_to_string(&classifier_config_path).map_err(candle_core::Error::wrap)?;
        let classifier_pooling = Self::parse_classifier_pooling(&classifier_config_str)?;

        // 4. Load tokenizer from base model
        let tokenizer_path = format!("{}/tokenizer.json", base_model_path);
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

        // 5. Load base model weights
        let base_weights_path = format!("{}/model.safetensors", base_model_path);
        if !std::path::Path::new(&base_weights_path).exists() {
            let unified_err = config_errors::file_not_found(&base_weights_path);
            return Err(candle_core::Error::from(unified_err));
        }

        let base_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&base_weights_path),
                DType::F32,
                &device,
            )
            .map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::ModernBERT,
                    "base model weights loading",
                    format!(
                        "Failed to load base model weights from {}: {}",
                        base_weights_path, e
                    ),
                    &base_weights_path
                );
                candle_core::Error::from(unified_err)
            })?
        };

        // 6. Load base ModernBERT model - try both with and without prefix
        let model = if let Ok(model) = ModernBert::load(base_vb.clone(), &config) {
            model
        } else if let Ok(model) = ModernBert::load(base_vb.pp("_orig_mod"), &config) {
            model
        } else {
            let unified_err = model_error!(
                ModelErrorType::ModernBERT,
                "base model loading",
                "Failed to load base ModernBERT model with or without _orig_mod prefix",
                base_model_path
            );
            return Err(candle_core::Error::from(unified_err));
        };

        // 7. Load classifier weights from classifier path
        let classifier_weights_path = format!("{}/model.safetensors", classifier_path);
        if !std::path::Path::new(&classifier_weights_path).exists() {
            let unified_err = config_errors::file_not_found(&classifier_weights_path);
            return Err(candle_core::Error::from(unified_err));
        }

        let classifier_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&classifier_weights_path),
                DType::F32,
                &device,
            )
            .map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::Classifier,
                    "classifier weights loading",
                    format!(
                        "Failed to load classifier weights from {}: {}",
                        classifier_weights_path, e
                    ),
                    &classifier_weights_path
                );
                candle_core::Error::from(unified_err)
            })?
        };

        // Try to load head from classifier (if exists)
        let head = FixedModernBertHead::load_for_artifact(
            classifier_vb.pp("head"),
            &config,
            &classifier_config_str,
        )?;

        // 8. Load classifier weights from classifier path
        let classifier = FixedModernBertClassifier::load_with_classes(
            classifier_vb.pp("classifier"),
            &config,
            num_classes,
        )
        .map_err(|e| {
            let unified_err = model_error!(
                ModelErrorType::Classifier,
                "classifier loading",
                format!("Failed to load classifier from {}: {}", classifier_path, e),
                classifier_path
            );
            candle_core::Error::from(unified_err)
        })?;

        let tokenizer_wrapper = Self::tokenizer_for_config(
            tokenizer,
            &config,
            variant,
            device.clone(),
            max_sequence_length,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok(Self {
            model: Arc::new(model),
            head,
            classifier,
            classifier_pooling,
            tokenizer: tokenizer_wrapper,
            device,
            config,
            num_classes,
            variant,
        })
    }

    /// Load mmBERT (multilingual) model from directory
    /// Convenience method that explicitly loads as Multilingual variant
    pub fn load_mmbert_from_directory(
        model_path: &str,
        use_cpu: bool,
    ) -> Result<Self, candle_core::Error> {
        Self::load_from_directory_with_variant(model_path, use_cpu, ModernBertVariant::Multilingual)
    }

    /// Load an mmBERT-32K checkpoint with the default 512-token input budget.
    /// Convenience method that explicitly loads as Multilingual32K variant.
    /// Use load_from_directory_with_max_sequence_length to request a larger
    /// budget. Capacity and RoPE parameters come from the checkpoint config;
    /// the variant name does not enable a scaling algorithm.
    /// Reference: https://huggingface.co/llm-semantic-router/mmbert-32k-yarn
    pub fn load_mmbert_32k_from_directory(
        model_path: &str,
        use_cpu: bool,
    ) -> Result<Self, candle_core::Error> {
        Self::load_from_directory_with_variant(
            model_path,
            use_cpu,
            ModernBertVariant::Multilingual32K,
        )
    }

    /// Get the model variant (Standard, Multilingual, or Multilingual32K)
    pub fn variant(&self) -> ModernBertVariant {
        self.variant
    }

    /// Check if this is a multilingual (mmBERT) model (8K or 32K)
    pub fn is_multilingual(&self) -> bool {
        matches!(
            self.variant,
            ModernBertVariant::Multilingual | ModernBertVariant::Multilingual32K
        )
    }

    /// Historical variant predicate; this does not identify the active RoPE math.
    pub fn is_32k_yarn(&self) -> bool {
        self.variant == ModernBertVariant::Multilingual32K
    }

    /// classify_internal classifies text using real model inference - REAL IMPLEMENTATION
    fn classify_internal(&self, text: &str) -> Result<(usize, f32, Vec<f32>), candle_core::Error> {
        self.classify_text_with_activation(text, false)
    }

    /// Full categorical (softmax) or independent multi-label (sigmoid) scores.
    /// The caller must validate this choice against the artifact's task contract.
    pub fn classify_text_with_activation(
        &self,
        text: &str,
        multi_label: bool,
    ) -> Result<(usize, f32, Vec<f32>), candle_core::Error> {
        run_on_inference_pool(&self.device, || self.classify_on_device(text, multi_label))
    }

    fn classify_on_device(
        &self,
        text: &str,
        multi_label: bool,
    ) -> Result<(usize, f32, Vec<f32>), candle_core::Error> {
        // 1. Tokenize input text
        let tokenization_result = self.tokenizer.tokenize(text).map_err(|e| {
            let unified_err = processing_errors::tensor_operation("tokenization", &e.to_string());
            candle_core::Error::from(unified_err)
        })?;

        if tokenization_result.truncated {
            candle_core::bail!(
                "input exceeds the classifier's {} token budget",
                self.tokenizer.get_config().max_length
            );
        }
        // 2. Create input tensors
        let (input_ids, attention_mask) = self
            .tokenizer
            .create_tensors(&tokenization_result)
            .map_err(|e| {
                let unified_err =
                    processing_errors::tensor_operation("tensor creation", &e.to_string());
                candle_core::Error::from(unified_err)
            })?;

        self.classify_tensors_with_activation(&input_ids, &attention_mask, multi_label)
    }

    /// Classify an unpadded token window, with its special tokens already restored.
    /// Positions start at zero for every call. No tokenizer can truncate or change it.
    pub fn classify_tokens_with_activation(
        &self,
        ids: &[u32],
        multi_label: bool,
    ) -> Result<(usize, f32, Vec<f32>), candle_core::Error> {
        if ids.is_empty() || ids.len() > self.tokenizer.get_config().max_length {
            candle_core::bail!("token window is empty or exceeds the classifier budget");
        }
        run_on_inference_pool(&self.device, || {
            let input_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
            let attention_mask = Tensor::ones((1, ids.len()), DType::U32, &self.device)?;
            self.classify_tensors_with_activation(&input_ids, &attention_mask, multi_label)
        })
    }

    fn classify_tensors_with_activation(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        multi_label: bool,
    ) -> Result<(usize, f32, Vec<f32>), candle_core::Error> {
        // 3. Forward pass through ModernBERT model
        let model_output = self.model.forward(input_ids, attention_mask)?;

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
            head.forward(&pooled_output)?
        } else {
            pooled_output
        };

        let logits = classifier_input
            .apply(&self.classifier.classifier)?
            .to_dtype(DType::F32)?;
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        if flat.len() != self.num_classes || flat.iter().any(|value| !value.is_finite()) {
            candle_core::bail!("classifier returned invalid or non-finite logits");
        }
        let probabilities = if multi_label {
            candle_nn::ops::sigmoid(&logits)?
        } else {
            candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?
        };
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

        Ok((predicted_class, max_prob, probabilities_vec))
    }

    /// Classify text and return the top-1 prediction (class index + confidence).
    pub fn classify_text(&self, text: &str) -> Result<(usize, f32), candle_core::Error> {
        let (predicted_class, max_prob, _) = self.classify_internal(text)?;
        Ok((predicted_class, max_prob))
    }

    /// Classify text and return the top-1 prediction together with the full
    /// softmax probability distribution across all classes.
    pub fn classify_text_with_probabilities(
        &self,
        text: &str,
    ) -> Result<(usize, f32, Vec<f32>), candle_core::Error> {
        self.classify_internal(text)
    }

    pub fn fit_prefix_to_window(&self, prefix: &str, suffix: &str) -> Result<String> {
        crate::core::tokenization_window::fit_prefix_to_window(
            self.tokenizer.as_ref(),
            prefix,
            suffix,
        )
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

/// Token entity tuple returned by `classify_tokens`: (token, label_id, confidence, start, end)
type TokenEntityTuple = (String, usize, f32, usize, usize);

/// One token's resolved BIO classification, consumed by [`merge_bio_entities`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct BioToken<'a> {
    /// Full BIO label, for example `B-PERSON`, `I-PERSON` or `O`.
    pub label: &'a str,
    /// Byte offset of the token's start in the original text.
    pub start: usize,
    /// Byte offset of the token's end in the original text.
    pub end: usize,
    /// Probability the model assigned to `label`.
    pub confidence: f32,
}

/// A run of BIO tokens merged into a single entity.
#[derive(Debug, Clone)]
pub(crate) struct BioEntity {
    pub entity_type: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
    /// Arithmetic mean of the per-token confidences merged into this entity.
    pub confidence: f32,
    /// Number of tokens merged into this entity.
    pub token_count: u32,
}

impl BioEntity {
    /// Fold one more token's confidence into the running arithmetic mean.
    ///
    /// This deliberately is not `(self.confidence + confidence) / 2.0`. That
    /// form is a running pairwise fold, which gives the last token weight 1/2
    /// and the leading `B-` token weight 1/2^(n-1), so an entity's confidence
    /// ends up depending on how many tokens it happened to split into.
    fn accumulate(&mut self, confidence: f32) {
        self.token_count += 1;
        self.confidence += (confidence - self.confidence) / self.token_count as f32;
    }
}

/// Trim surrounding whitespace from an entity span.
///
/// Sub-word tokenizers attach the preceding space to a word-initial token, so
/// a `PERSON` span starts on the space before the name. Callers slice the text
/// with these offsets, and PII masking then eats that space.
fn trim_entity_span(text: &str, start: usize, end: usize) -> (usize, usize) {
    let slice = &text[start..end];
    let trimmed = slice.trim_matches(|c: char| c.is_whitespace());
    if trimmed.is_empty() {
        return (start, end);
    }
    let lead = slice.len() - slice.trim_start_matches(|c: char| c.is_whitespace()).len();
    let trail = slice.len() - slice.trim_end_matches(|c: char| c.is_whitespace()).len();
    (start + lead, end - trail)
}

/// Merge a sequence of BIO-labelled tokens into entities.
///
/// `tokens` must already have special tokens removed and must be in text order.
/// Offsets are byte offsets into `text`.
///
/// An `I-` tag that does not continue an open entity opens one of its own type.
/// Models are not obliged to emit `B-`: the mmBERT-32K PII detector labels
/// emails and domains with `I-` only, so requiring `B-` dropped them entirely.
pub(crate) fn merge_bio_entities(text: &str, tokens: &[BioToken<'_>]) -> Vec<BioEntity> {
    let mut entities: Vec<BioEntity> = Vec::new();
    let mut current: Option<BioEntity> = None;

    // Close `current`, trim its span and push it.
    fn flush(text: &str, entities: &mut Vec<BioEntity>, entity: BioEntity) {
        let mut entity = entity;
        let (start, end) = trim_entity_span(text, entity.start, entity.end);
        entity.start = start;
        entity.end = end;
        entity.text = text[start..end].to_string();
        entities.push(entity);
    }

    for token in tokens {
        let opens_new_entity = match token.label.strip_prefix("B-") {
            Some(entity_type) => Some(entity_type),
            None => match token.label.strip_prefix("I-") {
                // Continues the open entity only when the type matches,
                // otherwise it starts a new one.
                Some(entity_type) => match current {
                    Some(ref entity) if entity.entity_type == entity_type => None,
                    _ => Some(entity_type),
                },
                None => None,
            },
        };

        if let Some(entity_type) = opens_new_entity {
            if let Some(entity) = current.take() {
                flush(text, &mut entities, entity);
            }
            current = Some(BioEntity {
                entity_type: entity_type.to_string(),
                start: token.start,
                end: token.end,
                text: text[token.start..token.end].to_string(),
                confidence: token.confidence,
                token_count: 1,
            });
        } else if token.label.starts_with("I-") {
            // Continuation of the open entity of the same type.
            if let Some(ref mut entity) = current {
                entity.end = token.end;
                entity.text = text[entity.start..entity.end].to_string();
                entity.accumulate(token.confidence);
            }
        } else {
            // O tag closes any open entity.
            if let Some(entity) = current.take() {
                flush(text, &mut entities, entity);
            }
        }
    }

    if let Some(entity) = current.take() {
        flush(text, &mut entities, entity);
    }

    entities
}

impl TraditionalModernBertTokenClassifier {
    pub fn get_num_classes(&self) -> usize {
        self.num_classes
    }

    /// Create a new traditional ModernBERT token classifier with auto-detected variant
    pub fn new(model_id: &str, use_cpu: bool) -> Result<Self> {
        // Auto-detect variant from config.json
        let config_path_str = format!("{}/config.json", model_id);
        let variant = ModernBertVariant::detect_from_config(&config_path_str)
            .unwrap_or(ModernBertVariant::Standard);
        Self::new_with_variant(model_id, use_cpu, variant)
    }

    /// Create a new token classifier with explicit variant specification
    pub fn new_with_variant(
        model_id: &str,
        use_cpu: bool,
        variant: ModernBertVariant,
    ) -> Result<Self> {
        Self::new_with_limit(model_id, use_cpu, variant, None)
    }

    pub fn new_with_max_sequence_length(
        model_id: &str,
        use_cpu: bool,
        max_sequence_length: usize,
    ) -> Result<Self> {
        let variant = ModernBertVariant::detect_from_config(&format!("{model_id}/config.json"))?;
        Self::new_with_variant_and_max_sequence_length(
            model_id,
            use_cpu,
            variant,
            max_sequence_length,
        )
    }

    pub fn new_with_variant_and_max_sequence_length(
        model_id: &str,
        use_cpu: bool,
        variant: ModernBertVariant,
        max_sequence_length: usize,
    ) -> Result<Self> {
        Self::new_with_limit(model_id, use_cpu, variant, Some(max_sequence_length))
    }

    fn new_with_limit(
        model_id: &str,
        use_cpu: bool,
        variant: ModernBertVariant,
        max_sequence_length: Option<usize>,
    ) -> Result<Self> {
        let device = resolve_device(use_cpu);

        // Load model configuration
        let config_path = std::path::Path::new(model_id).join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| E::msg(format!("Failed to read config.json: {}", e)))?;
        let config = TraditionalModernBertClassifier::parse_model_config(&config_str)?;
        let max_sequence_length =
            TraditionalModernBertClassifier::resolve_sequence_length(&config, max_sequence_length)?;

        let tokenizer_path = std::path::Path::new(model_id).join("tokenizer.json");
        let base_tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| E::msg(format!("Failed to load tokenizer: {}", e)))?;
        let tokenizer = TraditionalModernBertClassifier::tokenizer_for_config(
            base_tokenizer,
            &config,
            variant,
            device.clone(),
            max_sequence_length,
        )?;

        // Load model weights
        let weights_path = std::path::Path::new(model_id).join("model.safetensors");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };

        // Load ModernBERT model (following old architecture pattern)
        let model = ModernBert::load(vb.clone(), &config)?;

        // Load head (optional) - following old architecture pattern
        let head = FixedModernBertHead::load_for_artifact(vb.pp("head"), &config, &config_str)?;

        // Get number of classes from config.json id2label field (single source of truth)
        // For models that don't include id2label in config.json,
        // we fall back to num_labels if available, or default to 2 (binary classification)
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let num_classes = config_json.get("id2label")
            .and_then(|v| v.as_object())
            .map(|obj| obj.len())
            .or_else(|| config_json.get("num_labels").and_then(|v| v.as_u64()).map(|n| n as usize))
            .unwrap_or_else(|| {
                // Default to 2 classes for binary token classification (e.g., SUPPORTED/HALLUCINATED)
                println!("  config.json missing id2label field, defaulting to 2 classes (binary classification)");
                2
            });

        // Load token classifier with correct number of classes
        let classifier =
            FixedModernBertTokenClassifier::load_with_classes(vb.clone(), &config, num_classes)?;

        drain_loader_queue(&device);
        Ok(Self {
            model: Arc::new(model),
            head,
            classifier,
            tokenizer,
            device,
            config,
            num_classes,
            labels: crate::ffi::classify::load_id2label_from_config(&config_path.to_string_lossy())
                .unwrap_or_default(),
            variant,
        })
    }

    /// Create mmBERT (multilingual) token classifier
    pub fn new_mmbert(model_id: &str, use_cpu: bool) -> Result<Self> {
        Self::new_with_variant(model_id, use_cpu, ModernBertVariant::Multilingual)
    }

    /// Create the historical multilingual 32K token-classifier variant.
    /// Capacity and default/YaRN math still come from config.json.
    /// Reference: https://huggingface.co/llm-semantic-router/mmbert-32k-yarn
    pub fn new_mmbert_32k(model_id: &str, use_cpu: bool) -> Result<Self> {
        Self::new_with_variant(model_id, use_cpu, ModernBertVariant::Multilingual32K)
    }

    /// Get the model variant (Standard, Multilingual, or Multilingual32K)
    pub fn variant(&self) -> ModernBertVariant {
        self.variant
    }

    /// Check if this is a multilingual (mmBERT) model (8K or 32K)
    pub fn is_multilingual(&self) -> bool {
        matches!(
            self.variant,
            ModernBertVariant::Multilingual | ModernBertVariant::Multilingual32K
        )
    }

    /// Historical variant predicate; this does not identify the active RoPE math.
    pub fn is_32k_yarn(&self) -> bool {
        self.variant == ModernBertVariant::Multilingual32K
    }

    /// Classify tokens in text
    pub fn classify_tokens(&self, text: &str) -> Result<Vec<TokenEntityTuple>> {
        run_on_inference_pool(&self.device, || self.classify_tokens_on_device(text))
    }

    /// Score an already tokenized task input without re-tokenizing its pair template.
    pub fn classify_encoded_tokens(
        &self,
        encoding: &tokenizers::Encoding,
    ) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            encoding.len() <= self.tokenizer.get_config().max_length,
            "input_limit: pair exceeds the token classifier input budget"
        );
        run_on_inference_pool(&self.device, || {
            let ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
            let mask = Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;
            let sequence = self.model.forward(&ids, &mask)?;
            let hidden = match &self.head {
                Some(head) => head.forward(&sequence)?,
                None => sequence,
            };
            let logits = self.classifier.forward(&hidden)?;
            Ok(ops::softmax(&logits, D::Minus1)?
                .squeeze(0)?
                .to_vec2::<f32>()?)
        })
    }

    /// Execute exact token IDs, reconcile overlaps, and decode BIO only once.
    pub fn classify_token_windows(
        &self,
        text: &str,
        plan: &crate::core::sequence_windows::EncodedTokenWindows,
    ) -> Result<Vec<TokenEntityTuple>> {
        run_on_inference_pool(&self.device, || {
            anyhow::ensure!(
                self.labels
                    .values()
                    .any(|label| label.starts_with("B-") || label.starts_with("I-")),
                "capability: token windows require a BIO token head"
            );
            let mut merger = crate::core::token_windows::TokenWindowMerger::new(
                plan.offsets.len(),
                self.labels.len(),
            );
            for window in &plan.windows {
                anyhow::ensure!(
                    window.ids.len() <= self.tokenizer.get_config().max_length,
                    "input_limit: token window exceeds the prepared model budget"
                );
                let input_ids =
                    Tensor::from_vec(window.ids.clone(), (1, window.ids.len()), &self.device)?;
                let mask = Tensor::ones((1, window.ids.len()), DType::U32, &self.device)?;
                let sequence = self.model.forward(&input_ids, &mask)?;
                let hidden = match &self.head {
                    Some(head) => head.forward(&sequence)?,
                    None => sequence,
                };
                let logits = self
                    .classifier
                    .forward(&hidden)?
                    .squeeze(0)?
                    .to_vec2::<f32>()?;
                merger
                    .add(window, plan.prefix_len, &logits)
                    .map_err(|e| anyhow::anyhow!("result_invalid: {e}"))?;
            }
            let logits = merger
                .finish()
                .map_err(|e| anyhow::anyhow!("result_invalid: {e}"))?;
            let mut labeled = Vec::with_capacity(logits.len());
            for (row, &(start, end)) in logits.iter().zip(&plan.offsets) {
                // Stable first-index argmax, matching the ordinary Candle task.
                let mut best = 0;
                for index in 1..row.len() {
                    if row[index] > row[best] {
                        best = index;
                    }
                }
                let maximum = row[best];
                let denominator: f32 = row.iter().map(|v| (v - maximum).exp()).sum();
                let label = self
                    .labels
                    .get(&best.to_string())
                    .ok_or_else(|| anyhow::anyhow!("result_invalid: missing token label"))?;
                labeled.push(BioToken {
                    label,
                    start,
                    end,
                    confidence: 1.0 / denominator,
                });
            }
            let mut results = Vec::new();
            for entity in merge_bio_entities(text, &labeled) {
                let class = self
                    .labels
                    .iter()
                    .filter_map(|(index, label)| {
                        (label
                            .strip_prefix("B-")
                            .or_else(|| label.strip_prefix("I-"))
                            == Some(entity.entity_type.as_str()))
                        .then(|| index.parse::<usize>().ok())
                        .flatten()
                    })
                    .min()
                    .ok_or_else(|| anyhow::anyhow!("result_invalid: missing entity label"))?;
                results.push((
                    entity.text,
                    class,
                    entity.confidence,
                    entity.start,
                    entity.end,
                ));
            }
            Ok(results)
        })
    }

    pub fn fit_prefix_to_window(&self, prefix: &str, suffix: &str) -> Result<String> {
        crate::core::tokenization_window::fit_prefix_to_window(
            self.tokenizer.as_ref(),
            prefix,
            suffix,
        )
    }

    fn classify_tokens_on_device(&self, text: &str) -> Result<Vec<TokenEntityTuple>> {
        // Tokenize the text
        let tokenization_result = self.tokenizer.tokenize(text)?;
        if tokenization_result.truncated {
            anyhow::bail!(
                "input exceeds the token classifier's {} token budget",
                self.tokenizer.get_config().max_length
            );
        }

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

        // Labels are captured with the binding at load time. Reloading a file on
        // every request would let an old generation silently adopt new semantics.
        let id2label = &self.labels;

        // Check if labels are BIO format (start with B- or I-) or simple format (like SUPPORTED/HALLUCINATED)
        let is_bio_format = id2label
            .values()
            .any(|v| v.starts_with("B-") || v.starts_with("I-"));

        // For simple token classification (non-BIO format), return individual token predictions
        if !is_bio_format {
            for (token_idx, token_probs) in probs_data.iter().enumerate() {
                if token_idx < tokenization_result.tokens.len()
                    && token_idx < tokenization_result.offsets.len()
                {
                    let pred_id = predictions_vec[token_idx] as usize;
                    let confidence = token_probs[pred_id];

                    let offset = tokenization_result.offsets[token_idx];
                    let token_text =
                        if offset.0 < text.len() && offset.1 <= text.len() && offset.0 < offset.1 {
                            text[offset.0..offset.1].to_string()
                        } else {
                            tokenization_result.tokens[token_idx].clone()
                        };

                    results.push((token_text, pred_id, confidence, offset.0, offset.1));
                }
            }
            return Ok(results);
        }

        // BIO tag entity extraction
        let labeled: Vec<BioToken<'_>> = predictions_vec
            .iter()
            .zip(tokenization_result.offsets.iter())
            .enumerate()
            // Skip special tokens (BOS, EOS, PAD, etc. have offset (0,0)).
            .filter(|(_, (_, offset))| !(offset.0 == 0 && offset.1 == 0))
            .map(|(i, (&pred_id, offset))| BioToken {
                label: id2label
                    .get(&pred_id.to_string())
                    .map(String::as_str)
                    .unwrap_or("O"),
                start: offset.0,
                end: offset.1,
                confidence: probs_data[i][pred_id as usize],
            })
            .collect();

        let entities = merge_bio_entities(text, &labeled);

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
        Some(&self.labels)
    }
}

mod instances;
pub use instances::ModernBertBackbone;

#[cfg(test)]
mod head_contract_tests {
    use super::*;
    use candle_nn::Module;
    use std::collections::HashMap;

    #[test]
    fn head_honors_configured_epsilon_and_rejects_partial_weights() -> candle_core::Result<()> {
        let config: Config = serde_json::from_value(serde_json::json!({
            "vocab_size":16,"hidden_size":4,"num_hidden_layers":2,"num_attention_heads":2,
            "intermediate_size":8,"max_position_embeddings":512,"layer_norm_eps":0.001,
            "pad_token_id":0,"global_attn_every_n_layers":2,"global_rope_theta":10000.0,
            "local_attention":4,"local_rope_theta":10000.0
        }))
        .unwrap();
        let device = Device::Cpu;
        let mut weights = HashMap::new();
        weights.insert("dense.weight".into(), Tensor::eye(4, DType::F32, &device)?);
        let partial = VarBuilder::from_tensors(weights.clone(), DType::F32, &device);
        assert!(FixedModernBertHead::load_for_artifact(partial, &config, "{}").is_err());
        weights.insert("norm.weight".into(), Tensor::ones(4, DType::F32, &device)?);
        let head = FixedModernBertHead::load(
            VarBuilder::from_tensors(weights, DType::F32, &device),
            &config,
        )?;
        let input = Tensor::new(&[[0.001f32, 0.002, 0.003, 0.004]], &device)?;
        let activated = input.gelu()?.to_vec2::<f32>()?.remove(0);
        let mean = activated.iter().sum::<f32>() / 4.0;
        let variance = activated.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        let actual = head.forward(&input)?.to_vec2::<f32>()?.remove(0);
        for (x, y) in activated.iter().zip(actual) {
            let expected = (x - mean) / (variance + config.layer_norm_eps as f32).sqrt();
            assert!((y - expected).abs() < 1e-5);
        }
        Ok(())
    }
}

pub mod reranker;
