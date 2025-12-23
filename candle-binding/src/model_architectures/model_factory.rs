//! Intelligent Model Factory - Dual-Path Selection
//!
//! This module provides a factory pattern for creating and managing both
//! Traditional and LoRA models through a unified interface, enabling seamless
//! switching between LoRACapable and TraditionalModel implementations.

use anyhow::{Error as E, Result};
use candle_core::Device;
use std::collections::HashMap;

use crate::model_architectures::config::PathSelectionStrategy;
use crate::model_architectures::lora::{LoRABertClassifier, LoRAMultiTaskResult};
use crate::model_architectures::routing::{DualPathRouter, ProcessingRequirements};
use crate::model_architectures::traditional::TraditionalBertClassifier;
use crate::model_architectures::traits::{
    FineTuningType, LoRACapable, ModelType, PoolingMethod, TaskType, TraditionalModel,
};
use crate::model_architectures::unified_interface::{
    ConfigurableModel, CoreModel, PathSpecialization,
};
//Import embedding models
use crate::model_architectures::embedding::{
    GemmaEmbeddingConfig, GemmaEmbeddingModel, Qwen3EmbeddingModel,
};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

/// Model factory configuration
#[derive(Debug, Clone)]
pub struct ModelFactoryConfig {
    /// Traditional model configuration
    pub traditional_config: Option<TraditionalModelConfig>,
    /// LoRA model configuration
    pub lora_config: Option<LoRAModelConfig>,
    /// Default path selection strategy
    pub default_strategy: PathSelectionStrategy,
    /// Use CPU for computation
    pub use_cpu: bool,
}

/// Traditional model configuration
#[derive(Debug, Clone)]
pub struct TraditionalModelConfig {
    /// Model identifier (HuggingFace Hub ID or local path)
    pub model_id: String,
    /// Number of classification classes
    pub num_classes: usize,
}

/// LoRA model configuration
#[derive(Debug, Clone)]
pub struct LoRAModelConfig {
    /// Base model identifier
    pub base_model_id: String,
    /// Path to LoRA adapters
    pub adapters_path: String,
    /// Task configurations
    pub task_configs: HashMap<TaskType, usize>,
}

/// Dual-path model wrapper that supports both LoRACapable and TraditionalModel traits
pub enum DualPathModel {
    /// Traditional model instance
    Traditional(TraditionalBertClassifier),
    /// LoRA model instance
    LoRA(LoRABertClassifier),
    /// Qwen3 embedding model
    Qwen3Embedding,
    /// Gemma embedding model
    GemmaEmbedding,
}

/// Intelligent model factory for dual-path architecture
pub struct ModelFactory {
    /// Available traditional models
    traditional_models: HashMap<String, TraditionalBertClassifier>,
    /// Available LoRA models
    lora_models: HashMap<String, LoRABertClassifier>,
    /// Qwen3 embedding model
    qwen3_embedding_model: Option<Qwen3EmbeddingModel>,
    /// Qwen3 tokenizer
    qwen3_tokenizer: Option<Tokenizer>,
    /// Qwen3 model path
    qwen3_model_path: Option<String>,
    /// Gemma embedding model
    gemma_embedding_model: Option<GemmaEmbeddingModel>,
    /// Gemma tokenizer
    gemma_tokenizer: Option<Tokenizer>,
    /// Gemma model path
    gemma_model_path: Option<String>,
    /// Intelligent router for path selection
    router: DualPathRouter,
    /// Computing device
    device: Device,
}

impl ModelFactory {
    /// Initialize the factory with device configuration
    pub fn new(device: Device) -> Self {
        Self {
            device,
            traditional_models: HashMap::new(),
            lora_models: HashMap::new(),
            qwen3_embedding_model: None,
            qwen3_tokenizer: None,
            qwen3_model_path: None,
            gemma_embedding_model: None,
            gemma_tokenizer: None,
            gemma_model_path: None,
            router: DualPathRouter::new(PathSelectionStrategy::Automatic),
        }
    }

    /// Register a traditional model
    pub fn register_traditional_model(
        &mut self,
        name: &str,
        model_id: String,
        num_classes: usize,
        use_cpu: bool,
    ) -> Result<()> {
        let model = TraditionalBertClassifier::new(&model_id, num_classes, use_cpu)?;
        self.traditional_models.insert(name.to_string(), model);

        Ok(())
    }

    /// Register a LoRA model
    pub fn register_lora_model(
        &mut self,
        name: &str,
        base_model_id: String,
        adapters_path: String,
        task_configs: HashMap<TaskType, usize>,
        use_cpu: bool,
    ) -> Result<()> {
        let model = LoRABertClassifier::new(&base_model_id, &adapters_path, task_configs, use_cpu)?;
        self.lora_models.insert(name.to_string(), model);

        Ok(())
    }

    /// Register Qwen3 embedding model
    pub fn register_qwen3_embedding_model(&mut self, model_path: &str) -> Result<()> {
        // Load model
        let model = Qwen3EmbeddingModel::load(model_path, &self.device)
            .map_err(|e| E::msg(format!("Failed to load Qwen3 model: {:?}", e)))?;

        // Load tokenizer
        let tokenizer_path = format!("{}/tokenizer.json", model_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            E::msg(format!(
                "Failed to load Qwen3 tokenizer from {}: {:?}",
                tokenizer_path, e
            ))
        })?;

        self.qwen3_embedding_model = Some(model);
        self.qwen3_tokenizer = Some(tokenizer);
        self.qwen3_model_path = Some(model_path.to_string());

        Ok(())
    }

    /// Register Gemma embedding model
    pub fn register_gemma_embedding_model(&mut self, model_path: &str) -> Result<()> {
        // Load config
        let config = GemmaEmbeddingConfig::from_pretrained(model_path)
            .map_err(|e| E::msg(format!("Failed to load Gemma config: {:?}", e)))?;

        // Build VarBuilder
        let safetensors_path = format!("{}/model.safetensors", model_path);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[safetensors_path.clone()],
                candle_core::DType::F32,
                &self.device,
            )
            .map_err(|e| E::msg(format!("Failed to load safetensors: {:?}", e)))?
        };

        // Load model
        let model = GemmaEmbeddingModel::load(model_path, &config, vb)
            .map_err(|e| E::msg(format!("Failed to load Gemma model: {:?}", e)))?;

        // Load tokenizer
        let tokenizer_path = format!("{}/tokenizer.json", model_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            E::msg(format!(
                "Failed to load Gemma tokenizer from {}: {:?}",
                tokenizer_path, e
            ))
        })?;

        self.gemma_embedding_model = Some(model);
        self.gemma_tokenizer = Some(tokenizer);
        self.gemma_model_path = Some(model_path.to_string());

        Ok(())
    }

    /// Create a dual-path model instance with intelligent routing
    pub fn create_dual_path_model(
        &self,
        requirements: &ProcessingRequirements,
    ) -> Result<DualPathModel> {
        let selection = self.router.select_path(requirements);

        match selection.selected_path {
            ModelType::Traditional => {
                if let Some(model) = self.traditional_models.get("default") {
                    Ok(DualPathModel::Traditional(
                        // Note: This is a conceptual example - in practice we might need to clone or use Rc/Arc
                        // For now, we'll create a simple reference wrapper
                        create_traditional_model_reference(model)?,
                    ))
                } else {
                    Err(E::msg("No traditional model available"))
                }
            }
            ModelType::LoRA => {
                if let Some(model) = self.lora_models.get("default") {
                    Ok(DualPathModel::LoRA(
                        // Note: Similar conceptual approach for LoRA models
                        create_lora_model_reference(model)?,
                    ))
                } else {
                    Err(E::msg("No LoRA model available"))
                }
            }
            ModelType::Qwen3Embedding => {
                // Direct routing to Qwen3 embedding model
                if self.qwen3_embedding_model.is_some() {
                    Ok(DualPathModel::Qwen3Embedding)
                } else {
                    Err(E::msg(
                        "Qwen3 embedding model not loaded. \
                         Please call init_embedding_models() with a valid Qwen3 model path.",
                    ))
                }
            }
            ModelType::GemmaEmbedding => {
                //  Direct routing to Gemma embedding model
                if self.gemma_embedding_model.is_some() {
                    Ok(DualPathModel::GemmaEmbedding)
                } else {
                    Err(E::msg(
                        "Gemma embedding model not loaded. \
                         Please call init_embedding_models() with a valid Gemma model path.",
                    ))
                }
            }
        }
    }

    /// Get available traditional models
    pub fn list_traditional_models(&self) -> Vec<&String> {
        self.traditional_models.keys().collect()
    }

    /// Get available LoRA models
    pub fn list_lora_models(&self) -> Vec<&String> {
        self.lora_models.keys().collect()
    }

    /// Get Qwen3 embedding model reference
    pub fn get_qwen3_model(&self) -> Option<&Qwen3EmbeddingModel> {
        self.qwen3_embedding_model.as_ref()
    }

    /// Get Qwen3 tokenizer reference
    pub fn get_qwen3_tokenizer(&self) -> Option<&Tokenizer> {
        self.qwen3_tokenizer.as_ref()
    }

    /// Get Gemma embedding model reference
    pub fn get_gemma_model(&self) -> Option<&GemmaEmbeddingModel> {
        self.gemma_embedding_model.as_ref()
    }

    /// Get Gemma tokenizer reference
    pub fn get_gemma_tokenizer(&self) -> Option<&Tokenizer> {
        self.gemma_tokenizer.as_ref()
    }

    /// Get Qwen3 model path
    pub fn get_qwen3_model_path(&self) -> Option<&str> {
        self.qwen3_model_path.as_deref()
    }

    /// Get Gemma model path
    pub fn get_gemma_model_path(&self) -> Option<&str> {
        self.gemma_model_path.as_deref()
    }

    /// Check if factory supports both paths
    pub fn supports_dual_path(&self) -> bool {
        !self.traditional_models.is_empty() && !self.lora_models.is_empty()
    }

    /// Get performance comparison between available models
    pub fn get_performance_comparison(&self) -> HashMap<ModelType, f32> {
        let mut comparison = HashMap::new();

        if !self.traditional_models.is_empty() {
            comparison.insert(ModelType::Traditional, 100.09); // ms, from benchmarks
        }

        if !self.lora_models.is_empty() {
            comparison.insert(ModelType::LoRA, 30.11); // ms, from benchmarks
        }

        comparison
    }
}

// Helper functions for model references (conceptual - would need proper implementation)
fn create_traditional_model_reference(
    _model: &TraditionalBertClassifier,
) -> Result<TraditionalBertClassifier> {
    // For now, return an error indicating this needs proper implementation
    // In practice, we might use Rc<RefCell<>>, Arc<Mutex<>>, or clone the model
    Err(E::msg(
        "Model reference creation not implemented - would need proper memory management",
    ))
}

fn create_lora_model_reference(_model: &LoRABertClassifier) -> Result<LoRABertClassifier> {
    // Similar to above - needs proper implementation
    Err(E::msg(
        "Model reference creation not implemented - would need proper memory management",
    ))
}

// Implement LoRACapable for DualPathModel
impl LoRACapable for DualPathModel {
    fn get_lora_rank(&self) -> usize {
        match self {
            DualPathModel::Traditional(_) => 0, // Traditional models don't have LoRA rank
            DualPathModel::LoRA(model) => model.get_lora_rank(),
            //Embedding models don't have LoRA rank
            DualPathModel::Qwen3Embedding | DualPathModel::GemmaEmbedding => 0,
        }
    }

    fn get_task_adapters(&self) -> Vec<TaskType> {
        match self {
            DualPathModel::Traditional(_) => vec![], // Traditional models don't have task adapters
            DualPathModel::LoRA(model) => model.get_task_adapters(),
            // Embedding models don't have task adapters
            DualPathModel::Qwen3Embedding | DualPathModel::GemmaEmbedding => vec![],
        }
    }

    fn supports_multi_task_parallel(&self) -> bool {
        match self {
            DualPathModel::Traditional(_) => false,
            DualPathModel::LoRA(model) => model.supports_multi_task_parallel(),
            //Embedding models don't support parallel multi-task
            DualPathModel::Qwen3Embedding | DualPathModel::GemmaEmbedding => false,
        }
    }
}

// Implement TraditionalModel trait for DualPathModel (3.2.2 requirement)
impl TraditionalModel for DualPathModel {
    type FineTuningConfig = serde_json::Value;

    fn get_fine_tuning_type(&self) -> FineTuningType {
        match self {
            DualPathModel::Traditional(_) => FineTuningType::Full, // Traditional models use full fine-tuning
            DualPathModel::LoRA(_) => FineTuningType::LayerWise, // LoRA uses layer-wise adaptation
            //Embedding models use full fine-tuning
            DualPathModel::Qwen3Embedding | DualPathModel::GemmaEmbedding => FineTuningType::Full,
        }
    }

    fn get_head_config(&self) -> Option<&Self::FineTuningConfig> {
        None // Not implemented yet
    }

    fn has_classification_head(&self) -> bool {
        match self {
            DualPathModel::Traditional(_) => true, // Traditional BERT models have classification heads
            DualPathModel::LoRA(_) => true,        // LoRA models support classification
            //Embedding models don't have classification heads
            DualPathModel::Qwen3Embedding | DualPathModel::GemmaEmbedding => false,
        }
    }

    fn has_token_classification_head(&self) -> bool {
        match self {
            DualPathModel::Traditional(_) => false, // Traditional BERT is for sequence classification
            DualPathModel::LoRA(_) => false,        // Not implemented yet
            //Embedding models don't have token classification heads
            DualPathModel::Qwen3Embedding | DualPathModel::GemmaEmbedding => false,
        }
    }

    fn sequential_forward(
        &self,
        input_ids: &candle_core::Tensor,
        attention_mask: &candle_core::Tensor,
        _task: TaskType,
    ) -> Result<Self::Output, Self::Error> {
        match self {
            DualPathModel::Traditional(model) => {
                let (class, confidence) = <TraditionalBertClassifier as CoreModel>::forward(
                    model,
                    input_ids,
                    attention_mask,
                )?;
                Ok(ModelOutput::Traditional { class, confidence })
            }
            DualPathModel::LoRA(model) => {
                // LoRA models can also do sequential processing
                let result =
                    <LoRABertClassifier as CoreModel>::forward(model, input_ids, attention_mask)?;
                Ok(ModelOutput::LoRA { result })
            }
            //Embedding models don't support sequential_forward (classification)
            DualPathModel::Qwen3Embedding | DualPathModel::GemmaEmbedding => {
                Err(candle_core::Error::Msg(
                    "Embedding models don't support classification (sequential_forward)"
                        .to_string(),
                ))
            }
        }
    }

    fn compatibility_version(&self) -> &str {
        "v1.0-dual-path-factory"
    }
}

/// Embedding model output
///
/// Represents the output from an embedding model, containing the generated
/// embedding vector and metadata about the pooling method used.
#[derive(Debug, Clone)]
pub struct EmbeddingOutput {
    /// The generated embedding tensor
    ///
    /// Shape: `[batch_size, embedding_dim]` or `[batch_size, target_dim]` for Matryoshka
    pub embedding: candle_core::Tensor,

    /// Dimension of the embedding
    ///
    /// This is the actual dimension of the returned embedding, which may be
    /// less than the model's full dimension if Matryoshka truncation was applied.
    ///
    /// ## Examples
    /// - Full dimension: 768
    /// - Matryoshka dimensions: 512, 256, 128
    pub dim: usize,

    /// Pooling method used to generate this embedding
    ///
    /// ## Values
    /// - `PoolingMethod::LastToken`: Qwen3-style last token pooling
    /// - `PoolingMethod::Mean`: BERT/Gemma-style mean pooling
    /// - `PoolingMethod::CLS`: Original BERT CLS token
    pub pooling_method: PoolingMethod,
}

/// Unified output type for multi-path models
///
/// Extended from dual-path (Traditional, LoRA) to support embedding models.
#[derive(Debug, Clone)]
pub enum ModelOutput {
    /// Traditional model output
    Traditional { class: usize, confidence: f32 },
    /// LoRA model output
    LoRA { result: LoRAMultiTaskResult },
    /// Embedding model output
    ///
    /// Used by long-context embedding models like Qwen3 and GemmaEmbedding.
    Embedding { output: EmbeddingOutput },
}

impl std::fmt::Debug for DualPathModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DualPathModel::Traditional(_) => f.debug_struct("DualPathModel::Traditional").finish(),
            DualPathModel::LoRA(_) => f.debug_struct("DualPathModel::LoRA").finish(),
            // Embedding models
            DualPathModel::Qwen3Embedding => {
                f.debug_struct("DualPathModel::Qwen3Embedding").finish()
            }
            DualPathModel::GemmaEmbedding => {
                f.debug_struct("DualPathModel::GemmaEmbedding").finish()
            }
        }
    }
}

/// Implementation of CoreModel
///
/// This provides a unified interface that automatically delegates to the
/// appropriate Traditional or LoRA implementation.
impl CoreModel for DualPathModel {
    type Config = ModelFactoryConfig;
    type Error = candle_core::Error;
    type Output = ModelOutput;

    fn model_type(&self) -> ModelType {
        // Direct implementation (copied from deleted ModelBackbone)
        match self {
            DualPathModel::Traditional(_) => ModelType::Traditional,
            DualPathModel::LoRA(_) => ModelType::LoRA,
            //Precise embedding model types
            DualPathModel::Qwen3Embedding => ModelType::Qwen3Embedding,
            DualPathModel::GemmaEmbedding => ModelType::GemmaEmbedding,
        }
    }

    fn forward(
        &self,
        input_ids: &candle_core::Tensor,
        attention_mask: &candle_core::Tensor,
    ) -> Result<Self::Output, Self::Error> {
        // Direct implementation (copied from deleted ModelBackbone)
        match self {
            DualPathModel::Traditional(model) => {
                let (class, confidence) = <TraditionalBertClassifier as CoreModel>::forward(
                    model,
                    input_ids,
                    attention_mask,
                )?;
                Ok(ModelOutput::Traditional { class, confidence })
            }
            DualPathModel::LoRA(model) => {
                let result =
                    <LoRABertClassifier as CoreModel>::forward(model, input_ids, attention_mask)?;
                Ok(ModelOutput::LoRA { result })
            }
            //Embedding models don't support classification via CoreModel::forward
            DualPathModel::Qwen3Embedding | DualPathModel::GemmaEmbedding => {
                Err(candle_core::Error::Msg(
                    "Embedding models don't support classification (CoreModel::forward)"
                        .to_string(),
                ))
            }
        }
    }

    fn get_config(&self) -> &Self::Config {
        // DualPathModel will need to store config when struct is updated
        unimplemented!("get_config will be implemented when ModelFactoryConfig is stored in struct")
    }
}

/// Implementation of PathSpecialization for DualPathModel
///
/// This provides intelligent path-specific characteristics that adapt
/// based on the currently active path (Traditional or LoRA).
impl PathSpecialization for DualPathModel {
    fn supports_parallel(&self) -> bool {
        // Direct implementation (copied from deleted ModelBackbone)
        match self {
            DualPathModel::Traditional(model) => {
                <TraditionalBertClassifier as PathSpecialization>::supports_parallel(model)
            }
            DualPathModel::LoRA(model) => {
                <LoRABertClassifier as PathSpecialization>::supports_parallel(model)
            }
            // Embedding models support parallel processing
            DualPathModel::Qwen3Embedding | DualPathModel::GemmaEmbedding => true,
        }
    }

    fn get_confidence_threshold(&self) -> f32 {
        // Direct implementation (copied from deleted ModelBackbone)
        match self {
            DualPathModel::Traditional(model) => {
                <TraditionalBertClassifier as PathSpecialization>::get_confidence_threshold(model)
            }
            DualPathModel::LoRA(model) => {
                <LoRABertClassifier as PathSpecialization>::get_confidence_threshold(model)
            }
            //Embedding models don't have classification confidence threshold
            DualPathModel::Qwen3Embedding | DualPathModel::GemmaEmbedding => 0.0,
        }
    }

    fn optimal_batch_size(&self) -> usize {
        match self {
            DualPathModel::Traditional(_) => 16, // Conservative for traditional
            DualPathModel::LoRA(_) => 32,        // Efficient for LoRA
            //Embedding models can handle larger batches
            DualPathModel::Qwen3Embedding => 64, // Qwen3 supports 32K context
            DualPathModel::GemmaEmbedding => 48, // Gemma is smaller, faster
        }
    }
}

/// Implementation of ConfigurableModel for DualPathModel
///
/// This enables factory-pattern model creation using the new interface.
impl ConfigurableModel for DualPathModel {
    fn load(_config: &Self::Config, _device: &candle_core::Device) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        // DualPathModel has complex factory-based initialization
        // This will be properly implemented when ModelFactory is refactored
        unimplemented!("ConfigurableModel::load will be implemented when ModelFactory is refactored for new interface")
    }
}
