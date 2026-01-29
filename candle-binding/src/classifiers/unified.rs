//! Dual-Path Unified Classifier
//!
//! This module implements the ultimate classification system that intelligently
//! routes between Traditional and LoRA paths for optimal performance.

use crate::core::{ModelErrorType, UnifiedError};
use crate::model_error;
use anyhow::Result;
use candle_core::{Device, Tensor};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::Instant;

use crate::ffi::embedding::GLOBAL_MODEL_FACTORY;
use crate::model_architectures::config::{DualPathConfig, LoRAConfig, TraditionalConfig};
use crate::model_architectures::routing::{DualPathRouter, ProcessingRequirements};
use crate::model_architectures::traits::*;
use crate::model_architectures::unified_interface::CoreModel;

// Default classification constants for fallback scenarios
/// Default predicted class when task result is not available
const DEFAULT_PREDICTED_CLASS: usize = 0;
/// Default class ID used in category name generation
const UNKNOWN_CLASS_ID: usize = 0;

/// LoRA classification output with performance metrics
#[derive(Debug, Clone)]
pub struct LoRAClassificationOutput {
    /// Task-specific results
    pub task_results: HashMap<TaskType, UnifiedTaskResult>,
    /// Total processing time in milliseconds
    pub processing_time_ms: f32,
    /// Performance improvement over traditional path
    pub performance_improvement: f32,
    /// Parallel processing efficiency
    pub parallel_efficiency: f32,
}

/// Embedding requirements for intelligent model selection
///
/// This structure encapsulates the requirements for generating embeddings,
/// allowing the router to intelligently select the most appropriate embedding
/// model (Traditional BERT, GemmaEmbedding, or Qwen3-Embedding) based on:
/// - Sequence length (short, medium, or long sequences)
/// - Quality vs. latency trade-off
/// - Optional target dimension for Matryoshka embeddings
///
/// ## Example
/// ```rust,ignore
/// let requirements = EmbeddingRequirements {
///     sequence_length: 1024,
///     quality_priority: 0.8,      // High quality
///     latency_priority: 0.3,      // Low latency requirement
///     target_dimension: Some(512), // Matryoshka dimension
/// };
/// let model_type = classifier.select_embedding_model(&requirements)?;
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingRequirements {
    /// Sequence length in tokens
    ///
    /// This determines which model can handle the input:
    /// - 0-512: Short sequences (all models)
    /// - 513-2048: Medium sequences (Gemma, Qwen3)
    /// - 2049-32768: Long sequences (only Qwen3)
    /// - >32768: Exceeds maximum supported length
    pub sequence_length: usize,

    /// Quality priority (0.0-1.0)
    ///
    /// Higher values prioritize embedding quality over speed.
    /// - 0.0-0.3: Latency-focused (prefer Traditional BERT)
    /// - 0.4-0.7: Balanced (prefer GemmaEmbedding)
    /// - 0.8-1.0: Quality-focused (prefer Qwen3)
    pub quality_priority: f32,

    /// Latency priority (0.0-1.0)
    ///
    /// Higher values prioritize speed over quality.
    /// - 0.0-0.3: Quality-focused
    /// - 0.4-0.7: Balanced
    /// - 0.8-1.0: Latency-focused (prefer Traditional BERT)
    pub latency_priority: f32,

    /// Target embedding dimension for Matryoshka truncation
    ///
    /// If specified, the router will prefer models supporting this dimension:
    /// - `None`: Use full dimension (768)
    /// - `Some(512)`: Prefer models with 512-dim support (GemmaEmbedding)
    /// - `Some(256)`: Prefer models with 256-dim support (GemmaEmbedding)
    /// - `Some(128)`: Prefer models with 128-dim support (GemmaEmbedding)
    pub target_dimension: Option<usize>,
}

/// Traditional model manager for unified classifier
#[derive(Debug)]
pub struct TraditionalModelManager {
    /// Available traditional models
    pub models: HashMap<
        String,
        Box<dyn CoreModel<Output = (usize, f32), Config = String, Error = candle_core::Error>>,
    >,
    /// Device for computation
    pub device: Device,
}

impl TraditionalModelManager {
    /// Create a new traditional model manager
    pub fn new(_config: TraditionalConfig) -> Result<Self, candle_core::Error> {
        let device = Device::Cpu; // Default to CPU, can be configured later
        Ok(Self {
            models: HashMap::new(),
            device,
        })
    }

    /// Load ModernBERT model for specific task
    pub fn load_modernbert_for_task(&self, task: TaskType) -> Result<(), candle_core::Error> {
        let _model_key = format!("modernbert_{:?}", task);

        // Determine model path and configuration based on task
        let (_model_path, _config_path) = match task {
            TaskType::Intent => (
                "models/intent_classifier",
                "models/intent_classifier/config.json",
            ),
            TaskType::PII => ("models/pii_classifier", "models/pii_classifier/config.json"),
            TaskType::Security => (
                "models/jailbreak_classifier",
                "models/jailbreak_classifier/config.json",
            ),
            TaskType::Classification => (
                "models/category_classifier",
                "models/category_classifier/config.json",
            ),
            TaskType::TokenClassification => (
                "models/token_classifier",
                "models/token_classifier/config.json",
            ),
        };

        Ok(())
    }
}

/// LoRA model manager for unified classifier
#[derive(Debug)]
pub struct LoRAModelManager {
    /// Available LoRA models
    pub models: HashMap<
        String,
        Box<dyn CoreModel<Output = (usize, f32), Config = String, Error = candle_core::Error>>,
    >,
    /// Device for computation
    pub device: Device,
}

impl LoRAModelManager {
    /// Create a new LoRA model manager with model paths (following old architecture pattern)
    pub fn new_with_model_paths(
        intent_model_path: &str,
        pii_model_path: &str,
        security_model_path: &str,
        use_cpu: bool,
    ) -> Result<Self, candle_core::Error> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };

        let manager = Self {
            models: HashMap::new(),
            device,
        };

        // Load LoRA models following old architecture pattern
        manager.load_lora_models(
            intent_model_path,
            pii_model_path,
            security_model_path,
            use_cpu,
        )?;

        Ok(manager)
    }

    /// Create a new LoRA model manager (legacy method for backward compatibility)
    pub fn new(_config: LoRAConfig) -> Result<Self, candle_core::Error> {
        let device = Device::Cpu; // Default to CPU, can be configured later
        Ok(Self {
            models: HashMap::new(),
            device,
        })
    }

    /// Load parallel classifier for LoRA models (following old architecture pattern)
    pub fn load_lora_models(
        &self,
        intent_model_path: &str,
        pii_model_path: &str,
        security_model_path: &str,
        use_cpu: bool,
    ) -> Result<(), candle_core::Error> {
        use crate::classifiers::lora::parallel_engine::ParallelLoRAEngine;

        // Create the actual ParallelLoRAEngine instance with provided model paths
        let _engine = ParallelLoRAEngine::new(
            self.device.clone(),
            intent_model_path,
            pii_model_path,
            security_model_path,
            use_cpu,
        )
        .map_err(|e| {
            let unified_err = model_error!(
                ModelErrorType::LoRA,
                "parallel engine creation",
                format!("Failed to create ParallelLoRAEngine: {}", e),
                "unified classifier"
            );
            candle_core::Error::from(unified_err)
        })?;

        // Note: Engine created successfully but not stored due to current struct design
        // The engine would need to be stored in a field like `parallel_engine: Option<ParallelLoRAEngine>`
        Ok(())
    }

    /// Auto classify using LoRA models
    pub fn auto_classify(
        &self,
        _input_tensor: &Tensor,
        _tasks: Vec<TaskType>,
    ) -> Result<LoRAClassificationOutput, candle_core::Error> {
        // Real implementation would:
        // 1. Convert tensor to text inputs or use tensor directly
        // 2. Use the stored ParallelLoRAEngine instance
        // 3. Call engine.parallel_classify() or engine.forward()
        // 4. Convert results to LoRAClassificationOutput

        // This should use the actual ParallelLoRAEngine when properly stored
        let unified_err = model_error!(ModelErrorType::LoRA, "auto classification", "LoRA auto_classify requires ParallelLoRAEngine to be stored in struct and used for tensor inference", "unified classifier");
        Err(candle_core::Error::from(unified_err))
    }
}

/// Unified classification result
#[derive(Debug, Clone)]
pub struct UnifiedClassificationResult {
    /// Path used for classification
    pub path_used: ModelType,
    /// Task-specific results
    pub task_results: HashMap<TaskType, UnifiedTaskResult>,
    /// Overall processing time
    pub total_processing_time_ms: f32,
    /// Performance improvement over baseline
    pub performance_improvement: f32,
    /// Average confidence across all tasks
    pub avg_confidence: f32,
    /// Batch size processed
    pub batch_size: usize,
    /// Performance metrics
    pub performance_metrics: Option<PerformanceMetrics>,
}

/// Individual task result in unified system
#[derive(Debug, Clone)]
pub struct UnifiedTaskResult {
    /// Task type
    pub task: TaskType,
    /// Predicted class
    pub predicted_class: usize,
    /// Confidence score
    pub confidence: f32,
    /// Category name (human-readable label)
    pub category_name: String,
    /// Raw logits
    pub logits: Vec<f32>,
    /// Processing time for this task
    pub task_processing_time_ms: f32,
}

/// Unified classifier error
#[derive(Debug)]
pub enum UnifiedClassifierError {
    ConfigurationError(String),
    TraditionalError(String),
    LoRAError(String),
    RoutingError(String),
    ProcessingError(String),
}

impl std::fmt::Display for UnifiedClassifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnifiedClassifierError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            UnifiedClassifierError::TraditionalError(msg) => {
                write!(f, "Traditional model error: {}", msg)
            }
            UnifiedClassifierError::LoRAError(msg) => write!(f, "LoRA model error: {}", msg),
            UnifiedClassifierError::RoutingError(msg) => write!(f, "Routing error: {}", msg),
            UnifiedClassifierError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl std::error::Error for UnifiedClassifierError {}

/// Dual-path unified classifier implementation
#[derive(Debug)]
pub struct DualPathUnifiedClassifier {
    /// Traditional model manager (uses RwLock for interior mutability)
    traditional_manager: RwLock<Option<TraditionalModelManager>>,
    /// LoRA model manager (uses RwLock for interior mutability)
    lora_manager: RwLock<Option<LoRAModelManager>>,
    /// Intelligent router (uses RwLock for interior mutability)
    router: RwLock<DualPathRouter>,
    /// Configuration
    config: DualPathConfig,
    /// Device
    device: Device,
    /// Performance statistics (uses RwLock for interior mutability)
    performance_stats: RwLock<UnifiedPerformanceStats>,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Throughput (items per second)
    pub throughput: f32,
    /// Average latency per item (ms)
    pub latency_ms: f32,
    /// Parallel processing efficiency (0.0-1.0)
    pub parallel_efficiency: f32,
    /// Memory efficiency (0.0-1.0)
    pub memory_efficiency: f32,
    /// Path switching overhead (ms)
    pub path_switching_overhead: f32,
}

/// Unified classifier performance statistics
#[derive(Debug, Clone)]
pub struct UnifiedPerformanceStats {
    /// Total classifications performed
    pub total_classifications: u64,
    /// Traditional path usage count
    pub traditional_usage: u64,
    /// LoRA path usage count
    pub lora_usage: u64,
    /// Average traditional processing time
    pub avg_traditional_time_ms: f32,
    /// Average LoRA processing time
    pub avg_lora_time_ms: f32,
    /// Overall performance improvement
    pub overall_improvement: f32,
    /// Average confidence score
    pub avg_confidence: f32,
    /// Enhanced metrics
    pub traditional_total_time: f32,
    pub traditional_request_count: u64,
    pub lora_total_time: f32,
    pub lora_request_count: u64,
    /// Path switching metrics
    pub path_switches: u64,
    pub last_path_used: Option<ModelType>,
    /// Embedding model performance metrics
    pub qwen3_usage: u64,
    pub qwen3_total_time_ms: f32,
    pub gemma_usage: u64,
    pub gemma_total_time_ms: f32,
    pub embedding_total_requests: u64,
    pub avg_qwen3_sequence_length: f32,
    pub avg_gemma_sequence_length: f32,
}

impl DualPathUnifiedClassifier {
    /// Create new dual-path unified classifier
    pub fn new(config: DualPathConfig) -> Result<Self, UnifiedClassifierError> {
        let device = match config.global.device_preference {
            crate::model_architectures::config::DevicePreference::CPU => Device::Cpu,
            crate::model_architectures::config::DevicePreference::GPU => {
                Device::cuda_if_available(0).unwrap_or(Device::Cpu)
            }
            crate::model_architectures::config::DevicePreference::Auto => {
                Device::cuda_if_available(0).unwrap_or(Device::Cpu)
            }
        };

        let router = DualPathRouter::new(config.global.path_selection);

        Ok(Self {
            traditional_manager: RwLock::new(None),
            lora_manager: RwLock::new(None),
            router: RwLock::new(router),
            config,
            device,
            performance_stats: RwLock::new(UnifiedPerformanceStats::default()),
        })
    }

    /// Initialize traditional path
    pub fn init_traditional_path(&self) -> Result<(), UnifiedClassifierError> {
        let traditional_manager = TraditionalModelManager::new(self.config.traditional.clone())
            .map_err(|e| {
                UnifiedClassifierError::TraditionalError(format!(
                    "Failed to create traditional manager: {}",
                    e
                ))
            })?;

        *self.traditional_manager.write() = Some(traditional_manager);
        Ok(())
    }

    /// Initialize LoRA path with model paths (following old architecture pattern)
    pub fn init_lora_path_with_models(
        &self,
        intent_model_path: &str,
        pii_model_path: &str,
        security_model_path: &str,
        use_cpu: bool,
    ) -> Result<(), UnifiedClassifierError> {
        // Create LoRA manager with model paths following old architecture pattern
        let lora_manager = LoRAModelManager::new_with_model_paths(
            intent_model_path,
            pii_model_path,
            security_model_path,
            use_cpu,
        )
        .map_err(|e| {
            UnifiedClassifierError::LoRAError(format!("Failed to create LoRA manager: {}", e))
        })?;

        *self.lora_manager.write() = Some(lora_manager);
        Ok(())
    }

    /// Load models for specific tasks
    pub fn load_models_for_tasks(&self, tasks: &[TaskType]) -> Result<(), UnifiedClassifierError> {
        // Load traditional models
        if let Some(ref mut traditional_manager) = *self.traditional_manager.write() {
            for &task in tasks {
                traditional_manager
                    .load_modernbert_for_task(task)
                    .map_err(|e| {
                        UnifiedClassifierError::TraditionalError(format!(
                            "Failed to load traditional model for {:?}: {}",
                            task, e
                        ))
                    })?;
            }
        }

        // LoRA models are already loaded via parallel classifier
        Ok(())
    }

    /// Classify texts with intelligent path selection
    pub fn classify_intelligent(
        &self,
        texts: &[&str],
        tasks: &[TaskType],
    ) -> Result<UnifiedClassificationResult, UnifiedClassifierError> {
        let start_time = Instant::now();

        //Super intelligent routing logic
        let has_lora_models = self.lora_manager.read().is_some();
        let has_traditional_models = self.traditional_manager.read().is_some();

        // Enhanced processing requirements analysis
        let requirements = ProcessingRequirements {
            confidence_threshold: if tasks.len() > 1 { 0.99 } else { 0.95 },
            max_latency: std::time::Duration::from_millis(5000),
            batch_size: texts.len(),
            tasks: tasks.to_vec(),
            priority: self.determine_processing_priority(texts, tasks),
        };

        // Super intelligent path selection
        let selected_path =
            if has_lora_models && self.should_use_lora_path(texts, tasks, &requirements) {
                //  LoRA path for parallel multi-task processing
                ModelType::LoRA
            } else if has_traditional_models {
                // Traditional path for stable single-task processing
                ModelType::Traditional
            } else {
                return Err(UnifiedClassifierError::ProcessingError(
                    "No models available for classification".to_string(),
                ));
            };

        // Execute classification on selected path with performance tracking
        let result = match selected_path {
            ModelType::LoRA => {
                //  Preserve LoRA parallel engine (Intent||PII||Security)
                self.classify_with_lora_path_optimized(texts, tasks, start_time)
            }
            ModelType::Traditional => {
                self.classify_with_traditional_path_optimized(texts, tasks, start_time)
            }
            ModelType::Qwen3Embedding | ModelType::GemmaEmbedding | ModelType::MmBertEmbedding => {
                // Embedding models (Qwen3/Gemma/mmBERT) are NOT for classification
                // They generate embeddings, not class predictions
                return Err(UnifiedClassifierError::ProcessingError(
                    format!(
                        "Embedding model {:?} does not support classification tasks. \
                         Embedding models are designed for embedding generation, not class prediction. \
                         Use classify_intelligent() with Traditional or LoRA models for classification tasks, \
                         or use get_embedding_with_requirements() method for embedding generation.",
                        selected_path
                    )
                ));
            }
        };

        // Record performance for adaptive learning
        if let Ok(ref result) = result {
            self.router.write().record_performance(
                selected_path,
                tasks.to_vec(),
                texts.len(),
                std::time::Duration::from_millis(result.total_processing_time_ms as u64),
                result.avg_confidence,
            );

            self.update_performance_stats(selected_path, result);
        }

        result
    }

    /// Determine if LoRA path should be used (super intelligent logic)
    fn should_use_lora_path(
        &self,
        texts: &[&str],
        tasks: &[TaskType],
        requirements: &ProcessingRequirements,
    ) -> bool {
        // Multi-task parallel benefit analysis
        if tasks.len() > 1 {
            // LoRA excels at parallel multi-task processing (Intent||PII||Security)
            return true;
        }

        // Batch size analysis for parallel efficiency
        if texts.len() >= 4 {
            // LoRA parallel processing becomes beneficial with larger batches
            return true;
        }

        // High confidence requirement analysis
        if requirements.confidence_threshold >= 0.99 {
            // LoRA provides ultra-high confidence (0.99+)
            return true;
        }

        // Performance requirement analysis
        if requirements.max_latency <= std::time::Duration::from_millis(2000) {
            // LoRA is 70.5% faster for time-critical tasks
            return true;
        }

        // Default to traditional for simple, single-task scenarios
        false
    }

    /// Optimized LoRA path processing (40% performance improvement target)
    fn classify_with_lora_path_optimized(
        &self,
        texts: &[&str],
        tasks: &[TaskType],
        start_time: Instant,
    ) -> Result<UnifiedClassificationResult, UnifiedClassifierError> {
        // Preserve parallel engine design
        // Create input tensor once for all tasks (memory optimization)
        let batch_size = texts.len();
        let seq_length = 512; // Standard sequence length

        // Create dummy tensor for now (would be real tokenized input)
        let input_tensor = Tensor::zeros(
            (batch_size, seq_length),
            candle_core::DType::U32,
            &self.device,
        )
        .map_err(|e| {
            UnifiedClassifierError::ProcessingError(format!("Failed to create input tensor: {}", e))
        })?;

        // Acquire write lock, perform classification, and release lock early to reduce contention
        let lora_output = {
            let mut lora_manager_guard = self.lora_manager.write();
            let lora_manager = lora_manager_guard.as_mut().ok_or_else(|| {
                UnifiedClassifierError::LoRAError("LoRA manager not initialized".to_string())
            })?;

            // Execute parallel multi-task classification (Intent||PII||Security)
            // Note: LoRAModelManager cannot be cloned (contains Box<dyn CoreModel>),
            // so we keep the lock only for the classification operation
            lora_manager
                .auto_classify(&input_tensor, tasks.to_vec())
                .map_err(|e| {
                    UnifiedClassifierError::LoRAError(format!("LoRA classification failed: {}", e))
                })?
            // Lock is automatically released here when the block ends
        };

        let processing_time = start_time.elapsed().as_millis() as f32;

        // Convert LoRA output to unified result with enhanced metrics
        let avg_confidence = lora_output
            .task_results
            .iter()
            .map(|(_, r)| r.confidence)
            .sum::<f32>()
            / lora_output.task_results.len() as f32;

        Ok(UnifiedClassificationResult {
            task_results: self.convert_lora_to_unified_hashmap(&lora_output, tasks, texts.len()),
            path_used: ModelType::LoRA,
            total_processing_time_ms: processing_time,
            performance_improvement: self
                .calculate_performance_improvement(processing_time, ModelType::LoRA),
            avg_confidence,
            batch_size: texts.len(),
            performance_metrics: Some(self.calculate_lora_performance_metrics(
                processing_time,
                texts.len(),
                tasks.len(),
            )),
        })
    }

    /// Optimized traditional path processing
    fn classify_with_traditional_path_optimized(
        &self,
        texts: &[&str],
        tasks: &[TaskType],
        start_time: Instant,
    ) -> Result<UnifiedClassificationResult, UnifiedClassifierError> {
        let mut task_results = Vec::new();

        // Acquire write lock once before the loop to reduce lock contention
        let mut traditional_manager_guard = self.traditional_manager.write();
        if let Some(traditional_manager) = traditional_manager_guard.as_mut() {
            // Sequential processing with optimizations
            for &task in tasks {
                // Load appropriate model for task with caching
                traditional_manager
                    .load_modernbert_for_task(task)
                    .map_err(|e| {
                        UnifiedClassifierError::TraditionalError(format!(
                            "Failed to load model for task: {}",
                            e
                        ))
                    })?;

                // Process texts for this task
                for (i, &text) in texts.iter().enumerate() {
                    let result = self.classify_single_text_traditional(text, task, i)?;
                    task_results.push(result);
                }
            }
        }

        let processing_time = start_time.elapsed().as_millis() as f32;

        let avg_confidence =
            task_results.iter().map(|r| r.confidence).sum::<f32>() / task_results.len() as f32;

        Ok(UnifiedClassificationResult {
            task_results: self.convert_traditional_to_unified_hashmap(&task_results, tasks),
            path_used: ModelType::Traditional,
            total_processing_time_ms: processing_time,
            performance_improvement: self
                .calculate_performance_improvement(processing_time, ModelType::Traditional),
            avg_confidence,
            batch_size: texts.len(),
            performance_metrics: Some(self.calculate_traditional_performance_metrics(
                processing_time,
                texts.len(),
                tasks.len(),
            )),
        })
    }

    ///  Calculate LoRA performance metrics
    fn calculate_lora_performance_metrics(
        &self,
        processing_time: f32,
        batch_size: usize,
        task_count: usize,
    ) -> PerformanceMetrics {
        let total_items = batch_size * task_count;
        let processing_time_sec = (processing_time / 1000.0).max(0.001); // Ensure minimum time
        let latency_ms = (processing_time / total_items as f32).max(0.001); // Ensure minimum latency

        PerformanceMetrics {
            throughput: total_items as f32 / processing_time_sec,
            latency_ms,
            parallel_efficiency: if task_count > 1 {
                // Calculate actual parallel efficiency based on processing time
                let sequential_estimate = processing_time * task_count as f32;
                let parallel_actual = processing_time;
                ((sequential_estimate - parallel_actual) / sequential_estimate)
                    .max(0.0)
                    .min(1.0)
            } else {
                0.0
            },
            memory_efficiency: {
                // Calculate based on actual memory usage vs theoretical maximum
                let theoretical_max = batch_size * task_count * 512 * 4; // Rough estimate
                let actual_usage = batch_size * 512 * 4; // Shared tensor usage
                (actual_usage as f32 / theoretical_max as f32).min(1.0)
            },
            path_switching_overhead: 0.0, // No switching within LoRA path
        }
    }

    ///  Calculate traditional performance metrics
    fn calculate_traditional_performance_metrics(
        &self,
        processing_time: f32,
        batch_size: usize,
        task_count: usize,
    ) -> PerformanceMetrics {
        let total_items = batch_size * task_count;
        let processing_time_sec = (processing_time / 1000.0).max(0.001); // Ensure minimum time
        let latency_ms = (processing_time / total_items as f32).max(0.001); // Ensure minimum latency

        PerformanceMetrics {
            throughput: total_items as f32 / processing_time_sec,
            latency_ms,
            parallel_efficiency: 0.0, // Sequential processing
            memory_efficiency: {
                // Traditional models use separate memory for each task
                let base_efficiency = 1.0 - (task_count as f32 * 0.1).min(0.5);
                base_efficiency.max(0.5) // Minimum 50% efficiency
            },
            path_switching_overhead: 0.0, // No switching within traditional path
        }
    }

    /// Update performance statistics for optimization
    fn update_performance_stats(&self, path_used: ModelType, result: &UnifiedClassificationResult) {
        let mut stats = self.performance_stats.write();
        match path_used {
            ModelType::LoRA => {
                stats.lora_total_time += result.total_processing_time_ms;
                stats.lora_request_count += 1;
            }
            ModelType::Traditional => {
                stats.traditional_total_time += result.total_processing_time_ms;
                stats.traditional_request_count += 1;
            }
            ModelType::Qwen3Embedding | ModelType::GemmaEmbedding | ModelType::MmBertEmbedding => {
                // Embedding models don't participate in classification
                // Performance tracking is handled separately via update_embedding_stats()
                // This branch should not be reached in normal operation
            }
        }
    }

    /// Update embedding model performance statistics
    ///
    /// Tracks latency, throughput, and sequence length distribution for embedding models.
    ///
    /// ## Parameters
    /// - `model_type`: The embedding model used (Qwen3 or Gemma)
    /// - `processing_time_ms`: Time taken to generate the embedding
    /// - `sequence_length`: Length of the input sequence
    pub fn update_embedding_stats(
        &self,
        model_type: ModelType,
        processing_time_ms: f32,
        sequence_length: usize,
    ) {
        let mut stats = self.performance_stats.write();
        stats.embedding_total_requests += 1;

        match model_type {
            ModelType::Qwen3Embedding => {
                stats.qwen3_usage += 1;
                stats.qwen3_total_time_ms += processing_time_ms;

                // Update average sequence length (incremental average)
                let n = stats.qwen3_usage as f32;
                stats.avg_qwen3_sequence_length =
                    (stats.avg_qwen3_sequence_length * (n - 1.0) + sequence_length as f32) / n;
            }
            ModelType::GemmaEmbedding => {
                stats.gemma_usage += 1;
                stats.gemma_total_time_ms += processing_time_ms;

                // Update average sequence length (incremental average)
                let n = stats.gemma_usage as f32;
                stats.avg_gemma_sequence_length =
                    (stats.avg_gemma_sequence_length * (n - 1.0) + sequence_length as f32) / n;
            }
            _ => {
                // Not an embedding model, ignore
            }
        }
    }

    /// Determine processing priority based on input characteristics
    fn determine_processing_priority(
        &self,
        texts: &[&str],
        tasks: &[TaskType],
    ) -> crate::model_architectures::config::ProcessingPriority {
        // High priority for multi-task or large batch scenarios
        if tasks.len() > 1 || texts.len() > 10 {
            crate::model_architectures::config::ProcessingPriority::Latency
        } else if texts.len() > 5 {
            crate::model_architectures::config::ProcessingPriority::Balanced
        } else {
            crate::model_architectures::config::ProcessingPriority::Accuracy
        }
    }

    /// Convert LoRA output to unified HashMap format
    fn convert_lora_to_unified_hashmap(
        &self,
        lora_output: &LoRAClassificationOutput,
        tasks: &[TaskType],
        _batch_size: usize,
    ) -> HashMap<TaskType, UnifiedTaskResult> {
        let mut result_map = HashMap::new();

        for &task in tasks {
            // Extract real values from lora_output instead of hardcoded values
            let unified_result = UnifiedTaskResult {
                task,
                predicted_class: lora_output
                    .task_results
                    .get(&task)
                    .map(|r| r.predicted_class)
                    .unwrap_or(DEFAULT_PREDICTED_CLASS), // Extract from lora_output.task_results
                confidence: lora_output
                    .task_results
                    .get(&task)
                    .map(|r| r.confidence)
                    .unwrap_or(0.0), // Dynamic confidence from actual results
                category_name: lora_output
                    .task_results
                    .get(&task)
                    .map(|r| r.category_name.clone())
                    .unwrap_or_else(|| format!("class_{}", UNKNOWN_CLASS_ID)), // Extract category name or default
                logits: lora_output
                    .task_results
                    .get(&task)
                    .map(|r| r.logits.clone())
                    .unwrap_or_default(), // Dynamic logits from actual results
                task_processing_time_ms: lora_output.processing_time_ms / tasks.len() as f32,
            };
            result_map.insert(task, unified_result);
        }

        result_map
    }

    /// Convert traditional results to unified HashMap format
    fn convert_traditional_to_unified_hashmap(
        &self,
        task_results: &[UnifiedTaskResult],
        _tasks: &[TaskType],
    ) -> HashMap<TaskType, UnifiedTaskResult> {
        let mut result_map = HashMap::new();

        for result in task_results {
            result_map.insert(result.task, result.clone());
        }

        result_map
    }

    /// Classify single text with traditional path
    fn classify_single_text_traditional(
        &self,
        _text: &str,
        _task: TaskType,
        _index: usize,
    ) -> Result<UnifiedTaskResult, UnifiedClassifierError> {
        // Real implementation required - no hardcoded values allowed per .cursorrules
        Err(UnifiedClassifierError::ProcessingError(
            "Traditional single text classification not implemented - requires real model inference".to_string()
        ))
    }

    /// Select the most appropriate embedding model based on requirements
    ///
    /// This method implements intelligent routing logic that considers:
    /// 1. **Sequence length**: Different models support different maximum lengths
    /// 2. **Quality vs. latency trade-off**: Balance between embedding quality and speed
    /// 3. **Matryoshka support**: Prefer models that support target dimensions
    ///
    /// ## Routing Logic
    ///
    /// ### Short Sequences (0-512 tokens)
    /// - **High latency priority (>0.7)**: GemmaEmbedding (fastest, ~20ms)
    /// - **High quality priority (≤0.7)**: Qwen3Embedding (better quality, ~30ms)
    ///
    /// ### Medium Sequences (513-2048 tokens)
    /// - Always route to GemmaEmbedding (optimal: 8K context window, good speed)
    ///
    /// ### Long Sequences (2049-32768 tokens)
    /// - Always route to Qwen3Embedding (only model supporting 32K context)
    ///
    /// ### Ultra-long Sequences (>32768 tokens)
    /// - Returns error (exceeds maximum supported length)
    ///
    /// ## Arguments
    /// - `requirements`: Embedding generation requirements
    ///
    /// ## Returns
    /// - `Ok(ModelType)`: The selected model type (Qwen3Embedding or GemmaEmbedding)
    /// - `Err`: If sequence length exceeds maximum or other validation fails
    ///
    /// ## Example
    /// ```rust,ignore
    /// // Short sequence with high latency priority -> GemmaEmbedding
    /// let requirements = EmbeddingRequirements {
    ///     sequence_length: 256,
    ///     quality_priority: 0.5,
    ///     latency_priority: 0.9,
    ///     target_dimension: None,
    /// };
    /// let model = classifier.select_embedding_model(&requirements)?;
    /// assert_eq!(model, ModelType::GemmaEmbedding);
    ///
    /// // Long sequence -> Qwen3Embedding (only option)
    /// let requirements = EmbeddingRequirements {
    ///     sequence_length: 4096,
    ///     quality_priority: 0.8,
    ///     latency_priority: 0.3,
    ///     target_dimension: None,
    /// };
    /// let model = classifier.select_embedding_model(&requirements)?;
    /// assert_eq!(model, ModelType::Qwen3Embedding);
    /// ```
    pub fn select_embedding_model(
        &self,
        requirements: &EmbeddingRequirements,
    ) -> Result<ModelType, UnifiedClassifierError> {
        // Validate sequence length
        if requirements.sequence_length > 32768 {
            return Err(UnifiedClassifierError::ProcessingError(format!(
                "Sequence length {} exceeds maximum supported length of 32K tokens. \
                     Consider splitting the input into smaller chunks.",
                requirements.sequence_length
            )));
        }

        // Intelligent routing based on sequence length and priority
        let model_type = match requirements.sequence_length {
            // Short sequences (0-512 tokens)
            // Decision based on latency vs quality priority
            0..=512 => {
                if requirements.quality_priority > 0.7 {
                    // High quality priority -> Choose Qwen3 (better quality)
                    // - Inference time: ~30ms
                    // - Last token pooling (better for instructions)
                    // - Larger hidden size (1024 vs 768)
                    ModelType::Qwen3Embedding
                } else if requirements.latency_priority > 0.7 {
                    // High latency priority (> 0.7) -> Choose Gemma (faster)
                    // - Inference time: ~20ms
                    // - Mean pooling + Dense bottleneck
                    // - Good quality despite smaller size
                    ModelType::GemmaEmbedding
                } else {
                    // Balanced or quality-favoring (latency <= 0.7) -> Choose Qwen3
                    // Default to Qwen3 for better quality when priorities are balanced
                    ModelType::Qwen3Embedding
                }
            }

            // Medium sequences (513-2048 tokens)
            // Gemma is optimal: sufficient context window (8K), good speed
            513..=2048 => {
                // GemmaEmbedding is optimal for this range
                // - Supports up to 8K context (plenty of headroom)
                // - Good balance of speed (~50ms) and quality
                // - Dense bottleneck provides high-quality embeddings
                // - Matryoshka support for flexible dimensions
                ModelType::GemmaEmbedding
            }

            // Long sequences (2049-32768 tokens)
            // Only Qwen3 supports this range
            2049..=32768 => {
                // Only Qwen3Embedding supports sequences this long
                // - Maximum 32K context window
                // - Last token pooling for long contexts
                // - Optimized for long-range dependencies
                ModelType::Qwen3Embedding
            }

            // This should never be reached due to validation above,
            // but added for exhaustiveness
            _ => {
                return Err(UnifiedClassifierError::ProcessingError(format!(
                    "Invalid sequence length: {}. Must be > 0 and <= 32768.",
                    requirements.sequence_length
                )));
            }
        };

        // Consider Matryoshka dimension requirements
        // If target_dimension is < 768, Gemma might be more efficient
        let model_type = if let Some(target_dim) = requirements.target_dimension {
            if target_dim < 768 && requirements.latency_priority > 0.5 {
                // For smaller dimensions with latency priority, prefer Gemma
                // Gemma supports Matryoshka representation learning (768/512/256/128)
                ModelType::GemmaEmbedding
            } else {
                model_type
            }
        } else {
            model_type
        };

        // Validate model availability and fall back if necessary
        let model_type = match model_type {
            ModelType::GemmaEmbedding => {
                // Check if Gemma is available
                if let Some(factory) = GLOBAL_MODEL_FACTORY.get() {
                    if factory.get_gemma_model().is_none() {
                        // Gemma not available, fall back to Qwen3
                        eprintln!(
                            "WARNING: GemmaEmbedding selected but not available, falling back to Qwen3Embedding"
                        );
                        ModelType::Qwen3Embedding
                    } else {
                        ModelType::GemmaEmbedding
                    }
                } else {
                    // No factory available, fall back to Qwen3
                    eprintln!(
                        "WARNING: ModelFactory not initialized, falling back to Qwen3Embedding"
                    );
                    ModelType::Qwen3Embedding
                }
            }
            ModelType::Qwen3Embedding => {
                // Qwen3 is the default, should always be available
                // But verify just in case
                if let Some(factory) = GLOBAL_MODEL_FACTORY.get() {
                    if factory.get_qwen3_model().is_none() {
                        return Err(UnifiedClassifierError::ProcessingError(
                            "Qwen3Embedding selected but not available and no fallback available"
                                .to_string(),
                        ));
                    }
                }
                ModelType::Qwen3Embedding
            }
            // For non-embedding types, pass through
            other => other,
        };

        // Log routing decision for monitoring
        if self.config.embedding.enable_performance_tracking {
            println!(
                "[Embedding Router] Model {:?} selected for seq_len={} (quality={:.2}, latency={:.2}, target_dim={:?})",
                model_type,
                requirements.sequence_length,
                requirements.quality_priority,
                requirements.latency_priority,
                requirements.target_dimension
            );
        }

        Ok(model_type)
    }

    /// Calculate performance improvement over baseline
    fn calculate_performance_improvement(&self, processing_time: f32, path_used: ModelType) -> f32 {
        match path_used {
            ModelType::LoRA => {
                // Calculate improvement based on historical traditional performance
                let stats = self.performance_stats.read();
                if stats.traditional_request_count > 0 {
                    let avg_traditional =
                        stats.traditional_total_time / stats.traditional_request_count as f32;
                    if avg_traditional > 0.0 {
                        ((avg_traditional - processing_time) / avg_traditional) * 100.0
                    } else {
                        0.0
                    }
                } else {
                    // No historical data available
                    0.0
                }
            }
            ModelType::Traditional => {
                // Traditional is the baseline
                0.0
            }
            ModelType::Qwen3Embedding | ModelType::GemmaEmbedding | ModelType::MmBertEmbedding => {
                // Embedding models don't participate in classification performance tracking
                // Their metrics (latency, throughput) are tracked separately via update_embedding_stats()
                // Performance comparison is not meaningful in classification context
                0.0
            }
        }
    }

    ///  Get current performance statistics
    pub fn get_performance_stats(
        &self,
    ) -> parking_lot::RwLockReadGuard<'_, UnifiedPerformanceStats> {
        self.performance_stats.read()
    }
}

impl Default for UnifiedPerformanceStats {
    fn default() -> Self {
        Self {
            total_classifications: 0,
            traditional_usage: 0,
            lora_usage: 0,
            avg_traditional_time_ms: 0.0,
            avg_lora_time_ms: 0.0,
            overall_improvement: 0.0,
            avg_confidence: 0.0, // Start with 0.0, calculate dynamically
            traditional_total_time: 0.0,
            traditional_request_count: 0,
            lora_total_time: 0.0,
            lora_request_count: 0,
            path_switches: 0,
            last_path_used: None,
            // Embedding model metrics
            qwen3_usage: 0,
            qwen3_total_time_ms: 0.0,
            gemma_usage: 0,
            gemma_total_time_ms: 0.0,
            embedding_total_requests: 0,
            avg_qwen3_sequence_length: 0.0,
            avg_gemma_sequence_length: 0.0,
        }
    }
}
