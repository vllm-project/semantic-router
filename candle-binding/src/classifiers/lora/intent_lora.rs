//! Intent classification with LoRA adapters
//!
//! High-performance intent classification using real model inference
//! Supports both BERT and ModernBERT/mmBERT models

use crate::core::{processing_errors, ModelErrorType, UnifiedError};
use crate::model_architectures::lora::bert_lora::HighPerformanceBertClassifier;
use crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier;
use crate::model_error;
use candle_core::Result;
use std::path::Path;
use std::time::Instant;

/// Classifier backend enum to avoid Box<dyn Trait>
enum ClassifierBackend {
    Bert(HighPerformanceBertClassifier),
    ModernBert(TraditionalModernBertClassifier),
}

/// Intent classifier with real model inference (merged LoRA models)
/// Supports both BERT and ModernBERT/mmBERT architectures
pub struct IntentLoRAClassifier {
    /// Classifier backend (either BERT or ModernBERT/mmBERT)
    backend: ClassifierBackend,
    /// Confidence threshold for predictions
    confidence_threshold: f32,
    /// Intent labels mapping
    intent_labels: Vec<String>,
    /// Model path for reference
    model_path: String,
}

/// Intent classification result
#[derive(Debug, Clone)]
pub struct IntentResult {
    pub intent: String,
    pub confidence: f32,
    pub processing_time_ms: u64,
}

impl IntentLoRAClassifier {
    /// Create new intent classifier using real model inference
    /// Automatically detects if model is BERT or ModernBERT/mmBERT
    pub fn new(model_path: &str, use_cpu: bool) -> Result<Self> {
        // Load labels from model config
        let intent_labels = Self::load_labels_from_config(model_path)?;
        let num_classes = intent_labels.len();

        // Detect model type and create appropriate backend
        let backend = if Self::is_modernbert(model_path) {
            // Use existing TraditionalModernBertClassifier (supports both ModernBERT and mmBERT)
            let classifier = TraditionalModernBertClassifier::load_from_directory(
                model_path, use_cpu,
            )
            .map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "intent classifier creation",
                    format!("Failed to create ModernBERT/mmBERT classifier: {}", e),
                    model_path
                );
                candle_core::Error::from(unified_err)
            })?;
            ClassifierBackend::ModernBert(classifier)
        } else {
            // Use standard BERT classifier
            let classifier = HighPerformanceBertClassifier::new(model_path, num_classes, use_cpu)
                .map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "intent classifier creation",
                    format!("Failed to create BERT classifier: {}", e),
                    model_path
                );
                candle_core::Error::from(unified_err)
            })?;
            ClassifierBackend::Bert(classifier)
        };

        // Load threshold from global config
        let confidence_threshold = {
            use crate::core::config_loader::GlobalConfigLoader;
            GlobalConfigLoader::load_intent_threshold().unwrap_or(0.6)
        };

        Ok(Self {
            backend,
            confidence_threshold,
            intent_labels,
            model_path: model_path.to_string(),
        })
    }

    /// Check if model is ModernBERT/mmBERT architecture
    fn is_modernbert(model_path: &str) -> bool {
        let config_path = Path::new(model_path).join("config.json");
        if let Ok(config_str) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
                let model_type = config
                    .get("model_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                return model_type == "modernbert";
            }
        }
        false
    }

    /// Check if this classifier is using an mmBERT (multilingual) model
    pub fn is_multilingual(&self) -> bool {
        matches!(&self.backend, ClassifierBackend::ModernBert(c) if c.is_multilingual())
    }

    /// Classify using the appropriate backend
    fn classify_with_backend(&self, text: &str) -> Result<(usize, f32)> {
        match &self.backend {
            ClassifierBackend::Bert(c) => c
                .classify_text(text)
                .map_err(|e| candle_core::Error::Msg(format!("BERT classification failed: {}", e))),
            ClassifierBackend::ModernBert(c) => c.classify_text(text),
        }
    }

    /// Load intent labels from model config.json using unified config loader
    fn load_labels_from_config(model_path: &str) -> Result<Vec<String>> {
        use crate::core::config_loader;

        match config_loader::load_intent_labels(model_path) {
            Ok(result) => Ok(result),
            Err(unified_err) => Err(candle_core::Error::from(unified_err)),
        }
    }

    /// Classify intent using real model inference
    pub fn classify_intent(&self, text: &str) -> Result<IntentResult> {
        let start_time = Instant::now();

        // Use appropriate backend (BERT or ModernBERT/mmBERT) for classification
        let (predicted_class, confidence) = self.classify_with_backend(text).map_err(|e| {
            let unified_err = model_error!(
                ModelErrorType::LoRA,
                "intent classification",
                format!("Classification failed: {}", e),
                text
            );
            candle_core::Error::from(unified_err)
        })?;

        // Map class index to intent label - fail if class not found
        let intent = if predicted_class < self.intent_labels.len() {
            self.intent_labels[predicted_class].clone()
        } else {
            let unified_err = model_error!(
                ModelErrorType::LoRA,
                "intent classification",
                format!(
                    "Invalid class index {} not found in labels (max: {})",
                    predicted_class,
                    self.intent_labels.len()
                ),
                text
            );
            return Err(candle_core::Error::from(unified_err));
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(IntentResult {
            intent,
            confidence,
            processing_time_ms: processing_time,
        })
    }

    /// Classify intent and return (class_index, confidence, intent_label) for FFI
    pub fn classify_with_index(&self, text: &str) -> Result<(usize, f32, String)> {
        // Use appropriate backend (BERT or ModernBERT/mmBERT) for classification
        let (predicted_class, confidence) = self.classify_with_backend(text).map_err(|e| {
            let unified_err = model_error!(
                ModelErrorType::LoRA,
                "intent classification",
                format!("Classification failed: {}", e),
                text
            );
            candle_core::Error::from(unified_err)
        })?;

        // Map class index to intent label - fail if class not found
        let intent = if predicted_class < self.intent_labels.len() {
            self.intent_labels[predicted_class].clone()
        } else {
            let unified_err = model_error!(
                ModelErrorType::LoRA,
                "intent classification",
                format!(
                    "Invalid class index {} not found in labels (max: {})",
                    predicted_class,
                    self.intent_labels.len()
                ),
                text
            );
            return Err(candle_core::Error::from(unified_err));
        };

        Ok((predicted_class, confidence, intent))
    }

    /// Parallel classification for multiple texts using rayon
    ///
    /// # Performance
    /// - Uses rayon for parallel processing across available CPU cores
    /// - Efficient for batch sizes > 10
    /// - No lock contention during inference
    pub fn parallel_classify(&self, texts: &[&str]) -> Result<Vec<IntentResult>> {
        use rayon::prelude::*;

        // Process each text using real model inference in parallel
        texts
            .par_iter()
            .map(|text| self.classify_intent(text))
            .collect()
    }

    /// Batch classification for multiple texts (optimized)
    pub fn batch_classify(&self, texts: &[&str]) -> Result<Vec<IntentResult>> {
        let start_time = Instant::now();

        // For batch, use parallel classify (TraditionalModernBertClassifier doesn't expose batch)
        let batch_results: Vec<(usize, f32)> = texts
            .iter()
            .map(|text| self.classify_with_backend(text))
            .collect::<Result<Vec<_>>>()
            .map_err(|e: candle_core::Error| {
                let unified_err = processing_errors::batch_processing(texts.len(), &e.to_string());
                candle_core::Error::from(unified_err)
            })?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        let mut results = Vec::new();
        for (i, (predicted_class, confidence)) in batch_results.iter().enumerate() {
            let intent = if *predicted_class < self.intent_labels.len() {
                self.intent_labels[*predicted_class].clone()
            } else {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "batch intent classification",
                    format!("Invalid class index {} not found in labels (max: {}) for text at position {}",
                           predicted_class, self.intent_labels.len(), i),
                    &format!("batch[{}]", i)
                );
                return Err(candle_core::Error::from(unified_err));
            };

            results.push(IntentResult {
                intent,
                confidence: *confidence,
                processing_time_ms: processing_time,
            });
        }

        Ok(results)
    }
}
