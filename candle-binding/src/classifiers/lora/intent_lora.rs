//! Intent classification with LoRA adapters
//!
//! High-performance intent classification using real model inference

use crate::core::{processing_errors, ModelErrorType, UnifiedError};
use crate::model_architectures::lora::bert_lora::HighPerformanceBertClassifier;
use crate::model_error;
use candle_core::Result;
use std::time::Instant;

/// Intent classifier with real model inference (merged LoRA models)
pub struct IntentLoRAClassifier {
    /// High-performance BERT classifier for intent classification
    bert_classifier: HighPerformanceBertClassifier,
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
    pub fn new(model_path: &str, use_cpu: bool) -> Result<Self> {
        // Load labels from model config
        let intent_labels = Self::load_labels_from_config(model_path)?;
        let num_classes = intent_labels.len();

        // Load the high-performance BERT classifier for merged LoRA models
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

        // Load threshold from global config instead of hardcoding
        let confidence_threshold = {
            use crate::core::config_loader::GlobalConfigLoader;
            GlobalConfigLoader::load_intent_threshold().unwrap_or(0.6) // Default from config.yaml classifier.category_model.threshold
        };

        Ok(Self {
            bert_classifier: classifier,
            confidence_threshold,
            intent_labels,
            model_path: model_path.to_string(),
        })
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

        // Use real BERT model for classification
        let (predicted_class, confidence) =
            self.bert_classifier.classify_text(text).map_err(|e| {
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
        // Use real BERT model for classification
        let (predicted_class, confidence) =
            self.bert_classifier.classify_text(text).map_err(|e| {
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

        // Use BERT's batch processing capability
        let batch_results = self.bert_classifier.classify_batch(texts).map_err(|e| {
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
