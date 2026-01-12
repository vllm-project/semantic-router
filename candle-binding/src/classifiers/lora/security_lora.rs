//! Security detection with LoRA adapters
//!
//! High-performance security threat detection using real model inference
//! Supports both BERT and ModernBERT/mmBERT models

use crate::core::{processing_errors, ModelErrorType, UnifiedError};
use crate::model_architectures::lora::bert_lora::{
    HighPerformanceBertClassifier, HighPerformanceModernBertClassifier,
    is_mmbert_model, TextClassifier,
};
use crate::model_error;
use candle_core::Result;
use std::time::Instant;

/// Security detector with real model inference (merged LoRA models)
/// Supports both BERT and ModernBERT/mmBERT architectures
pub struct SecurityLoRAClassifier {
    /// High-performance classifier (either BERT or ModernBERT/mmBERT)
    classifier: Box<dyn TextClassifier>,
    /// Confidence threshold for threat detection
    confidence_threshold: f32,
    /// Threat type labels
    threat_types: Vec<String>,
    /// Model path for reference
    model_path: String,
    /// Whether this is an mmBERT (multilingual) model
    is_multilingual: bool,
}

/// Security detection result
#[derive(Debug, Clone)]
pub struct SecurityResult {
    pub is_threat: bool,
    pub threat_types: Vec<String>,
    pub severity_score: f32,
    pub confidence: f32,
    pub processing_time_ms: u64,
}

impl SecurityLoRAClassifier {
    /// Create new security detector using real model inference
    /// Automatically detects if model is BERT or ModernBERT/mmBERT
    pub fn new(model_path: &str, use_cpu: bool) -> Result<Self> {
        // Load labels from model config
        let threat_types = Self::load_labels_from_config(model_path)?;
        let num_classes = threat_types.len();

        // Detect model type and create appropriate classifier
        let is_multilingual = is_mmbert_model(model_path);
        let classifier: Box<dyn TextClassifier> = if is_multilingual || Self::is_modernbert(model_path) {
            // Use ModernBERT/mmBERT classifier
            Box::new(HighPerformanceModernBertClassifier::new(model_path, num_classes, use_cpu)
                .map_err(|e| {
                    let unified_err = model_error!(
                        ModelErrorType::LoRA,
                        "security classifier creation",
                        format!("Failed to create ModernBERT/mmBERT classifier: {}", e),
                        model_path
                    );
                    candle_core::Error::from(unified_err)
                })?)
        } else {
            // Use standard BERT classifier
            Box::new(HighPerformanceBertClassifier::new(model_path, num_classes, use_cpu)
                .map_err(|e| {
                    let unified_err = model_error!(
                        ModelErrorType::LoRA,
                        "security classifier creation",
                        format!("Failed to create BERT classifier: {}", e),
                        model_path
                    );
                    candle_core::Error::from(unified_err)
                })?)
        };

        // Load threshold from global config instead of hardcoding
        let confidence_threshold = {
            use crate::core::config_loader::GlobalConfigLoader;
            GlobalConfigLoader::load_security_threshold().unwrap_or(0.7)
        };

        Ok(Self {
            classifier,
            confidence_threshold,
            threat_types,
            model_path: model_path.to_string(),
            is_multilingual,
        })
    }

    /// Check if model is ModernBERT architecture
    fn is_modernbert(model_path: &str) -> bool {
        use std::path::Path;
        let config_path = Path::new(model_path).join("config.json");
        if let Ok(config_str) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
                let model_type = config.get("model_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                return model_type == "modernbert";
            }
        }
        false
    }

    /// Check if this classifier is using an mmBERT (multilingual) model
    pub fn is_multilingual(&self) -> bool {
        self.is_multilingual
    }

    /// Load threat labels from model config.json using unified config loader
    fn load_labels_from_config(model_path: &str) -> Result<Vec<String>> {
        use crate::core::config_loader;

        match config_loader::load_security_labels(model_path) {
            Ok(result) => Ok(result),
            Err(unified_err) => Err(candle_core::Error::from(unified_err)),
        }
    }

    /// Classify text and return (class_index, confidence, label) for FFI compatibility
    /// This is a simpler interface for jailbreak detection that matches the intent classifier pattern
    pub fn classify_with_index(&self, text: &str) -> Result<(usize, f32, String)> {
        // Use appropriate model (BERT or ModernBERT/mmBERT) for classification
        let (predicted_class, confidence) =
            self.classifier.classify_text(text).map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "jailbreak classification",
                    format!("Classification failed: {}", e),
                    text
                );
                candle_core::Error::from(unified_err)
            })?;

        // Map class index to label - fail if class not found
        let label = if predicted_class < self.threat_types.len() {
            self.threat_types[predicted_class].clone()
        } else {
            let unified_err = model_error!(
                ModelErrorType::LoRA,
                "jailbreak classification",
                format!(
                    "Invalid class index {} not found in labels (max: {})",
                    predicted_class,
                    self.threat_types.len()
                ),
                text
            );
            return Err(candle_core::Error::from(unified_err));
        };

        Ok((predicted_class, confidence, label))
    }

    /// Detect security threats using real model inference
    pub fn detect_threats(&self, text: &str) -> Result<SecurityResult> {
        let start_time = Instant::now();

        // Use appropriate model (BERT or ModernBERT/mmBERT) for security detection
        let (predicted_class, confidence) =
            self.classifier.classify_text(text).map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "security detection",
                    format!("Security detection failed: {}", e),
                    text
                );
                candle_core::Error::from(unified_err)
            })?;

        // Map class index to threat type label - fail if class not found
        let threat_type = if predicted_class < self.threat_types.len() {
            self.threat_types[predicted_class].clone()
        } else {
            let unified_err = model_error!(
                ModelErrorType::LoRA,
                "security classification",
                format!(
                    "Invalid class index {} not found in labels (max: {})",
                    predicted_class,
                    self.threat_types.len()
                ),
                text
            );
            return Err(candle_core::Error::from(unified_err));
        };

        // Determine if threat is detected based on class label (instead of hardcoded index)
        let is_threat = !threat_type.to_lowercase().contains("safe")
            && !threat_type.to_lowercase().contains("benign")
            && !threat_type.to_lowercase().contains("no_threat");

        // Get detected threat types
        let detected_threats = if is_threat {
            vec![threat_type]
        } else {
            Vec::new()
        };

        // Use confidence as severity score (no artificial scaling)
        let severity_score = if is_threat { confidence } else { 0.0 };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(SecurityResult {
            is_threat,
            threat_types: detected_threats,
            severity_score,
            confidence,
            processing_time_ms: processing_time,
        })
    }

    /// Parallel security detection for multiple texts using rayon
    ///
    /// # Performance
    /// - Uses rayon for parallel processing across available CPU cores
    /// - Efficient for batch sizes > 10
    /// - No lock contention during inference
    pub fn parallel_detect(&self, texts: &[&str]) -> Result<Vec<SecurityResult>> {
        use rayon::prelude::*;

        // Process each text using real model inference in parallel
        texts
            .par_iter()
            .map(|text| self.detect_threats(text))
            .collect()
    }

    /// Batch security detection for multiple texts (optimized)
    pub fn batch_detect(&self, texts: &[&str]) -> Result<Vec<SecurityResult>> {
        let start_time = Instant::now();

        // Use BERT's batch processing capability
        let batch_results = self.classifier.classify_batch(texts).map_err(|e| {
            let unified_err = processing_errors::batch_processing(texts.len(), &e.to_string());
            candle_core::Error::from(unified_err)
        })?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        let mut results = Vec::new();
        for (i, (predicted_class, confidence)) in batch_results.iter().enumerate() {
            // Map class index to threat type label - fail if class not found
            let threat_type = if *predicted_class < self.threat_types.len() {
                self.threat_types[*predicted_class].clone()
            } else {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "batch security classification",
                    format!("Invalid class index {} not found in labels (max: {}) for text at position {}",
                           predicted_class, self.threat_types.len(), i),
                    &format!("batch[{}]", i)
                );
                return Err(candle_core::Error::from(unified_err));
            };

            // Determine if threat is detected based on class label
            let is_threat = !threat_type.to_lowercase().contains("safe")
                && !threat_type.to_lowercase().contains("benign")
                && !threat_type.to_lowercase().contains("no_threat");

            // Get detected threat types
            let detected_threats = if is_threat {
                vec![threat_type]
            } else {
                Vec::new()
            };

            // Use confidence as severity score (no artificial scaling)
            let severity_score = if is_threat { *confidence } else { 0.0 };

            results.push(SecurityResult {
                is_threat,
                threat_types: detected_threats,
                severity_score,
                confidence: *confidence,
                processing_time_ms: processing_time,
            });
        }

        Ok(results)
    }
}
