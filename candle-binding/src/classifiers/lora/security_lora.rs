//! Security detection with LoRA adapters
//!
//! High-performance security threat detection using real model inference

use crate::core::{processing_errors, ModelErrorType, UnifiedError};
use crate::model_architectures::lora::bert_lora::HighPerformanceBertClassifier;
use crate::model_error;
use candle_core::Result;
use std::time::Instant;

/// Security detector with real model inference (merged LoRA models)
pub struct SecurityLoRAClassifier {
    /// High-performance BERT classifier for security detection
    bert_classifier: HighPerformanceBertClassifier,
    /// Confidence threshold for threat detection
    confidence_threshold: f32,
    /// Threat type labels
    threat_types: Vec<String>,
    /// Model path for reference
    model_path: String,
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
    pub fn new(model_path: &str, use_cpu: bool) -> Result<Self> {
        // Load labels from model config
        let threat_types = Self::load_labels_from_config(model_path)?;
        let num_classes = threat_types.len();

        // Create high-performance BERT classifier for security detection
        let bert_classifier = HighPerformanceBertClassifier::new(model_path, num_classes, use_cpu)
            .map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "security classifier creation",
                    format!("Failed to create BERT classifier: {}", e),
                    model_path
                );
                candle_core::Error::from(unified_err)
            })?;

        Ok(Self {
            bert_classifier,
            confidence_threshold: 0.5,
            threat_types,
            model_path: model_path.to_string(),
        })
    }

    /// Load threat labels from model config.json using unified config loader
    fn load_labels_from_config(model_path: &str) -> Result<Vec<String>> {
        use crate::core::config_loader;

        match config_loader::load_security_labels(model_path) {
            Ok(result) => Ok(result),
            Err(unified_err) => Err(candle_core::Error::from(unified_err)),
        }
    }

    /// Detect security threats using real model inference
    pub fn detect_threats(&self, text: &str) -> Result<SecurityResult> {
        let start_time = Instant::now();

        // Use real BERT model for security detection
        let (predicted_class, confidence) =
            self.bert_classifier.classify_text(text).map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "security detection",
                    format!("Security detection failed: {}", e),
                    text
                );
                candle_core::Error::from(unified_err)
            })?;

        // Determine if threat is detected based on predicted class
        let is_threat = predicted_class > 0; // Assuming class 0 is "benign" or "safe"

        // Get detected threat types
        let mut detected_threats = Vec::new();
        if is_threat && predicted_class < self.threat_types.len() {
            detected_threats.push(self.threat_types[predicted_class].clone());
        }

        // Calculate severity score based on confidence and threat type
        let severity_score = if is_threat {
            confidence * 0.9 // High severity for detected threats
        } else {
            0.0 // No severity for safe content
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(SecurityResult {
            is_threat,
            threat_types: detected_threats,
            severity_score,
            confidence,
            processing_time_ms: processing_time,
        })
    }

    /// Parallel security detection for multiple texts
    pub fn parallel_detect(&self, texts: &[&str]) -> Result<Vec<SecurityResult>> {
        // Process each text using real model inference
        texts.iter().map(|text| self.detect_threats(text)).collect()
    }

    /// Batch security detection for multiple texts (optimized)
    pub fn batch_detect(&self, texts: &[&str]) -> Result<Vec<SecurityResult>> {
        let start_time = Instant::now();

        // Use BERT's batch processing capability
        let batch_results = self.bert_classifier.classify_batch(texts).map_err(|e| {
            let unified_err = processing_errors::batch_processing(texts.len(), &e.to_string());
            candle_core::Error::from(unified_err)
        })?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        let mut results = Vec::new();
        for (predicted_class, confidence) in batch_results {
            // Determine if threat is detected
            let is_threat = predicted_class > 0; // Assuming class 0 is "benign"

            // Get detected threat types
            let mut detected_threats = Vec::new();
            if is_threat && predicted_class < self.threat_types.len() {
                detected_threats.push(self.threat_types[predicted_class].clone());
            }

            // Calculate severity score
            let severity_score = if is_threat { confidence * 0.9 } else { 0.0 };

            results.push(SecurityResult {
                is_threat,
                threat_types: detected_threats,
                severity_score,
                confidence,
                processing_time_ms: processing_time,
            });
        }

        Ok(results)
    }
}
