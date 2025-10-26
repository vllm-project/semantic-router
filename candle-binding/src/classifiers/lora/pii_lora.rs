//! PII detection with LoRA adapters
//!
//! High-performance PII detection using real token classification model inference

use crate::core::{ModelErrorType, UnifiedError};
use crate::model_architectures::lora::bert_lora::HighPerformanceBertTokenClassifier;
use crate::model_error;
use candle_core::Result;
use std::time::Instant;

/// PII detector with real token classification model inference (merged LoRA models)
pub struct PIILoRAClassifier {
    /// High-performance BERT token classifier for PII detection
    bert_token_classifier: HighPerformanceBertTokenClassifier,
    /// Confidence threshold for PII detection
    confidence_threshold: f32,
    /// PII type labels
    pii_types: Vec<String>,
    /// Model path for reference
    model_path: String,
}

/// Individual PII occurrence with its own confidence
#[derive(Debug, Clone)]
pub struct PIIOccurrence {
    pub pii_type: String,
    pub confidence: f32,
    pub token: String,
    pub start_pos: usize,
    pub end_pos: usize,
}

/// PII detection result with individual occurrence confidences
#[derive(Debug, Clone)]
pub struct PIIResult {
    pub has_pii: bool,
    pub pii_types: Vec<String>,          // Keep for backward compatibility
    pub confidence: f32,                 // Overall confidence (average or max)
    pub occurrences: Vec<PIIOccurrence>, // Individual occurrences with their own confidence
    pub processing_time_ms: u64,
}

impl PIILoRAClassifier {
    /// Create new PII detector using real token classification model inference
    pub fn new(model_path: &str, use_cpu: bool) -> Result<Self> {
        // Load labels from model config
        let pii_types = Self::load_labels_from_config(model_path)?;
        let num_classes = pii_types.len();

        // Create high-performance BERT token classifier for PII detection
        let bert_token_classifier =
            HighPerformanceBertTokenClassifier::new(model_path, num_classes, use_cpu).map_err(
                |e| {
                    let unified_err = model_error!(
                        ModelErrorType::LoRA,
                        "PII token classifier creation",
                        format!("Failed to create BERT token classifier: {}", e),
                        model_path
                    );
                    candle_core::Error::from(unified_err)
                },
            )?;

        Ok(Self {
            bert_token_classifier,
            confidence_threshold: 0.5,
            pii_types,
            model_path: model_path.to_string(),
        })
    }

    /// Load PII labels from model config.json using unified config loader
    fn load_labels_from_config(model_path: &str) -> Result<Vec<String>> {
        use crate::core::config_loader;

        match config_loader::load_pii_labels(model_path) {
            Ok(result) => Ok(result),
            Err(unified_err) => Err(candle_core::Error::from(unified_err)),
        }
    }

    /// Detect PII using real token classification model inference
    pub fn detect_pii(&self, text: &str) -> Result<PIIResult> {
        let start_time = Instant::now();

        // Use real BERT token classifier for PII detection
        let token_results = self
            .bert_token_classifier
            .classify_tokens(text)
            .map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "PII token classification",
                    format!("PII token classification failed: {}", e),
                    text
                );
                candle_core::Error::from(unified_err)
            })?;

        // Create individual occurrences with their own confidence scores
        let mut occurrences = Vec::new();
        let mut detected_types = Vec::new();
        let mut confidence_scores = Vec::new();
        let mut has_pii = false;

        // Calculate confidence for "O" class for non-PII tokens
        let o_confidences: Vec<f32> = token_results
            .iter()
            .filter(|(_, class_idx, _)| *class_idx == 0) // "O" class
            .map(|(_, _, confidence)| *confidence)
            .collect();
        let avg_o_confidence = if o_confidences.is_empty() {
            0.0
        } else {
            o_confidences.iter().sum::<f32>() / o_confidences.len() as f32
        };

        // Process each token with its individual confidence
        for (i, (token, class_idx, confidence)) in token_results.iter().enumerate() {
            // Skip "O" (Outside) labels - class 0 typically means no PII
            if *class_idx > 0 && *class_idx < self.pii_types.len() {
                has_pii = true;
                confidence_scores.push(*confidence);

                let pii_type = &self.pii_types[*class_idx];
                if !detected_types.contains(pii_type) {
                    detected_types.push(pii_type.clone());
                }

                // Create individual occurrence with its own confidence
                occurrences.push(PIIOccurrence {
                    pii_type: pii_type.clone(),
                    confidence: *confidence, // Each occurrence keeps its individual confidence
                    token: token.clone(),
                    start_pos: i, // Token position in sequence
                    end_pos: i + 1,
                });
            }
        }

        // Calculate overall confidence without inflating individual confidences
        let final_confidence = if has_pii {
            // Use average confidence instead of max to avoid inflating significance
            confidence_scores.iter().sum::<f32>() / confidence_scores.len() as f32
        } else {
            // For no PII detected, use the confidence of the "O" (Outside) class
            avg_o_confidence
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(PIIResult {
            has_pii,
            pii_types: detected_types,
            confidence: final_confidence,
            occurrences, // Include individual occurrences with their own confidences
            processing_time_ms: processing_time,
        })
    }

    /// Parallel PII detection for multiple texts using rayon
    ///
    /// # Performance
    /// - Uses rayon for parallel processing across available CPU cores
    /// - Efficient for batch sizes > 10
    /// - No lock contention during inference
    pub fn parallel_detect(&self, texts: &[&str]) -> Result<Vec<PIIResult>> {
        use rayon::prelude::*;

        texts
            .par_iter()
            .map(|text| self.detect_pii(text))
            .collect::<Result<Vec<_>>>()
    }

    /// Batch PII detection for multiple texts
    pub fn batch_detect(&self, texts: &[&str]) -> Result<Vec<PIIResult>> {
        self.parallel_detect(texts)
    }
}
