//! Parallel LoRA processing engine
//!
//! Enables parallel execution of Intent||PII||Security classification tasks
//! Using rayon for efficient data parallelism

use crate::classifiers::lora::{
    intent_lora::{IntentLoRAClassifier, IntentResult},
    pii_lora::{PIILoRAClassifier, PIIResult},
    security_lora::{SecurityLoRAClassifier, SecurityResult},
};
use crate::core::{ModelErrorType, UnifiedError};
use crate::model_error;
use candle_core::{Device, Result};
use std::sync::Arc;

/// Parallel LoRA processing engine
pub struct ParallelLoRAEngine {
    intent_classifier: Arc<IntentLoRAClassifier>,
    pii_classifier: Arc<PIILoRAClassifier>,
    security_classifier: Arc<SecurityLoRAClassifier>,
    device: Device,
}

impl ParallelLoRAEngine {
    pub fn new(
        device: Device,
        intent_model_path: &str,
        pii_model_path: &str,
        security_model_path: &str,
        use_cpu: bool,
    ) -> Result<Self> {
        // Create intent classifier
        let intent_classifier = Arc::new(
            IntentLoRAClassifier::new(intent_model_path, use_cpu).map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "intent classifier creation",
                    format!("Failed to create intent classifier: {}", e),
                    intent_model_path
                );
                candle_core::Error::from(unified_err)
            })?,
        );

        // Create PII classifier
        let pii_classifier = Arc::new(PIILoRAClassifier::new(pii_model_path, use_cpu).map_err(
            |e| {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "PII classifier creation",
                    format!("Failed to create PII classifier: {}", e),
                    pii_model_path
                );
                candle_core::Error::from(unified_err)
            },
        )?);

        // Create security classifier
        let security_classifier = Arc::new(
            SecurityLoRAClassifier::new(security_model_path, use_cpu).map_err(|e| {
                let unified_err = model_error!(
                    ModelErrorType::LoRA,
                    "security classifier creation",
                    format!("Failed to create security classifier: {}", e),
                    security_model_path
                );
                candle_core::Error::from(unified_err)
            })?,
        );

        Ok(Self {
            intent_classifier,
            pii_classifier,
            security_classifier,
            device,
        })
    }

    /// Parallel classification across all three tasks using rayon
    ///
    /// # Performance
    /// - Uses rayon::join for parallel execution (no Arc<Mutex> overhead)
    /// - Simplified code: ~70 lines reduced to ~20 lines
    /// - No lock contention or synchronization overhead
    pub fn parallel_classify(&self, texts: &[&str]) -> Result<ParallelResult> {
        // Execute all three classifiers in parallel using rayon::join
        // Each task runs independently without shared mutable state
        let ((intent_results, pii_results), security_results) = rayon::join(
            || {
                rayon::join(
                    || self.intent_classifier.batch_classify(texts),
                    || self.pii_classifier.batch_detect(texts),
                )
            },
            || self.security_classifier.batch_detect(texts),
        );

        // Propagate errors from any task
        Ok(ParallelResult {
            intent_results: intent_results?,
            pii_results: pii_results?,
            security_results: security_results?,
        })
    }
}

/// Results from parallel classification
#[derive(Debug, Clone)]
pub struct ParallelResult {
    pub intent_results: Vec<IntentResult>,
    pub pii_results: Vec<PIIResult>,
    pub security_results: Vec<SecurityResult>,
}
