//! Parallel LoRA processing engine
//!
//! Enables parallel execution of Intent||PII||Security classification tasks
//! Using thread-based parallelism instead of async/await

use crate::classifiers::lora::{
    intent_lora::{IntentLoRAClassifier, IntentResult},
    pii_lora::{PIILoRAClassifier, PIIResult},
    security_lora::{SecurityLoRAClassifier, SecurityResult},
};
use crate::core::{concurrency_error, ModelErrorType, UnifiedError};
use crate::model_error;
use candle_core::{Device, Result};
use std::sync::{Arc, Mutex};
use std::thread;

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

    /// Parallel classification across all three tasks
    pub fn parallel_classify(&self, texts: &[&str]) -> Result<ParallelResult> {
        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        // Create shared results
        let intent_results = Arc::new(Mutex::new(Vec::new()));
        let pii_results = Arc::new(Mutex::new(Vec::new()));
        let security_results = Arc::new(Mutex::new(Vec::new()));

        let handles = vec![
            self.spawn_intent_task(texts_owned.clone(), Arc::clone(&intent_results)),
            self.spawn_pii_task(texts_owned.clone(), Arc::clone(&pii_results)),
            self.spawn_security_task(texts_owned, Arc::clone(&security_results)),
        ];

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|_| {
                let unified_err = concurrency_error(
                    "thread join",
                    "Failed to join parallel classification thread",
                );
                candle_core::Error::from(unified_err)
            })?;
        }

        Ok(ParallelResult {
            intent_results: Arc::try_unwrap(intent_results)
                .unwrap()
                .into_inner()
                .unwrap(),
            pii_results: Arc::try_unwrap(pii_results).unwrap().into_inner().unwrap(),
            security_results: Arc::try_unwrap(security_results)
                .unwrap()
                .into_inner()
                .unwrap(),
        })
    }

    fn spawn_intent_task(
        &self,
        texts: Vec<String>,
        results: Arc<Mutex<Vec<IntentResult>>>,
    ) -> thread::JoinHandle<()> {
        let classifier = Arc::clone(&self.intent_classifier);
        thread::spawn(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            match classifier.batch_classify(&text_refs) {
                Ok(task_results) => {
                    let mut guard = results.lock().unwrap();
                    *guard = task_results;
                }
                Err(e) => {
                    eprintln!("Intent classification failed: {}", e);
                }
            }
        })
    }

    fn spawn_pii_task(
        &self,
        texts: Vec<String>,
        results: Arc<Mutex<Vec<PIIResult>>>,
    ) -> thread::JoinHandle<()> {
        let classifier = Arc::clone(&self.pii_classifier);
        thread::spawn(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            if let Ok(task_results) = classifier.batch_detect(&text_refs) {
                let mut guard = results.lock().unwrap();
                *guard = task_results;
            }
        })
    }

    fn spawn_security_task(
        &self,
        texts: Vec<String>,
        results: Arc<Mutex<Vec<SecurityResult>>>,
    ) -> thread::JoinHandle<()> {
        let classifier = Arc::clone(&self.security_classifier);
        thread::spawn(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            match classifier.batch_detect(&text_refs) {
                Ok(task_results) => {
                    let mut guard = results.lock().unwrap();
                    *guard = task_results;
                }
                Err(e) => {
                    eprintln!("Security classification failed: {}", e);
                }
            }
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
