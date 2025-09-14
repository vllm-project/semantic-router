// Unified Classifier for Batch Inference Support
// This module implements a unified classification system that:
// 1. Uses a single shared ModernBERT encoder for all tasks
// 2. Supports true batch inference (multiple texts in one forward pass)
// 3. Provides multiple task heads (intent, PII, security) with shared backbone
// 4. Eliminates memory waste from multiple model instances

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{Linear, Module};
use candle_transformers::models::modernbert::{Config, ModernBert};
use serde_json;
use tokenizers::{Encoding, PaddingParams, PaddingStrategy, Tokenizer};

// Import our high-confidence LoRA classifiers
use crate::bert_official::{CandleBertClassifier, CandleBertTokenClassifier};

/// Unified classification result for a single text
#[derive(Debug, Clone)]
pub struct UnifiedClassificationResult {
    pub intent_result: IntentResult,
    pub pii_result: PIIResult,
    pub security_result: SecurityResult,
}

/// Intent classification result
#[derive(Debug, Clone)]
pub struct IntentResult {
    pub category: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
}

/// PII detection result
#[derive(Debug, Clone)]
pub struct PIIResult {
    pub has_pii: bool,
    pub pii_types: Vec<String>,
    pub confidence: f32,
    pub entities: Vec<String>, // Added for batch processing
}

/// Security detection result
#[derive(Debug, Clone)]
pub struct SecurityResult {
    pub is_jailbreak: bool,
    pub threat_type: String,
    pub confidence: f32,
}

/// Batch classification results
#[derive(Debug)]
pub struct BatchClassificationResult {
    pub intent_results: Vec<IntentResult>,
    pub pii_results: Vec<PIIResult>,
    pub security_results: Vec<SecurityResult>,
    pub batch_size: usize,
}

/// Unified classifier with shared ModernBERT backbone and multiple task heads
pub struct UnifiedClassifier {
    // Multi-architecture support for high-confidence LoRA models
    #[allow(dead_code)]
    architecture: String, // "bert", "roberta", or "modernbert"
    device: Device,

    // High-confidence LoRA classifiers wrapped in Arc for thread safety
    intent_classifier: Option<Arc<CandleBertClassifier>>,
    pii_classifier: Option<Arc<CandleBertTokenClassifier>>,
    security_classifier: Option<Arc<CandleBertClassifier>>,

    // Legacy ModernBERT support (for backward compatibility)
    encoder: Option<ModernBert>,
    tokenizer: Option<Tokenizer>,
    intent_head: Option<Linear>,
    pii_head: Option<Linear>,
    security_head: Option<Linear>,

    // Task label mappings
    intent_mapping: HashMap<usize, String>,
    pii_mapping: HashMap<usize, String>,
    security_mapping: HashMap<usize, String>,

    // Configuration
    max_sequence_length: usize,
    pad_token_id: u32,
}

impl UnifiedClassifier {
    /// Create a new unified classifier with high-confidence LoRA models
    pub fn new_with_lora_models(
        intent_model_path: &str,
        pii_model_path: &str,
        security_model_path: &str,
        architecture: &str, // "bert", "roberta", or "modernbert"
        use_cpu: bool,
    ) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        let mut classifier = Self {
            architecture: architecture.to_string(),
            device,
            intent_classifier: None,
            pii_classifier: None,
            security_classifier: None,
            encoder: None,
            tokenizer: None,
            intent_head: None,
            pii_head: None,
            security_head: None,
            intent_mapping: HashMap::new(),
            pii_mapping: HashMap::new(),
            security_mapping: HashMap::new(),
            max_sequence_length: 512,
            pad_token_id: 0,
        };

        // Load high-confidence LoRA models
        classifier.load_lora_models(intent_model_path, pii_model_path, security_model_path)?;

        Ok(classifier)
    }

    /// Load our high-confidence LoRA models
    fn load_lora_models(
        &mut self,
        intent_path: &str,
        pii_path: &str,
        security_path: &str,
    ) -> Result<()> {
        // Load intent classifier
        if Path::new(intent_path).exists() {
            let intent_labels = self.load_labels_from_path(intent_path)?;
            let num_classes = intent_labels.len();

            let intent_classifier = CandleBertClassifier::new(
                intent_path,
                num_classes,
                matches!(self.device, Device::Cpu),
            )?;

            self.intent_classifier = Some(Arc::new(intent_classifier));
            self.intent_mapping = intent_labels;
        }

        // Load security classifier
        if Path::new(security_path).exists() {
            let security_labels = self.load_labels_from_path(security_path)?;
            let num_classes = security_labels.len();

            let security_classifier = CandleBertClassifier::new(
                security_path,
                num_classes,
                matches!(self.device, Device::Cpu),
            )?;

            self.security_classifier = Some(Arc::new(security_classifier));
            self.security_mapping = security_labels;
        }

        // Load PII token classifier
        if Path::new(pii_path).exists() {
            let pii_labels = self.load_labels_from_path(pii_path)?;
            let num_classes = pii_labels.len();

            let pii_classifier = CandleBertTokenClassifier::new(
                pii_path,
                num_classes,
                matches!(self.device, Device::Cpu),
            )?;

            self.pii_classifier = Some(Arc::new(pii_classifier));
            self.pii_mapping = pii_labels;
        }

        Ok(())
    }

    /// Load label mappings from model directory
    fn load_labels_from_path(&self, model_path: &str) -> Result<HashMap<usize, String>> {
        // Try to load from config.json first
        let config_path = Path::new(model_path).join("config.json");
        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&config_str)?;

            if let Some(id2label) = config.get("id2label") {
                let mut labels = HashMap::new();
                if let Some(obj) = id2label.as_object() {
                    for (id_str, label) in obj {
                        if let (Ok(id), Some(label_str)) = (id_str.parse::<usize>(), label.as_str())
                        {
                            labels.insert(id, label_str.to_string());
                        }
                    }
                }
                if !labels.is_empty() {
                    return Ok(labels);
                }
            }
        }

        // Try to load from label_mapping.json
        let label_path = Path::new(model_path).join("label_mapping.json");
        if label_path.exists() {
            let label_str = std::fs::read_to_string(&label_path)?;
            let label_data: serde_json::Value = serde_json::from_str(&label_str)?;

            if let Some(id2label) = label_data.get("id_to_label") {
                let mut labels = HashMap::new();
                if let Some(obj) = id2label.as_object() {
                    for (id_str, label) in obj {
                        if let (Ok(id), Some(label_str)) = (id_str.parse::<usize>(), label.as_str())
                        {
                            labels.insert(id, label_str.to_string());
                        }
                    }
                }
                return Ok(labels);
            }
        }

        Err(E::msg("No label mapping found"))
    }

    /// Create a new unified classifier with dynamic label mappings (legacy ModernBERT)
    pub fn new(
        modernbert_path: &str,
        intent_head_path: &str,
        pii_head_path: &str,
        security_head_path: &str,
        intent_labels: Vec<String>,
        pii_labels: Vec<String>,
        security_labels: Vec<String>,
        use_cpu: bool,
    ) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Load shared ModernBERT encoder using real weights (legacy mode)
        let tokenizer = Self::load_tokenizer(modernbert_path)?;

        // Load configuration from the model directory
        let config_path = format!("{}/config.json", modernbert_path);
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;

        // Load model weights - try safetensors first, then pytorch
        let vb = if std::path::Path::new(&format!("{}/model.safetensors", modernbert_path)).exists()
        {
            let weights_path = format!("{}/model.safetensors", modernbert_path);
            unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[weights_path],
                    candle_core::DType::F32,
                    &device,
                )?
            }
        } else if std::path::Path::new(&format!("{}/pytorch_model.bin", modernbert_path)).exists() {
            let weights_path = format!("{}/pytorch_model.bin", modernbert_path);
            candle_nn::VarBuilder::from_pth(&weights_path, candle_core::DType::F32, &device)?
        } else {
            return Err(E::msg(format!(
                "No model weights found in {}",
                modernbert_path
            )));
        };

        // Load the real ModernBERT encoder
        let encoder = ModernBert::load(vb.clone(), &config)?;

        // Load task-specific heads with real weights
        let intent_head = Self::load_classification_head(
            &device,
            intent_head_path,
            intent_labels.len(),
            config.hidden_size,
        )?;
        let pii_head = Self::load_classification_head(
            &device,
            pii_head_path,
            pii_labels.len(),
            config.hidden_size,
        )?;
        let security_head = Self::load_classification_head(
            &device,
            security_head_path,
            security_labels.len(),
            config.hidden_size,
        )?;

        // Create label mappings from provided labels
        let intent_mapping = Self::create_mapping_from_labels(&intent_labels);
        let pii_mapping = Self::create_mapping_from_labels(&pii_labels);
        let security_mapping = Self::create_mapping_from_labels(&security_labels);

        Ok(Self {
            architecture: "modernbert".to_string(),
            device,
            intent_classifier: None,
            pii_classifier: None,
            security_classifier: None,
            encoder: Some(encoder),
            tokenizer: Some(tokenizer),
            intent_head: Some(intent_head),
            pii_head: Some(pii_head),
            security_head: Some(security_head),
            intent_mapping,
            pii_mapping,
            security_mapping,
            max_sequence_length: 512,
            pad_token_id: 0,
        })
    }

    /// Core batch classification method - processes multiple texts in one forward pass
    /// Supports both high-confidence LoRA models and legacy ModernBERT
    pub fn classify_batch(&self, texts: &[&str]) -> Result<BatchClassificationResult> {
        if texts.is_empty() {
            return Err(E::msg("Empty text batch"));
        }

        // Check if we have LoRA models
        if self.intent_classifier.is_some()
            || self.pii_classifier.is_some()
            || self.security_classifier.is_some()
        {
            return self.classify_batch_with_lora(texts);
        }

        // Fallback to legacy ModernBERT mode
        self.classify_batch_legacy(texts)
    }

    /// High-confidence batch classification using LoRA models with PARALLEL PROCESSING
    fn classify_batch_with_lora(&self, texts: &[&str]) -> Result<BatchClassificationResult> {
        // PERFORMANCE OPTIMIZATION: Parallel execution of 3 LoRA models
        // Instead of sequential: Intent -> PII -> Security (3x time)
        // Use parallel: Intent || PII || Security (1x time + overhead)

        let texts_vec: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        // Clone classifiers for thread safety (they're already Arc-wrapped internally)
        let intent_classifier = self.intent_classifier.clone();
        let pii_classifier = self.pii_classifier.clone();
        let security_classifier = self.security_classifier.clone();

        // Clone mappings for thread safety
        let intent_mapping = self.intent_mapping.clone();
        let pii_mapping = self.pii_mapping.clone();
        let security_mapping = self.security_mapping.clone();

        // Spawn parallel threads for each classification task
        let intent_handle = {
            let texts_clone = texts_vec.clone();
            let mapping_clone = intent_mapping.clone();
            thread::spawn(move || -> Result<Vec<IntentResult>> {
                if let Some(classifier) = intent_classifier {
                    let texts_refs: Vec<&str> = texts_clone.iter().map(|s| s.as_str()).collect();
                    match classifier.classify_batch(&texts_refs) {
                        Ok(batch_results) => Ok(batch_results
                            .into_iter()
                            .map(|(class_id, confidence)| {
                                let category = mapping_clone
                                    .get(&class_id)
                                    .unwrap_or(&format!("UNKNOWN_{}", class_id))
                                    .clone();
                                IntentResult {
                                    category,
                                    confidence,
                                    probabilities: Vec::new(),
                                }
                            })
                            .collect()),
                        Err(_) => Ok(texts_clone
                            .iter()
                            .map(|_| IntentResult {
                                category: "ERROR".to_string(),
                                confidence: 0.0,
                                probabilities: Vec::new(),
                            })
                            .collect()),
                    }
                } else {
                    Ok(texts_clone
                        .iter()
                        .map(|_| IntentResult {
                            category: "NO_CLASSIFIER".to_string(),
                            confidence: 0.0,
                            probabilities: Vec::new(),
                        })
                        .collect())
                }
            })
        };

        let pii_handle = {
            let texts_clone = texts_vec.clone();
            let mapping_clone = pii_mapping.clone();
            thread::spawn(move || -> Result<Vec<PIIResult>> {
                if let Some(classifier) = pii_classifier {
                    let texts_refs: Vec<&str> = texts_clone.iter().map(|s| s.as_str()).collect();
                    match classifier.classify_tokens_batch(&texts_refs) {
                        Ok(batch_results) => Ok(batch_results
                            .into_iter()
                            .map(|token_results| {
                                let entities: Vec<String> = token_results
                                    .iter()
                                    .filter(|(_, class_id, confidence)| {
                                        *class_id > 0 && *confidence > 0.5
                                    })
                                    .map(|(_token, class_id, _)| {
                                        mapping_clone
                                            .get(class_id)
                                            .unwrap_or(&format!("UNKNOWN_{}", class_id))
                                            .clone()
                                    })
                                    .collect();

                                PIIResult {
                                    has_pii: !entities.is_empty(),
                                    pii_types: entities.clone(),
                                    confidence: token_results
                                        .iter()
                                        .map(|(_, _, conf)| *conf)
                                        .fold(0.0, f32::max),
                                    entities,
                                }
                            })
                            .collect()),
                        Err(_) => Ok(texts_clone
                            .iter()
                            .map(|_| PIIResult {
                                has_pii: false,
                                pii_types: Vec::new(),
                                confidence: 0.0,
                                entities: Vec::new(),
                            })
                            .collect()),
                    }
                } else {
                    Ok(texts_clone
                        .iter()
                        .map(|_| PIIResult {
                            has_pii: false,
                            pii_types: Vec::new(),
                            confidence: 0.0,
                            entities: Vec::new(),
                        })
                        .collect())
                }
            })
        };

        let security_handle = {
            let texts_clone = texts_vec.clone();
            let mapping_clone = security_mapping.clone();
            thread::spawn(move || -> Result<Vec<SecurityResult>> {
                if let Some(classifier) = security_classifier {
                    let texts_refs: Vec<&str> = texts_clone.iter().map(|s| s.as_str()).collect();
                    match classifier.classify_batch(&texts_refs) {
                        Ok(batch_results) => Ok(batch_results
                            .into_iter()
                            .map(|(class_id, confidence)| {
                                let threat_type = mapping_clone
                                    .get(&class_id)
                                    .unwrap_or(&format!("UNKNOWN_{}", class_id))
                                    .clone();

                                SecurityResult {
                                    is_jailbreak: class_id == 1,
                                    threat_type,
                                    confidence,
                                }
                            })
                            .collect()),
                        Err(_) => Ok(texts_clone
                            .iter()
                            .map(|_| SecurityResult {
                                is_jailbreak: false,
                                threat_type: "ERROR".to_string(),
                                confidence: 0.0,
                            })
                            .collect()),
                    }
                } else {
                    Ok(texts_clone
                        .iter()
                        .map(|_| SecurityResult {
                            is_jailbreak: false,
                            threat_type: "NO_CLASSIFIER".to_string(),
                            confidence: 0.0,
                        })
                        .collect())
                }
            })
        };

        // Wait for all threads to complete and collect results
        let intent_results = intent_handle
            .join()
            .map_err(|_| E::msg("Intent classification thread panicked"))?
            .map_err(|e| E::msg(format!("Intent classification failed: {}", e)))?;

        let pii_results = pii_handle
            .join()
            .map_err(|_| E::msg("PII classification thread panicked"))?
            .map_err(|e| E::msg(format!("PII classification failed: {}", e)))?;

        let security_results = security_handle
            .join()
            .map_err(|_| E::msg("Security classification thread panicked"))?
            .map_err(|e| E::msg(format!("Security classification failed: {}", e)))?;

        Ok(BatchClassificationResult {
            intent_results,
            pii_results,
            security_results,
            batch_size: texts.len(),
        })
    }

    /// Legacy batch classification using ModernBERT (backward compatibility)
    fn classify_batch_legacy(&self, texts: &[&str]) -> Result<BatchClassificationResult> {
        // Step 1: Batch tokenization - tokenize all texts at once
        let encodings = self.tokenize_batch(texts)?;

        // Step 2: Create batch tensors with proper padding
        let (input_ids, attention_mask) = self.create_batch_tensors(&encodings)?;

        // Step 3: Single shared encoder forward pass - this is the key optimization!
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| E::msg("ModernBERT encoder not initialized"))?;
        let embeddings = encoder.forward(&input_ids, &attention_mask)?;

        // Step 4: Pool embeddings (CLS token or mean pooling)
        let pooled_embeddings = self.pool_embeddings(&embeddings, &attention_mask)?;

        // Step 5: Parallel multi-task head computation
        let intent_head = self
            .intent_head
            .as_ref()
            .ok_or_else(|| E::msg("Intent head not initialized"))?;
        let pii_head = self
            .pii_head
            .as_ref()
            .ok_or_else(|| E::msg("PII head not initialized"))?;
        let security_head = self
            .security_head
            .as_ref()
            .ok_or_else(|| E::msg("Security head not initialized"))?;

        let intent_logits = intent_head.forward(&pooled_embeddings)?;
        let pii_logits = pii_head.forward(&pooled_embeddings)?;
        let security_logits = security_head.forward(&pooled_embeddings)?;

        // Step 6: Process results for each task
        let intent_results = self.process_intent_batch(&intent_logits)?;
        let pii_results = self.process_pii_batch(&pii_logits)?;
        let security_results = self.process_security_batch(&security_logits)?;

        Ok(BatchClassificationResult {
            intent_results,
            pii_results,
            security_results,
            batch_size: texts.len(),
        })
    }

    /// Tokenize a batch of texts efficiently
    fn tokenize_batch(&self, texts: &[&str]) -> Result<Vec<Encoding>> {
        let tokenizer_ref = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| E::msg("Tokenizer not initialized"))?;
        let mut tokenizer = tokenizer_ref.clone();

        // Configure padding for batch processing
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: self.pad_token_id,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        // Batch encode all texts
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(E::msg)?;

        Ok(encodings)
    }

    /// Create batch tensors from encodings with proper padding
    fn create_batch_tensors(&self, encodings: &[Encoding]) -> Result<(Tensor, Tensor)> {
        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.len().min(self.max_sequence_length))
            .max()
            .unwrap_or(self.max_sequence_length);

        // Initialize tensors
        let mut input_ids = vec![vec![self.pad_token_id; max_len]; batch_size];
        let mut attention_mask = vec![vec![0u32; max_len]; batch_size];

        // Fill tensors with actual data
        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let len = ids.len().min(max_len);

            // Copy input IDs and attention mask
            for j in 0..len {
                input_ids[i][j] = ids[j];
                attention_mask[i][j] = mask[j];
            }
        }

        // Convert to tensors
        let input_ids_tensor = Tensor::new(input_ids, &self.device)?;
        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)?;

        Ok((input_ids_tensor, attention_mask_tensor))
    }

    /// Pool embeddings using CLS token (first token)
    fn pool_embeddings(&self, embeddings: &Tensor, _attention_mask: &Tensor) -> Result<Tensor> {
        // Use CLS token (index 0) for classification
        // Shape: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        let cls_embeddings = embeddings.i((.., 0, ..))?;
        Ok(cls_embeddings)
    }

    /// Process intent classification results
    fn process_intent_batch(&self, logits: &Tensor) -> Result<Vec<IntentResult>> {
        let probabilities = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;
        let probs_data = probabilities.to_vec2::<f32>()?;

        let mut results = Vec::new();
        for prob_row in probs_data {
            let (max_idx, max_prob) = prob_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let category = self
                .intent_mapping
                .get(&max_idx)
                .cloned()
                .unwrap_or_else(|| format!("unknown_{}", max_idx));

            results.push(IntentResult {
                category,
                confidence: *max_prob,
                probabilities: prob_row,
            });
        }

        Ok(results)
    }

    /// Process PII detection results
    fn process_pii_batch(&self, logits: &Tensor) -> Result<Vec<PIIResult>> {
        let probabilities = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;
        let probs_data = probabilities.to_vec2::<f32>()?;

        let mut results = Vec::new();
        for prob_row in probs_data {
            // For PII, we use a threshold-based approach
            let mut pii_types = Vec::new();
            let mut max_confidence = 0.0f32;

            for (idx, &prob) in prob_row.iter().enumerate() {
                if prob > 0.5 {
                    // Threshold for PII detection
                    if let Some(pii_type) = self.pii_mapping.get(&idx) {
                        pii_types.push(pii_type.clone());
                        max_confidence = max_confidence.max(prob);
                    }
                }
            }

            results.push(PIIResult {
                has_pii: !pii_types.is_empty(),
                pii_types,
                confidence: max_confidence,
                entities: Vec::new(), // Simplified for now
            });
        }

        Ok(results)
    }

    /// Process security detection results
    fn process_security_batch(&self, logits: &Tensor) -> Result<Vec<SecurityResult>> {
        let probabilities = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;
        let probs_data = probabilities.to_vec2::<f32>()?;

        let mut results = Vec::new();
        for prob_row in probs_data {
            // Binary classification: [safe, jailbreak]
            let (max_idx, max_prob) = prob_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let is_jailbreak = max_idx == 1; // Index 1 is jailbreak
            let threat_type = self
                .security_mapping
                .get(&max_idx)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());

            results.push(SecurityResult {
                is_jailbreak,
                threat_type,
                confidence: *max_prob,
            });
        }

        Ok(results)
    }

    // Helper methods for loading components
    fn load_tokenizer(model_path: &str) -> Result<Tokenizer> {
        let tokenizer_path = format!("{}/tokenizer.json", model_path);
        Tokenizer::from_file(&tokenizer_path).map_err(E::msg)
    }

    fn load_classification_head(
        device: &Device,
        head_path: &str,
        num_classes: usize,
        hidden_size: usize,
    ) -> Result<Linear> {
        // Load classification head from existing model weights

        // Load model weights - try safetensors first, then pytorch
        let vb = if std::path::Path::new(&format!("{}/model.safetensors", head_path)).exists() {
            let weights_path = format!("{}/model.safetensors", head_path);
            unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[weights_path],
                    candle_core::DType::F32,
                    device,
                )?
            }
        } else if std::path::Path::new(&format!("{}/pytorch_model.bin", head_path)).exists() {
            let weights_path = format!("{}/pytorch_model.bin", head_path);
            candle_nn::VarBuilder::from_pth(&weights_path, candle_core::DType::F32, device)?
        } else {
            return Err(E::msg(format!("No model weights found in {}", head_path)));
        };

        // Try to load classifier weights - try different possible paths
        let classifier = if let Ok(weights) =
            vb.get((num_classes, hidden_size), "classifier.weight")
        {
            // Standard classifier path
            let bias = vb.get((num_classes,), "classifier.bias").ok();
            Linear::new(weights, bias)
        } else if let Ok(weights) =
            vb.get((num_classes, hidden_size), "_orig_mod.classifier.weight")
        {
            // Torch.compile models with _orig_mod prefix
            let bias = vb.get((num_classes,), "_orig_mod.classifier.bias").ok();
            Linear::new(weights, bias)
        } else {
            return Err(E::msg(format!("No classifier weights found in {} - tried 'classifier.weight' and '_orig_mod.classifier.weight'", head_path)));
        };

        Ok(classifier)
    }

    /// Create mapping from provided labels
    fn create_mapping_from_labels(labels: &[String]) -> HashMap<usize, String> {
        let mut mapping = HashMap::new();
        for (i, label) in labels.iter().enumerate() {
            mapping.insert(i, label.clone());
        }
        mapping
    }
}

// Global unified classifier instance
lazy_static::lazy_static! {
    pub static ref UNIFIED_CLASSIFIER: Arc<Mutex<Option<UnifiedClassifier>>> = Arc::new(Mutex::new(None));
}

/// Get reference to the global unified classifier
pub fn get_unified_classifier() -> Result<std::sync::MutexGuard<'static, Option<UnifiedClassifier>>>
{
    Ok(UNIFIED_CLASSIFIER.lock().unwrap())
}
