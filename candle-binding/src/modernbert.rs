// ModernBERT binding for classification tasks
// Based on ModernBERT implementation in candle-transformers

use std::ffi::{c_char, CStr};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_core::{IndexOp, D};
use candle_nn::ops;
use candle_nn::Module;
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::{
    ClassifierConfig, ClassifierPooling, Config, ModernBert,
};
use libc;
use serde_json;
use std::collections::HashMap;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

// ================================================================================================
// FIXED MODERNBERT IMPLEMENTATION
// ================================================================================================
// This implementation fixes the bugs in candle-transformers ModernBERT:
// 1. Proper token ID to embedding conversion
// 2. Correct pooling logic (CLS vs MEAN)
// 3. Proper error handling and validation

/// Fixed ModernBERT classifier that handles embeddings correctly
#[derive(Clone)]
pub struct FixedModernBertClassifier {
    classifier: candle_nn::Linear,
}

impl FixedModernBertClassifier {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let num_classes = config
            .classifier_config
            .as_ref()
            .map(|cc| cc.id2label.len())
            .unwrap_or(2);

        let classifier = candle_nn::Linear::new(
            vb.get((num_classes, config.hidden_size), "classifier.weight")?,
            Some(vb.get((num_classes,), "classifier.bias")?),
        );

        Ok(Self { classifier })
    }
}

impl Module for FixedModernBertClassifier {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let logits = xs.apply(&self.classifier)?;
        // Apply softmax to get probabilities
        ops::softmax(&logits, D::Minus1)
    }
}

/// Fixed ModernBERT head (dense layer + layer norm)
#[derive(Clone)]
pub struct FixedModernBertHead {
    dense: candle_nn::Linear,
    layer_norm: candle_nn::LayerNorm,
}

impl FixedModernBertHead {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = candle_nn::Linear::new(
            vb.get((config.hidden_size, config.hidden_size), "dense.weight")?,
            None,
        );

        // Load layer norm - it's called "norm" not "layer_norm" in this model!
        // And no bias based on actual model inspection
        let layer_norm = candle_nn::LayerNorm::new(
            vb.get((config.hidden_size,), "norm.weight")?,
            // Create a zero bias tensor since LayerNorm::new requires it but the model doesn't have one
            Tensor::zeros((config.hidden_size,), DType::F32, vb.device())?,
            1e-12,
        );

        Ok(Self { dense, layer_norm })
    }
}

impl Module for FixedModernBertHead {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = xs.apply(&self.dense)?;
        // Apply GELU activation
        let xs = xs.gelu()?;
        xs.apply(&self.layer_norm)
    }
}

/// Fixed ModernBERT sequence classification model that properly handles embeddings
#[derive(Clone)]
pub struct FixedModernBertForSequenceClassification {
    model: ModernBert,                 // Use the base model (this should work)
    head: Option<FixedModernBertHead>, // Head might not exist in some ModernBERT models
    classifier: FixedModernBertClassifier,
    classifier_pooling: ClassifierPooling,
}

/// Fixed ModernBERT token classifier for token-level predictions
#[derive(Clone)]
pub struct FixedModernBertTokenClassifier {
    classifier: candle_nn::Linear,
}

impl FixedModernBertTokenClassifier {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let num_classes = config
            .classifier_config
            .as_ref()
            .map(|cc| cc.id2label.len())
            .unwrap_or(2);

        let classifier = candle_nn::Linear::new(
            vb.get((num_classes, config.hidden_size), "classifier.weight")?,
            Some(vb.get((num_classes,), "classifier.bias")?),
        );

        Ok(Self { classifier })
    }
}

impl Module for FixedModernBertTokenClassifier {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // For token classification, we don't apply softmax here
        // as we need raw logits for each token position
        xs.apply(&self.classifier)
    }
}

/// Fixed ModernBERT token classification model that properly handles embeddings
#[derive(Clone)]
pub struct FixedModernBertForTokenClassification {
    model: ModernBert,                 // Use the base model
    head: Option<FixedModernBertHead>, // Head might not exist in some ModernBERT models
    classifier: FixedModernBertTokenClassifier,
}

impl FixedModernBertForTokenClassification {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let model = ModernBert::load(vb.clone(), config)?;

        // Try to load head - it might not exist in all ModernBERT models
        let head = match vb.get(
            (config.hidden_size, config.hidden_size),
            "head.dense.weight",
        ) {
            Ok(_) => {
                let head_vb = vb.pp("head");
                Some(FixedModernBertHead::load(head_vb, config)?)
            }
            Err(_) => None,
        };

        let classifier = FixedModernBertTokenClassifier::load(vb.clone(), config)?;

        Ok(Self {
            model,
            head,
            classifier,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Get embeddings from the base model
        let output = self.model.forward(xs, mask).map_err(|e| {
            let error_str = format!("{e}");
            E::msg(format!("Base model failed: {error_str}"))
        })?;

        // Apply head (dense + layer norm) if it exists
        let classifier_input = match &self.head {
            Some(head) => head.forward(&output).map_err(E::msg)?,
            None => output,
        };

        // Apply token classifier to get logits for each token position
        let logits = self.classifier.forward(&classifier_input).map_err(E::msg)?;

        Ok(logits)
    }
}

impl FixedModernBertForSequenceClassification {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let model = ModernBert::load(vb.clone(), config)?;

        // Try to load head - it might not exist in all ModernBERT models
        let head = match vb.get(
            (config.hidden_size, config.hidden_size),
            "head.dense.weight",
        ) {
            Ok(_) => {
                let head_vb = vb.pp("head");
                Some(FixedModernBertHead::load(head_vb, config)?)
            }
            Err(_) => None,
        };

        let classifier = FixedModernBertClassifier::load(vb.clone(), config)?;

        let classifier_pooling = config
            .classifier_config
            .as_ref()
            .map(|cc| cc.classifier_pooling)
            .unwrap_or(ClassifierPooling::CLS);

        Ok(Self {
            model,
            head,
            classifier,
            classifier_pooling,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Get embeddings from the base model
        let output = self.model.forward(xs, mask).map_err(|e| {
            let error_str = format!("{e}");
            E::msg(format!("Base model failed: {error_str}"))
        })?;

        // Apply correct pooling logic
        let pooled = match self.classifier_pooling {
            ClassifierPooling::CLS => output.i((.., 0, ..))?,
            ClassifierPooling::MEAN => {
                let mask_expanded = mask.unsqueeze(D::Minus1)?.to_dtype(DType::F32)?;
                let masked_output = output.broadcast_mul(&mask_expanded)?;
                let sum_output = masked_output.sum(1)?;
                let mask_sum = mask.sum_keepdim(1)?.to_dtype(DType::F32)?;
                sum_output.broadcast_div(&mask_sum)?
            }
        };

        // Apply head (dense + layer norm) if it exists
        let classifier_input = match &self.head {
            Some(head) => head.forward(&pooled).map_err(E::msg)?,
            None => pooled,
        };

        // Apply classifier (linear + softmax)
        let probabilities = self.classifier.forward(&classifier_input).map_err(E::msg)?;

        Ok(probabilities)
    }
}

// Enum to hold different types of ModernBERT models
pub enum ModernBertModel {
    Sequence(FixedModernBertForSequenceClassification),
    Token(FixedModernBertForTokenClassification),
}

// Structure to hold ModernBERT model and tokenizer for text classification
pub struct ModernBertClassifier {
    model: ModernBertModel,
    tokenizer: Tokenizer,
    device: Device,
    pad_token_id: u32,
    is_token_classification: bool,
}

lazy_static::lazy_static! {
    static ref MODERNBERT_CLASSIFIER: Arc<Mutex<Option<ModernBertClassifier>>> = Arc::new(Mutex::new(None));
    static ref MODERNBERT_PII_CLASSIFIER: Arc<Mutex<Option<ModernBertClassifier>>> = Arc::new(Mutex::new(None));
    static ref MODERNBERT_JAILBREAK_CLASSIFIER: Arc<Mutex<Option<ModernBertClassifier>>> = Arc::new(Mutex::new(None));
}

// Structure to hold classification result
#[repr(C)]
pub struct ModernBertClassificationResult {
    pub class: i32,
    pub confidence: f32,
}

// Structure to hold classification result with full probability distribution
#[repr(C)]
pub struct ModernBertClassificationResultWithProbs {
    pub class: i32,
    pub confidence: f32,
    pub probabilities: *mut f32,
    pub num_classes: i32,
}

// Structure to hold token classification entity result
#[repr(C)]
pub struct ModernBertTokenEntity {
    pub entity_type: *mut c_char,
    pub start: i32,
    pub end: i32,
    pub text: *mut c_char,
    pub confidence: f32,
}

// Structure to hold token classification result (array of entities)
#[repr(C)]
pub struct ModernBertTokenClassificationResult {
    pub entities: *mut ModernBertTokenEntity,
    pub num_entities: i32,
}

impl ModernBertClassifier {
    pub fn new(model_id: &str, use_cpu: bool) -> Result<Self> {
        Self::new_internal(model_id, use_cpu, false)
    }

    pub fn new_token_classification(model_id: &str, use_cpu: bool) -> Result<Self> {
        Self::new_internal(model_id, use_cpu, true)
    }

    /// Internal implementation using the fixed ModernBERT
    fn new_internal(model_id: &str, use_cpu: bool, is_token_classification: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Check if this is a SentenceTransformer ModernBERT model
        let _is_sentence_transformer = Path::new(model_id).join("modules.json").exists();

        let (config_filename, tokenizer_filename, weights_filename, use_pth) =
            if Path::new(model_id).exists() {
                // Local model path
                let config_path = Path::new(model_id).join("config.json");
                let tokenizer_path = Path::new(model_id).join("tokenizer.json");

                // Check for safetensors first, fall back to PyTorch
                let weights_path = if Path::new(model_id).join("model.safetensors").exists() {
                    (
                        Path::new(model_id)
                            .join("model.safetensors")
                            .to_string_lossy()
                            .to_string(),
                        false,
                    )
                } else if Path::new(model_id).join("pytorch_model.bin").exists() {
                    (
                        Path::new(model_id)
                            .join("pytorch_model.bin")
                            .to_string_lossy()
                            .to_string(),
                        true,
                    )
                } else {
                    return Err(E::msg(format!("No model weights found in {model_id}")));
                };

                (
                    config_path.to_string_lossy().to_string(),
                    tokenizer_path.to_string_lossy().to_string(),
                    weights_path.0,
                    weights_path.1,
                )
            } else {
                return Err(E::msg(format!(
                    "HuggingFace Hub loading for ModernBERT {model_id} not yet implemented"
                )));
            };

        let config_str = std::fs::read_to_string(&config_filename)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)?
            }
        };

        // Check if we have id2label and label2id mappings either in classifier_config or at the top level
        let mut config = config;

        // Check if classifier_config exists and has mappings
        let has_classifier_config = config
            .classifier_config
            .as_ref()
            .map(|cc| !cc.id2label.is_empty())
            .unwrap_or(false);

        // If no classifier_config or it's empty, check for top-level id2label/label2id
        if !has_classifier_config {
            // Try to access top-level id2label and label2id fields

            let config_str = std::fs::read_to_string(config_filename)?;
            let config_json: serde_json::Value = serde_json::from_str(&config_str)?;

            if let (Some(id2label), Some(label2id)) = (
                config_json.get("id2label").and_then(|v| v.as_object()),
                config_json.get("label2id").and_then(|v| v.as_object()),
            ) {
                // Convert JSON objects to HashMap<String, String>
                let id2label_map: HashMap<String, String> = id2label
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("UNKNOWN").to_string()))
                    .collect();

                let label2id_map: HashMap<String, String> = label2id
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0).to_string()))
                    .collect();

                // Extract classifier_pooling from top-level config
                let classifier_pooling = config_json
                    .get("classifier_pooling")
                    .and_then(|v| v.as_str())
                    .map(|s| match s {
                        "cls" => ClassifierPooling::CLS,
                        "mean" => ClassifierPooling::MEAN,
                        _ => ClassifierPooling::CLS, // Default to CLS
                    })
                    .unwrap_or(ClassifierPooling::CLS);

                let classifier_config = ClassifierConfig {
                    id2label: id2label_map,
                    label2id: label2id_map,
                    classifier_pooling,
                };

                config.classifier_config = Some(classifier_config);
            } else {
                return Err(E::msg(
                    "No id2label/label2id mappings found in config - required for classification",
                ));
            }
        }

        // Load the appropriate ModernBERT model based on task type
        // Try standard naming first, then _orig_mod prefix if that fails
        let model = if is_token_classification {
            match FixedModernBertForTokenClassification::load(vb.clone(), &config) {
                Ok(model) => ModernBertModel::Token(model),
                Err(_) => {
                    // Try with _orig_mod prefix (torch.compile models)
                    ModernBertModel::Token(FixedModernBertForTokenClassification::load(
                        vb.pp("_orig_mod"),
                        &config,
                    )?)
                }
            }
        } else {
            match FixedModernBertForSequenceClassification::load(vb.clone(), &config) {
                Ok(model) => ModernBertModel::Sequence(model),
                Err(_) => {
                    // Try with _orig_mod prefix (torch.compile models)
                    ModernBertModel::Sequence(FixedModernBertForSequenceClassification::load(
                        vb.pp("_orig_mod"),
                        &config,
                    )?)
                }
            }
        };

        Ok(Self {
            model,
            tokenizer,
            device,
            pad_token_id: config.pad_token_id,
            is_token_classification,
        })
    }

    pub fn classify_text(&self, text: &str) -> Result<(usize, f32)> {
        if self.is_token_classification {
            return Err(E::msg(
                "Use classify_tokens for token classification models",
            ));
        }

        // Set up tokenizer
        let mut tokenizer = self.tokenizer.clone();

        // Set up padding - use config's pad_token_id and no truncation
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                pad_id: self.pad_token_id,
                ..Default::default()
            }))
            .with_truncation(None)
            .map_err(E::msg)?;

        // Tokenize input text
        let tokens = tokenizer.encode_batch(vec![text], true).map_err(E::msg)?;

        // Create tensors - convert to u32 for ModernBERT
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens: Vec<u32> = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens: Vec<u32> = tokens.get_attention_mask().to_vec();
                Tensor::new(tokens.as_slice(), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let input_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;

        // Input validation
        if input_ids.dims().len() != 2 {
            return Err(E::msg(format!(
                "Expected input_ids to have 2 dimensions [batch_size, seq_len], got {:?}",
                input_ids.dims()
            )));
        }
        if attention_mask.dims().len() != 2 {
            return Err(E::msg(format!(
                "Expected attention_mask to have 2 dimensions [batch_size, seq_len], got {:?}",
                attention_mask.dims()
            )));
        }
        if input_ids.dims()[0] != attention_mask.dims()[0]
            || input_ids.dims()[1] != attention_mask.dims()[1]
        {
            return Err(E::msg(format!(
                "input_ids and attention_mask must have same shape, got {:?} vs {:?}",
                input_ids.dims(),
                attention_mask.dims()
            )));
        }

        // Run through ModernBERT model
        let output = match &self.model {
            ModernBertModel::Sequence(model) => model.forward(&input_ids, &attention_mask)?,
            ModernBertModel::Token(_) => {
                return Err(E::msg(
                    "Internal error: token model in sequence classification",
                ))
            }
        };

        // Remove batch dimension if present
        let probabilities = if output.dims().len() > 1 {
            output.squeeze(0)?
        } else {
            output
        };

        // Convert to vector and find the class with highest probability
        let probabilities_vec = probabilities.to_vec1::<f32>()?;

        // Get the predicted class with highest probability
        let (predicted_idx, &max_prob) = probabilities_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        Ok((predicted_idx, max_prob))
    }

    /// Classify text and return full probability distribution
    pub fn classify_text_with_probs(&self, text: &str) -> Result<(usize, f32, Vec<f32>)> {
        if self.is_token_classification {
            return Err(E::msg(
                "Use classify_tokens for token classification models",
            ));
        }

        // Set up tokenizer
        let mut tokenizer = self.tokenizer.clone();

        // Set up padding - use config's pad_token_id and no truncation
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                pad_id: self.pad_token_id,
                ..Default::default()
            }))
            .with_truncation(None)
            .map_err(E::msg)?;

        // Tokenize input text
        let tokens = tokenizer.encode_batch(vec![text], true).map_err(E::msg)?;

        // Create tensors - convert to u32 for ModernBERT
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens: Vec<u32> = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens: Vec<u32> = tokens.get_attention_mask().to_vec();
                Tensor::new(tokens.as_slice(), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let input_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;

        // Input validation
        if input_ids.dims().len() != 2 {
            return Err(E::msg(format!(
                "Expected input_ids to have 2 dimensions [batch_size, seq_len], got {:?}",
                input_ids.dims()
            )));
        }
        if attention_mask.dims().len() != 2 {
            return Err(E::msg(format!(
                "Expected attention_mask to have 2 dimensions [batch_size, seq_len], got {:?}",
                attention_mask.dims()
            )));
        }
        if input_ids.dims()[0] != attention_mask.dims()[0]
            || input_ids.dims()[1] != attention_mask.dims()[1]
        {
            return Err(E::msg(format!(
                "input_ids and attention_mask must have same shape, got {:?} vs {:?}",
                input_ids.dims(),
                attention_mask.dims()
            )));
        }

        // Run through ModernBERT model
        let output = match &self.model {
            ModernBertModel::Sequence(model) => model.forward(&input_ids, &attention_mask)?,
            ModernBertModel::Token(_) => {
                return Err(E::msg(
                    "Internal error: token model in sequence classification",
                ))
            }
        };

        // Remove batch dimension if present
        let probabilities = if output.dims().len() > 1 {
            output.squeeze(0)?
        } else {
            output
        };

        // Convert to vector and get full probability distribution
        let probabilities_vec = probabilities.to_vec1::<f32>()?;

        // Get the predicted class with highest probability
        let (predicted_idx, &max_prob) = probabilities_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        // Return predicted class, max probability, and full distribution
        Ok((predicted_idx, max_prob, probabilities_vec))
    }

    pub fn classify_tokens(
        &self,
        text: &str,
        id2label: &HashMap<String, String>,
    ) -> Result<Vec<TokenEntity>> {
        if !self.is_token_classification {
            return Err(E::msg(
                "Use classify_text for sequence classification models",
            ));
        }

        // Set up tokenizer with offset mapping for span reconstruction
        let mut tokenizer = self.tokenizer.clone();

        // Set up padding and enable offset mapping
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                pad_id: self.pad_token_id,
                ..Default::default()
            }))
            .with_truncation(None)
            .map_err(E::msg)?;

        // Tokenize input text with offset mapping
        let tokens = tokenizer.encode_batch(vec![text], true).map_err(E::msg)?;
        let token_encoding = &tokens[0];

        // Get offset mapping for span reconstruction
        let offsets = token_encoding.get_offsets();

        // Create tensors - convert to u32 for ModernBERT
        let token_ids = {
            let tokens: Vec<u32> = token_encoding.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?
        };

        let attention_mask = {
            let tokens: Vec<u32> = token_encoding.get_attention_mask().to_vec();
            Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?
        };

        // Input validation
        if token_ids.dims().len() != 2 {
            return Err(E::msg(format!(
                "Expected token_ids to have 2 dimensions [batch_size, seq_len], got {:?}",
                token_ids.dims()
            )));
        }
        if attention_mask.dims().len() != 2 {
            return Err(E::msg(format!(
                "Expected attention_mask to have 2 dimensions [batch_size, seq_len], got {:?}",
                attention_mask.dims()
            )));
        }

        // Run through ModernBERT token classification model
        let logits = match &self.model {
            ModernBertModel::Token(model) => model.forward(&token_ids, &attention_mask)?,
            ModernBertModel::Sequence(_) => {
                return Err(E::msg(
                    "Internal error: sequence model in token classification",
                ))
            }
        };

        // Apply softmax to get probabilities for each token position
        let probabilities = ops::softmax(&logits, D::Minus1)?;

        // Remove batch dimension
        let probabilities = probabilities.squeeze(0)?;
        let logits = logits.squeeze(0)?;

        // Get predictions for each token
        let predictions = logits.argmax(D::Minus1)?;

        // Convert to vectors for processing
        let predictions_vec = predictions.to_vec1::<u32>()?;
        let probabilities_2d = probabilities.to_vec2::<f32>()?;

        // Extract entities from BIO tags
        let mut entities = Vec::new();
        let mut current_entity: Option<TokenEntity> = None;

        for (i, (&pred_id, offset)) in predictions_vec.iter().zip(offsets.iter()).enumerate() {
            // Skip special tokens (they have offset (0,0))
            if offset.0 == 0 && offset.1 == 0 && i > 0 {
                continue;
            }

            // Get label from prediction ID
            let label = id2label
                .get(&pred_id.to_string())
                .unwrap_or(&"O".to_string())
                .clone();
            let confidence = probabilities_2d[i][pred_id as usize];

            if label.starts_with("B-") {
                // Beginning of new entity
                if let Some(entity) = current_entity.take() {
                    entities.push(entity);
                }

                let entity_type = label[2..].to_string(); // Remove 'B-' prefix
                current_entity = Some(TokenEntity {
                    entity_type,
                    start: offset.0,
                    end: offset.1,
                    text: text[offset.0..offset.1].to_string(),
                    confidence,
                });
            } else if let Some(entity_type) = label.strip_prefix("I-") {
                // Inside current entity
                if let Some(ref mut entity) = current_entity {
                    // Remove 'I-' prefix
                    if entity.entity_type == entity_type {
                        // Extend current entity
                        entity.end = offset.1;
                        entity.text = text[entity.start..entity.end].to_string();
                        // Update confidence with average
                        entity.confidence = (entity.confidence + confidence) / 2.0;
                    } else {
                        // Different entity type, finish current and don't start new
                        entities.push(entity.clone());
                        current_entity = None;
                    }
                } // If no current entity, ignore I- tag
            } else {
                // Outside entity (O tag or different entity type)
                if let Some(entity) = current_entity.take() {
                    entities.push(entity);
                }
            }
        }

        // Don't forget the last entity
        if let Some(entity) = current_entity {
            entities.push(entity);
        }

        Ok(entities)
    }
}

// Structure to hold token entity information
#[derive(Debug, Clone)]
pub struct TokenEntity {
    pub entity_type: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub confidence: f32,
}

// Initialize the ModernBERT classifier model (called from Go)
#[no_mangle]
pub extern "C" fn init_modernbert_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match ModernBertClassifier::new(model_id, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = MODERNBERT_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT classifier: {e}");
            false
        }
    }
}

// Initialize the ModernBERT PII classifier model (called from Go)
#[no_mangle]
pub extern "C" fn init_modernbert_pii_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match ModernBertClassifier::new(model_id, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = MODERNBERT_PII_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT PII classifier: {e}");
            false
        }
    }
}

// Initialize the ModernBERT PII token classifier model (called from Go)
#[no_mangle]
pub extern "C" fn init_modernbert_pii_token_classifier(
    model_id: *const c_char,
    use_cpu: bool,
) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match ModernBertClassifier::new_token_classification(model_id, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = MODERNBERT_PII_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT PII token classifier: {e}");
            false
        }
    }
}

// Initialize the ModernBERT jailbreak classifier model (called from Go)
#[no_mangle]
pub extern "C" fn init_modernbert_jailbreak_classifier(
    model_id: *const c_char,
    use_cpu: bool,
) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match ModernBertClassifier::new(model_id, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = MODERNBERT_JAILBREAK_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT jailbreak classifier: {e}");
            false
        }
    }
}

// Classify text using ModernBERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_modernbert_text(text: *const c_char) -> ModernBertClassificationResult {
    let default_result = ModernBertClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = MODERNBERT_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ModernBertClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying text with ModernBERT: {e}");
                default_result
            }
        },
        None => {
            eprintln!("ModernBERT classifier not initialized");
            default_result
        }
    }
}

// Classify text and return full probability distribution using ModernBERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_modernbert_text_with_probabilities(
    text: *const c_char,
) -> ModernBertClassificationResultWithProbs {
    let default_result = ModernBertClassificationResultWithProbs {
        class: -1,
        confidence: 0.0,
        probabilities: std::ptr::null_mut(),
        num_classes: 0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = MODERNBERT_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text_with_probs(text) {
            Ok((class_idx, confidence, probabilities)) => {
                // Allocate memory for probabilities array
                let prob_len = probabilities.len();
                let prob_ptr = Box::into_raw(probabilities.into_boxed_slice()) as *mut f32;

                ModernBertClassificationResultWithProbs {
                    class: class_idx as i32,
                    confidence,
                    probabilities: prob_ptr,
                    num_classes: prob_len as i32,
                }
            }
            Err(e) => {
                eprintln!("Error classifying text with probabilities using ModernBERT: {e}");
                default_result
            }
        },
        None => {
            eprintln!("ModernBERT classifier not initialized");
            default_result
        }
    }
}

// Free the probability array allocated by classify_modernbert_text_with_probabilities
#[no_mangle]
pub extern "C" fn free_modernbert_probabilities(probabilities: *mut f32, num_classes: i32) {
    if !probabilities.is_null() && num_classes > 0 {
        unsafe {
            let _: Box<[f32]> = Box::from_raw(std::slice::from_raw_parts_mut(
                probabilities,
                num_classes as usize,
            ));
        }
    }
}

// Classify text for PII using ModernBERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_modernbert_pii_text(
    text: *const c_char,
) -> ModernBertClassificationResult {
    let default_result = ModernBertClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = MODERNBERT_PII_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ModernBertClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying PII text with ModernBERT: {e}");
                default_result
            }
        },
        None => {
            eprintln!("ModernBERT PII classifier not initialized");
            default_result
        }
    }
}

// Classify text for jailbreak detection using ModernBERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_modernbert_jailbreak_text(
    text: *const c_char,
) -> ModernBertClassificationResult {
    let default_result = ModernBertClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = MODERNBERT_JAILBREAK_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ModernBertClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying jailbreak text with ModernBERT: {e}");
                default_result
            }
        },
        None => {
            eprintln!("ModernBERT jailbreak classifier not initialized");
            default_result
        }
    }
}

// Helper function to create id2label mapping from config
fn load_id2label_from_config(config_path: &str) -> Result<HashMap<String, String>> {
    let config_str = std::fs::read_to_string(config_path)?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)?;

    // Try to get id2label from classifier_config first
    if let Some(classifier_config) = config_json.get("classifier_config") {
        if let Some(id2label) = classifier_config
            .get("id2label")
            .and_then(|v| v.as_object())
        {
            let id2label_map: HashMap<String, String> = id2label
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("UNKNOWN").to_string()))
                .collect();
            return Ok(id2label_map);
        }
    }

    // Fall back to top-level id2label
    if let Some(id2label) = config_json.get("id2label").and_then(|v| v.as_object()) {
        let id2label_map: HashMap<String, String> = id2label
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("UNKNOWN").to_string()))
            .collect();
        return Ok(id2label_map);
    }

    Err(E::msg("No id2label mapping found in config"))
}

// Classify text for PII token classification using ModernBERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_modernbert_pii_tokens(
    text: *const c_char,
    model_config_path: *const c_char,
) -> ModernBertTokenClassificationResult {
    let default_result = ModernBertTokenClassificationResult {
        entities: std::ptr::null_mut(),
        num_entities: -1,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let config_path = unsafe {
        match CStr::from_ptr(model_config_path).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    // Load id2label mapping from config
    let id2label = match load_id2label_from_config(config_path) {
        Ok(mapping) => mapping,
        Err(e) => {
            eprintln!("Error loading id2label mapping: {e}");
            return default_result;
        }
    };

    let bert_opt = MODERNBERT_PII_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_tokens(text, &id2label) {
            Ok(entities) => {
                // Convert Rust entities to C-compatible format
                let num_entities = entities.len() as i32;
                if num_entities == 0 {
                    return ModernBertTokenClassificationResult {
                        entities: std::ptr::null_mut(),
                        num_entities: 0,
                    };
                }

                // Allocate memory for entities array
                let entities_ptr = unsafe {
                    libc::malloc(
                        num_entities as usize * std::mem::size_of::<ModernBertTokenEntity>(),
                    ) as *mut ModernBertTokenEntity
                };

                if entities_ptr.is_null() {
                    eprintln!("Failed to allocate memory for entities");
                    return default_result;
                }

                // Fill the entities array
                for (i, entity) in entities.iter().enumerate() {
                    let entity_type_cstr =
                        std::ffi::CString::new(entity.entity_type.clone()).unwrap_or_default();
                    let text_cstr = std::ffi::CString::new(entity.text.clone()).unwrap_or_default();

                    unsafe {
                        (*entities_ptr.add(i)) = ModernBertTokenEntity {
                            entity_type: entity_type_cstr.into_raw(),
                            start: entity.start as i32,
                            end: entity.end as i32,
                            text: text_cstr.into_raw(),
                            confidence: entity.confidence,
                        };
                    }
                }

                ModernBertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities,
                }
            }
            Err(e) => {
                eprintln!("Error classifying PII tokens with ModernBERT: {e}");
                default_result
            }
        },
        None => {
            eprintln!("ModernBERT PII classifier not initialized");
            default_result
        }
    }
}

// Free memory allocated for token classification results (called from Go)
#[no_mangle]
pub extern "C" fn free_modernbert_token_result(result: ModernBertTokenClassificationResult) {
    if result.entities.is_null() || result.num_entities <= 0 {
        return;
    }

    unsafe {
        // Free individual strings in each entity
        for i in 0..result.num_entities {
            let entity = &*result.entities.add(i as usize);
            if !entity.entity_type.is_null() {
                let _ = std::ffi::CString::from_raw(entity.entity_type);
            }
            if !entity.text.is_null() {
                let _ = std::ffi::CString::from_raw(entity.text);
            }
        }

        // Free the entities array
        libc::free(result.entities as *mut libc::c_void);
    }
}
