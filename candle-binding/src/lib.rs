// This file is a binding for the candle-core and candle-transformers libraries.
// It is based on https://github.com/huggingface/candle/tree/main/candle-examples/examples/bert
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

pub mod bert_official;
pub mod modernbert;
pub mod unified_classifier;

// Re-export ModernBERT functions and structures
pub use modernbert::{
    classify_modernbert_jailbreak_text, classify_modernbert_pii_text, classify_modernbert_text,
    init_modernbert_classifier, init_modernbert_jailbreak_classifier,
    init_modernbert_pii_classifier, ModernBertClassificationResult,
};

// Re-export unified classifier functions and structures
pub use unified_classifier::{
    get_unified_classifier, BatchClassificationResult, IntentResult, PIIResult, SecurityResult,
    UnifiedClassificationResult, UnifiedClassifier, UNIFIED_CLASSIFIER,
};

use crate::bert_official::{CandleBertClassifier, CandleBertTokenClassifier};
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops, Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use tokenizers::TruncationDirection;
use tokenizers::TruncationParams;
use tokenizers::TruncationStrategy;

// Structure to hold BERT model and tokenizer for semantic similarity
pub struct BertSimilarity {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

// Structure to hold BERT model, tokenizer, and classification head for text classification
pub struct BertClassifier {
    model: CandleBertClassifier,
}

// ================================================================================================
// BERT TOKEN CLASSIFICATION IMPLEMENTATION
// ================================================================================================
// Following ModernBERT's design pattern for token-level classification

/// BERT token classifier for token-level predictions (e.g., NER, PII detection)
pub struct BertForTokenClassification {
    bert: BertModel,
    dropout: Option<candle_nn::Dropout>,
    classifier: Linear,
}

impl BertForTokenClassification {
    pub fn load(vb: VarBuilder, config: &Config, num_classes: usize) -> Result<Self> {
        let bert = BertModel::load(vb.clone(), config)?;

        // Create dropout layer (optional, based on config)
        let dropout = if config.hidden_dropout_prob > 0.0 {
            Some(candle_nn::Dropout::new(config.hidden_dropout_prob as f32))
        } else {
            None
        };

        // Create token classification head
        let classifier = candle_nn::Linear::new(
            vb.get((num_classes, config.hidden_size), "classifier.weight")?,
            Some(vb.get((num_classes,), "classifier.bias")?),
        );

        Ok(Self {
            bert,
            dropout,
            classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Get sequence output from BERT (all token representations)
        let sequence_output = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;

        // Apply dropout if configured
        let sequence_output = match &self.dropout {
            Some(dropout) => dropout.forward(&sequence_output, true).map_err(E::msg)?,
            None => sequence_output,
        };

        // Apply token classification head to get logits for each token
        Ok(sequence_output.apply(&self.classifier)?)
    }
}

/// Enum to hold different types of BERT models (following ModernBERT pattern)
pub enum BertModelType {
    Sequence(BertClassifier),
    Token(BertForTokenClassification),
}

/// Structure to hold token entity result (compatible with ModernBERT format)
#[repr(C)]
pub struct BertTokenEntity {
    pub entity_type: *mut c_char,
    pub start: i32,
    pub end: i32,
    pub text: *mut c_char,
    pub confidence: f32,
}

/// Structure to hold token classification result (array of entities)
#[repr(C)]
pub struct BertTokenClassificationResult {
    pub entities: *mut BertTokenEntity,
    pub num_entities: i32,
}

/// Enhanced BertClassifier that supports both sequence and token classification
pub struct UniversalBertClassifier {
    model: BertModelType,
    tokenizer: Tokenizer,
    device: Device,
}

impl UniversalBertClassifier {
    pub fn new_sequence_classification(
        model_id: &str,
        num_classes: usize,
        use_cpu: bool,
    ) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Load the existing BertClassifier for sequence classification
        let bert_classifier = BertClassifier::new(model_id, num_classes, use_cpu)?;

        Ok(Self {
            model: BertModelType::Sequence(bert_classifier),
            tokenizer: Tokenizer::from_file(format!("{}/tokenizer.json", model_id))
                .map_err(E::msg)?,
            device,
        })
    }

    pub fn new_token_classification(
        model_id: &str,
        num_classes: usize,
        use_cpu: bool,
    ) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Load config and tokenizer
        let config_path = format!("{}/config.json", model_id);
        let tokenizer_path = format!("{}/tokenizer.json", model_id);

        let config = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        // Use approximate GELU for better performance
        // Keep original activation function to match PyTorch exactly

        // Load model weights
        let weights_path = if Path::new(model_id).join("model.safetensors").exists() {
            format!("{}/model.safetensors", model_id)
        } else if Path::new(model_id).join("pytorch_model.bin").exists() {
            format!("{}/pytorch_model.bin", model_id)
        } else {
            return Err(E::msg(format!("No model weights found in {}", model_id)));
        };

        let use_pth = weights_path.ends_with(".bin");
        let vb = if use_pth {
            VarBuilder::from_pth(&weights_path, DType::F32, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? }
        };

        // Create token classification model
        let bert_token_classifier = BertForTokenClassification::load(vb, &config, num_classes)?;

        Ok(Self {
            model: BertModelType::Token(bert_token_classifier),
            tokenizer,
            device,
        })
    }

    /// Classify text for sequence classification
    pub fn classify_text(&self, text: &str) -> Result<(usize, f32)> {
        match &self.model {
            BertModelType::Sequence(classifier) => classifier.classify_text(text),
            BertModelType::Token(_) => Err(E::msg(
                "This model is configured for token classification, not sequence classification",
            )),
        }
    }

    /// Classify tokens for token classification (returns entities)
    pub fn classify_tokens(
        &self,
        text: &str,
        id2label: &HashMap<usize, String>,
    ) -> Result<Vec<TokenEntity>> {
        match &self.model {
            BertModelType::Token(classifier) => {
                // Tokenize input
                let encoding = self.tokenizer.encode(text, true).map_err(E::msg)?;
                let token_ids = encoding.get_ids().to_vec();
                let attention_mask = encoding.get_attention_mask().to_vec();
                let tokens = encoding.get_tokens().to_vec();

                // Create tensors
                let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
                let attention_mask_tensor =
                    Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;
                let token_type_ids = token_ids_tensor.zeros_like()?;

                // Get predictions
                let logits = classifier.forward(
                    &token_ids_tensor,
                    &token_type_ids,
                    Some(&attention_mask_tensor),
                )?;

                // Apply softmax to get probabilities
                let probabilities = ops::softmax(&logits, D::Minus1)?;

                // Extract entities from predictions
                self.extract_entities_from_predictions(&probabilities, &tokens, text, id2label)
            }
            BertModelType::Sequence(_) => Err(E::msg(
                "This model is configured for sequence classification, not token classification",
            )),
        }
    }

    /// Extract entities from token classification predictions
    fn extract_entities_from_predictions(
        &self,
        probabilities: &Tensor,
        tokens: &[String],
        original_text: &str,
        id2label: &HashMap<usize, String>,
    ) -> Result<Vec<TokenEntity>> {
        let probs_data = probabilities.squeeze(0)?.to_vec2::<f32>()?;
        let mut entities = Vec::new();
        let mut current_entity: Option<(String, usize, f32)> = None;

        for (token_idx, (token, token_probs)) in tokens.iter().zip(probs_data.iter()).enumerate() {
            // Skip special tokens
            if token.starts_with("[") && token.ends_with("]") {
                continue;
            }

            // Find the predicted class (highest probability)
            let (pred_class, confidence) = token_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, &prob)| (idx, prob))
                .unwrap_or((0, 0.0));

            let label = id2label
                .get(&pred_class)
                .unwrap_or(&"O".to_string())
                .clone();

            // Handle BIO tagging
            if label.starts_with("B-") {
                // Begin new entity
                if let Some((entity_type, start_idx, _)) = current_entity.take() {
                    // Finish previous entity
                    entities.push(TokenEntity {
                        entity_type,
                        start: start_idx as i32,
                        end: token_idx as i32,
                        text: self.extract_text_span(original_text, start_idx, token_idx)?,
                        confidence,
                    });
                }
                current_entity = Some((label[2..].to_string(), token_idx, confidence));
            } else if label.starts_with("I-") && current_entity.is_some() {
                // Continue current entity (update confidence if lower)
                if let Some((_, _, ref mut entity_confidence)) = current_entity {
                    *entity_confidence = entity_confidence.min(confidence);
                }
            } else {
                // "O" tag or end of entity
                if let Some((entity_type, start_idx, entity_confidence)) = current_entity.take() {
                    entities.push(TokenEntity {
                        entity_type,
                        start: start_idx as i32,
                        end: token_idx as i32,
                        text: self.extract_text_span(original_text, start_idx, token_idx)?,
                        confidence: entity_confidence,
                    });
                }
            }
        }

        // Handle any remaining entity
        if let Some((entity_type, start_idx, entity_confidence)) = current_entity {
            entities.push(TokenEntity {
                entity_type,
                start: start_idx as i32,
                end: tokens.len() as i32,
                text: self.extract_text_span(original_text, start_idx, tokens.len())?,
                confidence: entity_confidence,
            });
        }

        Ok(entities)
    }

    /// Extract text span from original text based on token positions
    fn extract_text_span(
        &self,
        _text: &str,
        start_token: usize,
        end_token: usize,
    ) -> Result<String> {
        // This is a simplified implementation
        // In practice, you'd need proper token-to-character mapping
        Ok(format!("entity_{}_{}", start_token, end_token))
    }
}

/// Token entity structure for compatibility
pub struct TokenEntity {
    pub entity_type: String,
    pub start: i32,
    pub end: i32,
    pub text: String,
    pub confidence: f32,
}

// ================================================================================================
// END OF BERT TOKEN CLASSIFICATION IMPLEMENTATION
// ================================================================================================

lazy_static::lazy_static! {
    static ref BERT_SIMILARITY: Arc<Mutex<Option<BertSimilarity>>> = Arc::new(Mutex::new(None));
    static ref BERT_CLASSIFIER: Arc<Mutex<Option<BertClassifier>>> = Arc::new(Mutex::new(None));
    static ref BERT_PII_CLASSIFIER: Arc<Mutex<Option<BertClassifier>>> = Arc::new(Mutex::new(None));
    static ref BERT_JAILBREAK_CLASSIFIER: Arc<Mutex<Option<BertClassifier>>> = Arc::new(Mutex::new(None));
}

// Structure to hold tokenization result
#[repr(C)]
pub struct TokenizationResult {
    pub token_ids: *mut i32,
    pub token_count: i32,
    pub tokens: *mut *mut c_char,
    pub error: bool,
}

impl BertSimilarity {
    pub fn new(model_id: &str, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Default to a sentence transformer model if not specified or empty
        let model_id = if model_id.is_empty() {
            "sentence-transformers/all-MiniLM-L6-v2"
        } else {
            model_id
        };

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
                // HuggingFace Hub model
                let repo =
                    Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());

                let api = Api::new()?;
                let api = api.repo(repo);
                let config = api.get("config.json")?;
                let tokenizer = api.get("tokenizer.json")?;

                // Try to get safetensors first, if that fails, fall back to pytorch_model.bin. This is for BAAI models
                // create a special case for BAAI to download the correct weights to avoid downloading the wrong weights
                let (weights, use_pth) = if model_id.starts_with("BAAI/") {
                    // BAAI models typically use PyTorch model format
                    (api.get("pytorch_model.bin")?, true)
                } else {
                    match api.get("model.safetensors") {
                        Ok(weights) => (weights, false),
                        Err(_) => {
                            println!(
                                "Safetensors model not found, trying PyTorch model instead..."
                            );
                            (api.get("pytorch_model.bin")?, true)
                        }
                    }
                };

                (
                    config.to_string_lossy().to_string(),
                    tokenizer.to_string_lossy().to_string(),
                    weights.to_string_lossy().to_string(),
                    use_pth,
                )
            };

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Use the approximate GELU for better performance
        // Keep original activation function to match PyTorch exactly

        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[weights_filename.clone()],
                    DType::F32,
                    &device,
                )?
            }
        };

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    // Tokenize a text string
    pub fn tokenize_text(
        &self,
        text: &str,
        max_length: Option<usize>,
    ) -> Result<(Vec<i32>, Vec<String>)> {
        // Encode the text with the tokenizer
        let mut tokenizer = self.tokenizer.clone();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_length.unwrap_or(512),
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(E::msg)?;

        let encoding = tokenizer.encode(text, true).map_err(E::msg)?;

        // Get token IDs and tokens
        let token_ids = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let tokens = encoding.get_tokens().to_vec();

        Ok((token_ids, tokens))
    }

    // Get embedding for a text
    pub fn get_embedding(&self, text: &str, max_length: Option<usize>) -> Result<Tensor> {
        // Encode the text with the tokenizer
        let mut tokenizer = self.tokenizer.clone();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_length.unwrap_or(512),
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(E::msg)?;

        let encoding = tokenizer.encode(text, true).map_err(E::msg)?;

        // Get token IDs and attention mask
        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();

        // Create tensors
        let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids_tensor.zeros_like()?;

        // Run the text through BERT with attention mask
        let embeddings = self.model.forward(
            &token_ids_tensor,
            &token_type_ids,
            Some(&attention_mask_tensor),
        )?;

        // Mean pooling: sum over tokens and divide by attention mask sum
        let sum_embeddings = embeddings.sum(1)?;
        let attention_sum = attention_mask_tensor.sum(1)?.to_dtype(embeddings.dtype())?;
        let pooled = sum_embeddings.broadcast_div(&attention_sum)?;

        // Convert to float32 and normalize
        let embedding = pooled.to_dtype(DType::F32)?;

        normalize_l2(&embedding)
    }

    // Calculate cosine similarity between two texts
    pub fn calculate_similarity(
        &self,
        text1: &str,
        text2: &str,
        max_length: Option<usize>,
    ) -> Result<f32> {
        let embedding1 = self.get_embedding(text1, max_length)?;
        let embedding2 = self.get_embedding(text2, max_length)?;

        // For normalized vectors, dot product equals cosine similarity
        let dot_product = embedding1.matmul(&embedding2.transpose(0, 1)?)?;

        // Extract the scalar value from the result
        let sim_value = dot_product.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;

        Ok(sim_value)
    }

    // Find most similar text from a list
    pub fn find_most_similar(
        &self,
        query_text: &str,
        candidates: &[&str],
        max_length: Option<usize>,
    ) -> Result<(usize, f32)> {
        if candidates.is_empty() {
            return Err(E::msg("Empty candidate list"));
        }

        let query_embedding = self.get_embedding(query_text, max_length)?;

        // Calculate similarity for each candidate individually
        let mut best_idx = 0;
        let mut best_score = -1.0;

        for (idx, candidate) in candidates.iter().enumerate() {
            let candidate_embedding = self.get_embedding(candidate, max_length)?;

            // Calculate similarity (dot product of normalized vectors = cosine similarity)
            let sim = query_embedding.matmul(&candidate_embedding.transpose(0, 1)?)?;
            let score = sim.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;

            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        Ok((best_idx, best_score))
    }
}

impl BertClassifier {
    pub fn new(model_id: &str, num_classes: usize, use_cpu: bool) -> Result<Self> {
        let model = CandleBertClassifier::new(model_id, num_classes, use_cpu)?;
        Ok(Self { model })
    }

    pub fn classify_text(&self, text: &str) -> Result<(usize, f32)> {
        self.model.classify_text(text)
    }

    pub fn classify_text_with_probs(&self, text: &str) -> Result<(usize, f32, Vec<f32>)> {
        // For now, the new BERT implementation doesn't return full probabilities
        // Return the classification result with empty probabilities
        let (class_idx, confidence) = self.model.classify_text(text)?;
        Ok((class_idx, confidence, vec![]))
    }
}

// Old implementation - to be removed
pub struct BertClassifierOld {
    model: BertModel,
    tokenizer: Tokenizer,
    classification_head: Linear,
    pooler: Option<Linear>,
    num_classes: usize,
    device: Device,
}

impl BertClassifierOld {
    pub fn new_old(model_id: &str, num_classes: usize, use_cpu: bool) -> Result<Self> {
        if num_classes < 2 {
            return Err(E::msg(format!(
                "Number of classes must be at least 2, got {num_classes}"
            )));
        }

        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        println!("Initializing classifier model: {model_id}");

        // Check if this is a SentenceTransformer linear classifier model
        let is_sentence_transformer = Path::new(model_id).join("modules.json").exists();

        if is_sentence_transformer {}

        let (config_filename, tokenizer_filename, weights_filename, use_pth) =
            if Path::new(model_id).exists() {
                // Local model path
                let config_path = Path::new(model_id).join("config.json");
                let tokenizer_path = Path::new(model_id).join("tokenizer.json");

                // For SentenceTransformer models, check both the root and 0_Transformer
                let weights_path = if is_sentence_transformer {
                    // First check if model weights are at the root level (most common for sentence-transformers)
                    if Path::new(model_id).join("model.safetensors").exists() {
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
                    }
                    // Otherwise check if there's a 0_Transformer directory
                    else {
                        let transformer_path = Path::new(model_id).join("0_Transformer");
                        if transformer_path.exists() {
                            if transformer_path.join("model.safetensors").exists() {
                                (
                                    transformer_path
                                        .join("model.safetensors")
                                        .to_string_lossy()
                                        .to_string(),
                                    false,
                                )
                            } else if transformer_path.join("pytorch_model.bin").exists() {
                                (
                                    transformer_path
                                        .join("pytorch_model.bin")
                                        .to_string_lossy()
                                        .to_string(),
                                    true,
                                )
                            } else {
                                return Err(E::msg(format!(
                                    "No transformer model weights found in {}",
                                    transformer_path.display()
                                )));
                            }
                        } else {
                            return Err(E::msg(format!("No model weights found in {model_id}")));
                        }
                    }
                } else if Path::new(model_id).join("model.safetensors").exists() {
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
                // HuggingFace Hub model
                let repo =
                    Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());

                let api = Api::new()?;
                let api = api.repo(repo);
                let config = api.get("config.json")?;
                let tokenizer = api.get("tokenizer.json")?;

                // Try safetensors first, fall back to PyTorch
                let (weights, use_pth) = match api.get("model.safetensors") {
                    Ok(weights) => (weights, false),
                    Err(_) => {
                        println!("Safetensors model not found, trying PyTorch model instead...");
                        (api.get("pytorch_model.bin")?, true)
                    }
                };

                (
                    config.to_string_lossy().to_string(),
                    tokenizer.to_string_lossy().to_string(),
                    weights.to_string_lossy().to_string(),
                    use_pth,
                )
            };

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Use approximate GELU for better performance
        // Keep original activation function to match PyTorch exactly

        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[weights_filename.clone()],
                    DType::F32,
                    &device,
                )?
            }
        };

        let model = BertModel::load(vb.clone(), &config)?;

        // Create a classification head
        // For SentenceTransformer models, we need to load the Dense layer weights from 2_Dense
        let (w, b) = if is_sentence_transformer {
            // Load the dense layer weights from 2_Dense
            let dense_dir = Path::new(model_id).join("2_Dense");

            let dense_config_path = dense_dir.join("config.json");

            if dense_config_path.exists() {
                println!("Found dense config at {}", dense_config_path.display());
                let dense_config = std::fs::read_to_string(dense_config_path)?;
                let dense_config: serde_json::Value = serde_json::from_str(&dense_config)?;

                // Get dimensions from the config
                let in_features = dense_config["in_features"].as_i64().unwrap_or(768) as usize;
                let out_features = dense_config["out_features"]
                    .as_i64()
                    .unwrap_or(num_classes as i64) as usize;

                println!(
                    "Dense layer dimensions: in_features={in_features}, out_features={out_features}"
                );

                // Try to load dense weights from safetensors or pytorch files
                let weights_path = if dense_dir.join("model.safetensors").exists() {
                    (
                        dense_dir
                            .join("model.safetensors")
                            .to_string_lossy()
                            .to_string(),
                        false,
                    )
                } else if dense_dir.join("pytorch_model.bin").exists() {
                    (
                        dense_dir
                            .join("pytorch_model.bin")
                            .to_string_lossy()
                            .to_string(),
                        true,
                    )
                } else {
                    return Err(E::msg(format!(
                        "No dense layer weights found in {}",
                        dense_dir.display()
                    )));
                };

                // Load the weights
                let dense_vb = if weights_path.1 {
                    VarBuilder::from_pth(&weights_path.0, DType::F32, &device)?
                } else {
                    unsafe {
                        VarBuilder::from_mmaped_safetensors(&[weights_path.0], DType::F32, &device)?
                    }
                };

                // Get the weight and bias tensors - PyTorch uses [out_features, in_features] format
                let weight = dense_vb.get((out_features, in_features), "linear.weight")?;
                // Transpose the weight matrix to match our expected format [in_features, out_features]
                let weight = weight.t()?;
                let bias = dense_vb.get(out_features, "linear.bias")?;

                (weight, bias)
            } else {
                // Fallback: create random weights as before
                println!("No dense config found, using random weights");
                let hidden_size = config.hidden_size;
                let w = Tensor::randn(0.0, 0.02, (hidden_size, num_classes), &device)?;
                let b = Tensor::zeros((num_classes,), DType::F32, &device)?;
                (w, b)
            }
        } else {
            // Regular BERT model: try to load classifier weights from main model file
            println!("Loading classifier weights from main BERT model file");

            // Load the main model weights
            let model_vb = if use_pth {
                VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
            } else {
                unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[weights_filename.clone()],
                        DType::F32,
                        &device,
                    )?
                }
            };

            // Try to load classifier weights - different models may use different names
            let classifier_weight_result = model_vb
                .get((num_classes, config.hidden_size), "classifier.weight")
                .or_else(|_| {
                    model_vb.get(
                        (num_classes, config.hidden_size),
                        "cls.predictions.decoder.weight",
                    )
                })
                .or_else(|_| {
                    model_vb.get(
                        (num_classes, config.hidden_size),
                        "classification_head.weight",
                    )
                });

            let classifier_bias_result = model_vb
                .get(num_classes, "classifier.bias")
                .or_else(|_| model_vb.get(num_classes, "cls.predictions.decoder.bias"))
                .or_else(|_| model_vb.get(num_classes, "classification_head.bias"));

            match (classifier_weight_result, classifier_bias_result) {
                (Ok(weight), Ok(bias)) => {
                    // PyTorch uses [out_features, in_features] format, transpose to [in_features, out_features]
                    let weight = weight.t()?;
                    (weight, bias)
                }
                _ => {
                    println!("Classifier weights not found in main model, using random weights");
                    let hidden_size = config.hidden_size;
                    let w = Tensor::randn(0.0, 0.02, (hidden_size, num_classes), &device)?;
                    let b = Tensor::zeros((num_classes,), DType::F32, &device)?;
                    (w, b)
                }
            }
        };

        let classification_head = Linear::new(w, Some(b));

        // Load pooler weights for sequence classification
        let pooler = {
            let model_vb = if use_pth {
                VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
            } else {
                unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[weights_filename.clone()],
                        DType::F32,
                        &device,
                    )?
                }
            };

            let pooler_weight_result = model_vb.get(
                (config.hidden_size, config.hidden_size),
                "bert.pooler.dense.weight",
            );
            let pooler_bias_result = model_vb.get(config.hidden_size, "bert.pooler.dense.bias");

            match (pooler_weight_result, pooler_bias_result) {
                (Ok(pooler_weight), Ok(pooler_bias)) => {
                    // PyTorch uses [out_features, in_features], transpose to [in_features, out_features]
                    let pooler_weight = pooler_weight.t()?;
                    Some(Linear::new(pooler_weight, Some(pooler_bias)))
                }
                _ => {
                    println!("Pooler weights not found, will use CLS token directly");
                    None
                }
            }
        };

        Ok(Self {
            model,
            tokenizer,
            classification_head,
            pooler,
            num_classes,
            device,
        })
    }

    pub fn classify_text(&self, text: &str) -> Result<(usize, f32)> {
        // Encode the text with the tokenizer
        let encoding = self.tokenizer.encode(text, true).map_err(E::msg)?;

        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();

        let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids_tensor.zeros_like()?;
        let attention_mask_tensor = Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;

        // Run the text through BERT
        let embeddings = self.model.forward(
            &token_ids_tensor,
            &token_type_ids,
            Some(&attention_mask_tensor),
        )?;

        // For sequence classification, use BERT pooler output (CLS token + linear + tanh)
        // Extract the [CLS] token embedding (index 0)
        let cls_token = embeddings.i((.., 0))?.to_dtype(DType::F32)?;

        // Apply BERT pooler if available
        let pooled_embedding = match &self.pooler {
            Some(pooler) => {
                // Apply pooler: linear transformation + tanh activation
                let pooler_output = cls_token.apply(pooler)?;
                pooler_output.tanh()?
            }
            None => {
                // Fallback to CLS token directly
                cls_token
            }
        };

        // Apply the linear layer (classification head) manually
        let weights = self.classification_head.weight().to_dtype(DType::F32)?;
        let bias = self
            .classification_head
            .bias()
            .unwrap()
            .to_dtype(DType::F32)?;

        // Use matmul with the weights matrix
        // Weights are already in the correct shape [768, 2] for input [1, 768]
        let logits = pooled_embedding.matmul(&weights)?;

        // Add bias
        let logits = logits.broadcast_add(&bias)?;

        // If logits has shape [1, num_classes], squeeze it to get [num_classes]
        let logits = if logits.dims().len() > 1 {
            logits.squeeze(0)?
        } else {
            logits
        };

        // Apply softmax to get probabilities
        let logits_vec = logits.to_vec1::<f32>()?;
        let max_logit = logits_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = logits_vec.iter().map(|&x| (x - max_logit).exp()).collect();
        let exp_sum: f32 = exp_values.iter().sum();
        let probabilities: Vec<f32> = exp_values.iter().map(|&x| x / exp_sum).collect();

        // Get the predicted class with highest probability
        let (predicted_idx, &max_prob) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        // Ensure we don't return a class index outside our expected range
        if predicted_idx >= self.num_classes {
            return Err(E::msg(format!(
                "Invalid class index: {} (num_classes: {})",
                predicted_idx, self.num_classes
            )));
        }

        Ok((predicted_idx, max_prob))
    }

    // Classify text and return full probability distribution
    pub fn classify_text_with_probs(&self, text: &str) -> Result<(usize, f32, Vec<f32>)> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let token_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let position_ids = Tensor::arange(0, tokens.len() as i64, &self.device)?
            .unsqueeze(0)?
            .to_dtype(candle_core::DType::U32)?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&position_ids))?;

        // Pool embeddings (mean pooling)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = embeddings.sum(1)?;
        let pooled_embedding = (embeddings / (n_tokens as f64))?;

        // Get classification head weights and bias
        let weights = self.classification_head.weight();
        let bias = self.classification_head.bias().unwrap();

        // Apply classification head
        // If weights are already transposed to [in_features, out_features]
        let logits = pooled_embedding.matmul(&weights)?;

        // Add bias
        let logits = logits.broadcast_add(&bias)?;

        // If logits has shape [1, num_classes], squeeze it to get [num_classes]
        let logits = if logits.dims().len() > 1 {
            logits.squeeze(0)?
        } else {
            logits
        };

        // Apply softmax to get probabilities
        let logits_vec = logits.to_vec1::<f32>()?;
        let max_logit = logits_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = logits_vec.iter().map(|&x| (x - max_logit).exp()).collect();
        let exp_sum: f32 = exp_values.iter().sum();
        let probabilities: Vec<f32> = exp_values.iter().map(|&x| x / exp_sum).collect();

        // Get the predicted class with highest probability
        let (predicted_idx, &max_prob) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        // Ensure we don't return a class index outside our expected range
        if predicted_idx >= self.num_classes {
            return Err(E::msg(format!(
                "Invalid class index: {} (num_classes: {})",
                predicted_idx, self.num_classes
            )));
        }

        Ok((predicted_idx, max_prob, probabilities))
    }
}

// Tokenize text (called from Go)
#[no_mangle]
pub extern "C" fn tokenize_text(text: *const c_char, max_length: i32) -> TokenizationResult {
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                return TokenizationResult {
                    token_ids: std::ptr::null_mut(),
                    token_count: 0,
                    tokens: std::ptr::null_mut(),
                    error: true,
                }
            }
        }
    };

    let bert_opt = BERT_SIMILARITY.lock().unwrap();
    let bert = match &*bert_opt {
        Some(b) => b,
        None => {
            eprintln!("BERT model not initialized");
            return TokenizationResult {
                token_ids: std::ptr::null_mut(),
                token_count: 0,
                tokens: std::ptr::null_mut(),
                error: true,
            };
        }
    };

    let max_length_opt = if max_length <= 0 {
        None
    } else {
        Some(max_length as usize)
    };
    match bert.tokenize_text(text, max_length_opt) {
        Ok((token_ids, tokens)) => {
            let count = token_ids.len() as i32;

            // Allocate memory for token IDs
            let ids_ptr = token_ids.as_ptr() as *mut i32;

            // Allocate memory for tokens
            let c_tokens: Vec<*mut c_char> = tokens
                .iter()
                .map(|s| CString::new(s.as_str()).unwrap().into_raw())
                .collect();

            let tokens_ptr = c_tokens.as_ptr() as *mut *mut c_char;

            // Don't drop the vectors - Go will own the memory now
            std::mem::forget(token_ids);
            std::mem::forget(c_tokens);

            TokenizationResult {
                token_ids: ids_ptr,
                token_count: count,
                tokens: tokens_ptr,
                error: false,
            }
        }
        Err(e) => {
            eprintln!("Error tokenizing text: {e}");
            TokenizationResult {
                token_ids: std::ptr::null_mut(),
                token_count: 0,
                tokens: std::ptr::null_mut(),
                error: true,
            }
        }
    }
}

// Free tokenization result allocated by Rust
#[no_mangle]
pub extern "C" fn free_tokenization_result(result: TokenizationResult) {
    if !result.token_ids.is_null() && result.token_count > 0 {
        unsafe {
            // Reconstruct and drop the token_ids vector
            let _ids_vec = Vec::from_raw_parts(
                result.token_ids,
                result.token_count as usize,
                result.token_count as usize,
            );

            // Reconstruct and drop each token string
            if !result.tokens.is_null() {
                let tokens_slice =
                    std::slice::from_raw_parts(result.tokens, result.token_count as usize);
                for &token_ptr in tokens_slice {
                    if !token_ptr.is_null() {
                        let _ = CString::from_raw(token_ptr);
                    }
                }

                // Reconstruct and drop the tokens vector
                let _tokens_vec = Vec::from_raw_parts(
                    result.tokens,
                    result.token_count as usize,
                    result.token_count as usize,
                );
            }
        }
    }
}

// Initialize the BERT model (called from Go)
#[no_mangle]
pub extern "C" fn init_similarity_model(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match BertSimilarity::new(model_id, use_cpu) {
        Ok(model) => {
            let mut bert_opt = BERT_SIMILARITY.lock().unwrap();
            *bert_opt = Some(model);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT: {e}");
            false
        }
    }
}

// Structure to hold similarity result
#[repr(C)]
pub struct SimilarityResult {
    pub index: i32, // Index of the most similar text
    pub score: f32, // Similarity score
}

// Structure to hold embedding result
#[repr(C)]
pub struct EmbeddingResult {
    pub data: *mut f32,
    pub length: i32,
    pub error: bool,
}

// Get embedding for a text (called from Go)
#[no_mangle]
pub extern "C" fn get_text_embedding(text: *const c_char, max_length: i32) -> EmbeddingResult {
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                return EmbeddingResult {
                    data: std::ptr::null_mut(),
                    length: 0,
                    error: true,
                }
            }
        }
    };

    let bert_opt = BERT_SIMILARITY.lock().unwrap();
    let bert = match &*bert_opt {
        Some(b) => b,
        None => {
            eprintln!("BERT model not initialized");
            return EmbeddingResult {
                data: std::ptr::null_mut(),
                length: 0,
                error: true,
            };
        }
    };

    let max_length_opt = if max_length <= 0 {
        None
    } else {
        Some(max_length as usize)
    };
    match bert.get_embedding(text, max_length_opt) {
        Ok(embedding) => {
            match embedding.flatten_all() {
                Ok(flat_embedding) => {
                    match flat_embedding.to_vec1::<f32>() {
                        Ok(vec) => {
                            let length = vec.len() as i32;
                            // Allocate memory that will be freed by Go
                            let data = vec.as_ptr() as *mut f32;
                            std::mem::forget(vec); // Don't drop the vector - Go will own the memory now
                            EmbeddingResult {
                                data,
                                length,
                                error: false,
                            }
                        }
                        Err(_) => EmbeddingResult {
                            data: std::ptr::null_mut(),
                            length: 0,
                            error: true,
                        },
                    }
                }
                Err(_) => EmbeddingResult {
                    data: std::ptr::null_mut(),
                    length: 0,
                    error: true,
                },
            }
        }
        Err(e) => {
            eprintln!("Error getting embedding: {e}");
            EmbeddingResult {
                data: std::ptr::null_mut(),
                length: 0,
                error: true,
            }
        }
    }
}

// Calculate similarity between two texts (called from Go)
#[no_mangle]
pub extern "C" fn calculate_similarity(
    text1: *const c_char,
    text2: *const c_char,
    max_length: i32,
) -> f32 {
    let text1 = unsafe {
        match CStr::from_ptr(text1).to_str() {
            Ok(s) => s,
            Err(_) => return -1.0,
        }
    };

    let text2 = unsafe {
        match CStr::from_ptr(text2).to_str() {
            Ok(s) => s,
            Err(_) => return -1.0,
        }
    };

    let bert_opt = BERT_SIMILARITY.lock().unwrap();
    let bert = match &*bert_opt {
        Some(b) => b,
        None => {
            eprintln!("BERT model not initialized");
            return -1.0;
        }
    };

    let max_length_opt = if max_length <= 0 {
        None
    } else {
        Some(max_length as usize)
    };
    match bert.calculate_similarity(text1, text2, max_length_opt) {
        Ok(similarity) => similarity,
        Err(e) => {
            eprintln!("Error calculating similarity: {e}");
            -1.0
        }
    }
}

// Find most similar text from a list (called from Go)
#[no_mangle]
pub extern "C" fn find_most_similar(
    query: *const c_char,
    candidates_ptr: *const *const c_char,
    num_candidates: i32,
    max_length: i32,
) -> SimilarityResult {
    let query = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s,
            Err(_) => {
                return SimilarityResult {
                    index: -1,
                    score: -1.0,
                }
            }
        }
    };

    // Convert the array of C strings to Rust strings
    let candidates: Vec<&str> = unsafe {
        let mut result = Vec::with_capacity(num_candidates as usize);
        let candidates_slice = std::slice::from_raw_parts(candidates_ptr, num_candidates as usize);

        for &cstr in candidates_slice {
            match CStr::from_ptr(cstr).to_str() {
                Ok(s) => result.push(s),
                Err(_) => {
                    return SimilarityResult {
                        index: -1,
                        score: -1.0,
                    }
                }
            }
        }

        result
    };

    let bert_opt = BERT_SIMILARITY.lock().unwrap();
    let bert = match &*bert_opt {
        Some(b) => b,
        None => {
            eprintln!("BERT model not initialized");
            return SimilarityResult {
                index: -1,
                score: -1.0,
            };
        }
    };

    let max_length_opt = if max_length <= 0 {
        None
    } else {
        Some(max_length as usize)
    };
    match bert.find_most_similar(query, &candidates, max_length_opt) {
        Ok((idx, score)) => SimilarityResult {
            index: idx as i32,
            score,
        },
        Err(e) => {
            eprintln!("Error finding most similar: {e}");
            SimilarityResult {
                index: -1,
                score: -1.0,
            }
        }
    }
}

// Free a C string allocated by Rust
#[no_mangle]
pub extern "C" fn free_cstring(s: *mut c_char) {
    unsafe {
        if !s.is_null() {
            let _ = CString::from_raw(s);
        }
    }
}

// Free embedding data allocated by Rust
#[no_mangle]
pub extern "C" fn free_embedding(data: *mut f32, length: i32) {
    if !data.is_null() && length > 0 {
        unsafe {
            // Reconstruct the vector so that Rust can properly deallocate it
            let _vec = Vec::from_raw_parts(data, length as usize, length as usize);
            // The vector will be dropped and the memory freed when _vec goes out of scope
        }
    }
}

// Helper function to L2 normalize a tensor
fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    let norm = v.sqr()?.sum_keepdim(1)?.sqrt()?;
    Ok(v.broadcast_div(&norm)?)
}

// New structure to hold classification result
#[repr(C)]
pub struct ClassificationResult {
    pub class: i32,
    pub confidence: f32,
}

// Structure to hold classification result with full probability distribution
#[repr(C)]
pub struct ClassificationResultWithProbs {
    pub class: i32,
    pub confidence: f32,
    pub probabilities: *mut f32,
    pub num_classes: i32,
}

// Initialize the BERT classifier model (called from Go)
#[no_mangle]
pub extern "C" fn init_classifier(
    model_id: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Ensure num_classes is valid
    if num_classes < 2 {
        eprintln!("Number of classes must be at least 2, got {num_classes}");
        return false;
    }

    match BertClassifier::new(model_id, num_classes as usize, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = BERT_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT classifier: {e}");
            false
        }
    }
}

// Initialize the BERT PII classifier model (called from Go)
#[no_mangle]
pub extern "C" fn init_pii_classifier(
    model_id: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Ensure num_classes is valid
    if num_classes < 2 {
        eprintln!("Number of classes must be at least 2, got {num_classes}");
        return false;
    }

    match BertClassifier::new(model_id, num_classes as usize, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = BERT_PII_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT PII classifier: {e}");
            false
        }
    }
}

// Initialize the BERT jailbreak classifier model (called from Go)
#[no_mangle]
pub extern "C" fn init_jailbreak_classifier(
    model_id: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Ensure num_classes is valid
    if num_classes < 2 {
        eprintln!("Number of classes must be at least 2, got {num_classes}");
        return false;
    }

    match BertClassifier::new(model_id, num_classes as usize, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = BERT_JAILBREAK_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT jailbreak classifier: {e}");
            false
        }
    }
}

// Classify text using BERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_text(text: *const c_char) -> ClassificationResult {
    let default_result = ClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = BERT_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying text: {e}");
                default_result
            }
        },
        None => {
            eprintln!("BERT classifier not initialized");
            default_result
        }
    }
}

// Classify text and return full probability distribution (called from Go)
#[no_mangle]
pub extern "C" fn classify_text_with_probabilities(
    text: *const c_char,
) -> ClassificationResultWithProbs {
    let default_result = ClassificationResultWithProbs {
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

    let bert_opt = BERT_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => {
                // For now, we don't have probabilities from the new BERT implementation
                // Return empty probabilities array
                let prob_len = 0;
                let prob_ptr = std::ptr::null_mut();

                ClassificationResultWithProbs {
                    class: class_idx as i32,
                    confidence,
                    probabilities: prob_ptr,
                    num_classes: prob_len as i32,
                }
            }
            Err(e) => {
                eprintln!("Error classifying text with probabilities: {e}");
                default_result
            }
        },
        None => {
            eprintln!("BERT classifier not initialized");
            default_result
        }
    }
}

// Free the probability array allocated by classify_text_with_probabilities
#[no_mangle]
pub extern "C" fn free_probabilities(probabilities: *mut f32, num_classes: i32) {
    if !probabilities.is_null() && num_classes > 0 {
        unsafe {
            let _: Box<[f32]> = Box::from_raw(std::slice::from_raw_parts_mut(
                probabilities,
                num_classes as usize,
            ));
        }
    }
}

// Classify text for PII using BERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_pii_text(text: *const c_char) -> ClassificationResult {
    let default_result = ClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = BERT_PII_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying PII text: {e}");
                default_result
            }
        },
        None => {
            eprintln!("BERT PII classifier not initialized");
            default_result
        }
    }
}

// Classify text for jailbreak detection using BERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_jailbreak_text(text: *const c_char) -> ClassificationResult {
    let default_result = ClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = BERT_JAILBREAK_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying jailbreak text: {e}");
                default_result
            }
        },
        None => {
            eprintln!("BERT jailbreak classifier not initialized");
            default_result
        }
    }
}

// ================================================================================================
// UNIFIED CLASSIFIER C INTERFACE
// ================================================================================================

/// C-compatible structure for unified batch results
#[repr(C)]
pub struct UnifiedBatchResult {
    pub intent_results: *mut CIntentResult,
    pub pii_results: *mut CPIIResult,
    pub security_results: *mut CSecurityResult,
    pub batch_size: i32,
    pub error: bool,
    pub error_message: *mut c_char,
}

/// C-compatible intent result
#[repr(C)]
pub struct CIntentResult {
    pub category: *mut c_char,
    pub confidence: f32,
    pub probabilities: *mut f32,
    pub num_probabilities: i32,
}

/// C-compatible PII result
#[repr(C)]
pub struct CPIIResult {
    pub has_pii: bool,
    pub pii_types: *mut *mut c_char,
    pub num_pii_types: i32,
    pub confidence: f32,
}

/// C-compatible security result
#[repr(C)]
pub struct CSecurityResult {
    pub is_jailbreak: bool,
    pub threat_type: *mut c_char,
    pub confidence: f32,
}

impl UnifiedBatchResult {
    /// Create an error result
    fn error(message: &str) -> Self {
        let error_msg =
            CString::new(message).unwrap_or_else(|_| CString::new("Unknown error").unwrap());
        Self {
            intent_results: std::ptr::null_mut(),
            pii_results: std::ptr::null_mut(),
            security_results: std::ptr::null_mut(),
            batch_size: 0,
            error: true,
            error_message: error_msg.into_raw(),
        }
    }

    /// Convert from Rust BatchClassificationResult to C-compatible structure
    fn from_batch_result(result: BatchClassificationResult) -> Self {
        let batch_size = result.batch_size as i32;

        // Convert intent results
        let intent_results = result
            .intent_results
            .into_iter()
            .map(|r| {
                let probs_len = r.probabilities.len();
                CIntentResult {
                    category: CString::new(r.category).unwrap().into_raw(),
                    confidence: r.confidence,
                    probabilities: {
                        let mut probs = r.probabilities.into_boxed_slice();
                        let ptr = probs.as_mut_ptr();
                        std::mem::forget(probs);
                        ptr
                    },
                    num_probabilities: probs_len as i32,
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let intent_ptr = Box::into_raw(intent_results) as *mut CIntentResult;

        // Convert PII results
        let pii_results = result
            .pii_results
            .into_iter()
            .map(|r| {
                let types_len = r.pii_types.len();
                CPIIResult {
                    has_pii: r.has_pii,
                    pii_types: {
                        let types: Vec<*mut c_char> = r
                            .pii_types
                            .into_iter()
                            .map(|t| CString::new(t).unwrap().into_raw())
                            .collect();
                        let mut types_box = types.into_boxed_slice();
                        let ptr = types_box.as_mut_ptr();
                        std::mem::forget(types_box);
                        ptr
                    },
                    num_pii_types: types_len as i32,
                    confidence: r.confidence,
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let pii_ptr = Box::into_raw(pii_results) as *mut CPIIResult;

        // Convert security results
        let security_results = result
            .security_results
            .into_iter()
            .map(|r| CSecurityResult {
                is_jailbreak: r.is_jailbreak,
                threat_type: CString::new(r.threat_type).unwrap().into_raw(),
                confidence: r.confidence,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let security_ptr = Box::into_raw(security_results) as *mut CSecurityResult;

        Self {
            intent_results: intent_ptr,
            pii_results: pii_ptr,
            security_results: security_ptr,
            batch_size,
            error: false,
            error_message: std::ptr::null_mut(),
        }
    }
}

/// Initialize unified classifier (called from Go)
#[no_mangle]
pub extern "C" fn init_unified_classifier_c(
    modernbert_path: *const c_char,
    intent_head_path: *const c_char,
    pii_head_path: *const c_char,
    security_head_path: *const c_char,
    intent_labels: *const *const c_char,
    intent_labels_count: usize,
    pii_labels: *const *const c_char,
    pii_labels_count: usize,
    security_labels: *const *const c_char,
    security_labels_count: usize,
    use_cpu: bool,
) -> bool {
    let modernbert_path = unsafe {
        match CStr::from_ptr(modernbert_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let intent_head_path = unsafe {
        match CStr::from_ptr(intent_head_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let pii_head_path = unsafe {
        match CStr::from_ptr(pii_head_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let security_head_path = unsafe {
        match CStr::from_ptr(security_head_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Convert C string arrays to Rust Vec<String>
    let intent_labels_vec = unsafe {
        std::slice::from_raw_parts(intent_labels, intent_labels_count)
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr).to_str().unwrap_or("").to_string())
            .collect::<Vec<String>>()
    };

    let pii_labels_vec = unsafe {
        std::slice::from_raw_parts(pii_labels, pii_labels_count)
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr).to_str().unwrap_or("").to_string())
            .collect::<Vec<String>>()
    };

    let security_labels_vec = unsafe {
        std::slice::from_raw_parts(security_labels, security_labels_count)
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr).to_str().unwrap_or("").to_string())
            .collect::<Vec<String>>()
    };

    match UnifiedClassifier::new(
        modernbert_path,
        intent_head_path,
        pii_head_path,
        security_head_path,
        intent_labels_vec,
        pii_labels_vec,
        security_labels_vec,
        use_cpu,
    ) {
        Ok(classifier) => {
            let mut global_classifier = UNIFIED_CLASSIFIER.lock().unwrap();
            *global_classifier = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize unified classifier: {e}");
            false
        }
    }
}

/// Classify batch of texts using unified classifier (called from Go)
#[no_mangle]
pub extern "C" fn classify_unified_batch(
    texts_ptr: *const *const c_char,
    num_texts: i32,
) -> UnifiedBatchResult {
    if texts_ptr.is_null() || num_texts <= 0 {
        return UnifiedBatchResult::error("Invalid input parameters");
    }

    // Convert C strings to Rust strings
    let texts = unsafe {
        std::slice::from_raw_parts(texts_ptr, num_texts as usize)
            .iter()
            .map(|&ptr| {
                if ptr.is_null() {
                    Err("Null text pointer")
                } else {
                    CStr::from_ptr(ptr).to_str().map_err(|_| "Invalid UTF-8")
                }
            })
            .collect::<Result<Vec<_>, _>>()
    };

    let texts = match texts {
        Ok(t) => t,
        Err(e) => return UnifiedBatchResult::error(e),
    };

    // Get unified classifier and perform batch classification
    match get_unified_classifier() {
        Ok(classifier_guard) => match classifier_guard.as_ref() {
            Some(classifier) => match classifier.classify_batch(&texts) {
                Ok(result) => UnifiedBatchResult::from_batch_result(result),
                Err(e) => UnifiedBatchResult::error(&format!("Classification failed: {}", e)),
            },
            None => UnifiedBatchResult::error("Unified classifier not initialized"),
        },
        Err(e) => UnifiedBatchResult::error(&format!("Failed to get classifier: {}", e)),
    }
}

/// Free unified batch result memory (called from Go)
#[no_mangle]
pub extern "C" fn free_unified_batch_result(result: UnifiedBatchResult) {
    if result.error {
        if !result.error_message.is_null() {
            unsafe {
                let _ = CString::from_raw(result.error_message);
            }
        }
        return;
    }

    let batch_size = result.batch_size as usize;

    // Free intent results
    if !result.intent_results.is_null() {
        unsafe {
            let intent_slice = std::slice::from_raw_parts_mut(result.intent_results, batch_size);
            for intent in intent_slice {
                if !intent.category.is_null() {
                    let _ = CString::from_raw(intent.category);
                }
                if !intent.probabilities.is_null() {
                    let _ = Vec::from_raw_parts(
                        intent.probabilities,
                        intent.num_probabilities as usize,
                        intent.num_probabilities as usize,
                    );
                }
            }
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(
                result.intent_results,
                batch_size,
            ));
        }
    }

    // Free PII results
    if !result.pii_results.is_null() {
        unsafe {
            let pii_slice = std::slice::from_raw_parts_mut(result.pii_results, batch_size);
            for pii in pii_slice {
                if !pii.pii_types.is_null() {
                    let types_slice =
                        std::slice::from_raw_parts_mut(pii.pii_types, pii.num_pii_types as usize);
                    for &mut type_ptr in types_slice {
                        if !type_ptr.is_null() {
                            let _ = CString::from_raw(type_ptr);
                        }
                    }
                    let _ = Vec::from_raw_parts(
                        pii.pii_types,
                        pii.num_pii_types as usize,
                        pii.num_pii_types as usize,
                    );
                }
            }
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(
                result.pii_results,
                batch_size,
            ));
        }
    }

    // Free security results
    if !result.security_results.is_null() {
        unsafe {
            let security_slice =
                std::slice::from_raw_parts_mut(result.security_results, batch_size);
            for security in security_slice {
                if !security.threat_type.is_null() {
                    let _ = CString::from_raw(security.threat_type);
                }
            }
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(
                result.security_results,
                batch_size,
            ));
        }
    }
}

// ================================================================================================
// BERT TOKEN CLASSIFICATION C INTERFACE
// ================================================================================================

// Global variable to hold BERT token classifier
lazy_static::lazy_static! {
    static ref BERT_TOKEN_CLASSIFIER: Arc<Mutex<Option<UniversalBertClassifier>>> = Arc::new(Mutex::new(None));

    // New official Candle BERT classifiers
    static ref CANDLE_BERT_CLASSIFIER: Arc<Mutex<Option<CandleBertClassifier>>> = Arc::new(Mutex::new(None));
    static ref CANDLE_BERT_TOKEN_CLASSIFIER: Arc<Mutex<Option<CandleBertTokenClassifier>>> = Arc::new(Mutex::new(None));
}

/// Initialize BERT token classifier (called from Go)
#[no_mangle]
pub extern "C" fn init_bert_token_classifier(
    model_path: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error converting model path: {e}");
                return false;
            }
        }
    };

    println!("Initializing BERT token classifier from: {model_path}");

    match UniversalBertClassifier::new_token_classification(
        model_path,
        num_classes as usize,
        use_cpu,
    ) {
        Ok(classifier) => {
            let mut bert_opt = BERT_TOKEN_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            println!("BERT token classifier initialized successfully");
            true
        }
        Err(e) => {
            eprintln!("Error initializing BERT token classifier: {e}");
            false
        }
    }
}

/// Classify tokens for PII detection using BERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_bert_pii_tokens(
    text: *const c_char,
    id2label_json: *const c_char,
) -> BertTokenClassificationResult {
    let default_result = BertTokenClassificationResult {
        entities: std::ptr::null_mut(),
        num_entities: 0,
    };

    // Parse input text
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    // Parse id2label mapping
    let id2label_str = unsafe {
        match CStr::from_ptr(id2label_json).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let id2label: HashMap<usize, String> = match serde_json::from_str(id2label_str) {
        Ok(mapping) => mapping,
        Err(e) => {
            eprintln!("Error parsing id2label mapping: {e}");
            return default_result;
        }
    };

    // Get classifier and classify tokens
    let bert_opt = BERT_TOKEN_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_tokens(text, &id2label) {
            Ok(entities) => {
                // Convert Rust entities to C-compatible format
                let num_entities = entities.len() as i32;
                if num_entities == 0 {
                    return default_result;
                }

                // Allocate memory for C entities
                let c_entities = entities
                    .into_iter()
                    .map(|entity| {
                        let entity_type = CString::new(entity.entity_type)
                            .unwrap_or_else(|_| CString::new("UNKNOWN").unwrap())
                            .into_raw();
                        let text = CString::new(entity.text)
                            .unwrap_or_else(|_| CString::new("").unwrap())
                            .into_raw();

                        BertTokenEntity {
                            entity_type,
                            start: entity.start,
                            end: entity.end,
                            text,
                            confidence: entity.confidence,
                        }
                    })
                    .collect::<Vec<_>>();

                let entities_ptr =
                    Box::into_raw(c_entities.into_boxed_slice()) as *mut BertTokenEntity;

                BertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities,
                }
            }
            Err(e) => {
                eprintln!("Error classifying tokens: {e}");
                default_result
            }
        },
        None => {
            eprintln!("BERT token classifier not initialized");
            default_result
        }
    }
}

/// Free memory allocated for BERT token classification result (called from Go)
#[no_mangle]
pub extern "C" fn free_bert_token_classification_result(result: BertTokenClassificationResult) {
    if !result.entities.is_null() && result.num_entities > 0 {
        unsafe {
            let entities_slice =
                std::slice::from_raw_parts_mut(result.entities, result.num_entities as usize);

            // Free individual entity strings
            for entity in entities_slice {
                if !entity.entity_type.is_null() {
                    let _ = CString::from_raw(entity.entity_type);
                }
                if !entity.text.is_null() {
                    let _ = CString::from_raw(entity.text);
                }
            }

            // Free the entities array
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(
                result.entities,
                result.num_entities as usize,
            ));
        }
    }
}

/// Initialize BERT sequence classifier using official Candle implementation (called from Go)
#[no_mangle]
pub extern "C" fn init_candle_bert_classifier(
    model_path: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match CandleBertClassifier::new(model_path, num_classes as usize, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = CANDLE_BERT_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(_e) => false,
    }
}

/// Initialize BERT token classifier using official Candle implementation (called from Go)
#[no_mangle]
pub extern "C" fn init_candle_bert_token_classifier(
    model_path: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match CandleBertTokenClassifier::new(model_path, num_classes as usize, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = CANDLE_BERT_TOKEN_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(_e) => false,
    }
}

/// Classify tokens using official Candle BERT token classifier with id2label mapping (called from Go)
#[no_mangle]
pub extern "C" fn classify_candle_bert_tokens_with_labels(
    text: *const c_char,
    id2label_json: *const c_char,
) -> BertTokenClassificationResult {
    let default_result = BertTokenClassificationResult {
        entities: std::ptr::null_mut(),
        num_entities: 0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let id2label_str = unsafe {
        match CStr::from_ptr(id2label_json).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    // Parse id2label mapping
    let id2label: std::collections::HashMap<String, String> =
        match serde_json::from_str(id2label_str) {
            Ok(mapping) => mapping,
            Err(e) => {
                eprintln!("Failed to parse id2label mapping: {}", e);
                return default_result;
            }
        };

    let bert_opt = CANDLE_BERT_TOKEN_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_tokens_with_spans(text) {
            Ok(results) => {
                // Convert results to C-compatible format with proper labels and spans
                let mut entities = Vec::new();

                for (token, class_idx, confidence, start_char, end_char) in results {
                    // Skip special tokens and O labels
                    if class_idx == 0
                        || token.starts_with("##")
                        || token == "[CLS]"
                        || token == "[SEP]"
                    {
                        continue;
                    }

                    // Get actual label name from mapping
                    let label_name = id2label
                        .get(&class_idx.to_string())
                        .unwrap_or(&format!("CLASS_{}", class_idx))
                        .clone();

                    // Extract actual text from original text using character spans
                    let actual_text = if start_char < end_char && end_char <= text.len() {
                        text[start_char..end_char].to_string()
                    } else {
                        token.clone()
                    };

                    let entity = BertTokenEntity {
                        entity_type: CString::new(label_name).unwrap().into_raw(),
                        start: start_char as i32,
                        end: end_char as i32,
                        text: CString::new(actual_text).unwrap().into_raw(),
                        confidence,
                    };
                    entities.push(entity);
                }

                if entities.is_empty() {
                    return default_result;
                }

                let entities_ptr = entities.as_mut_ptr();
                let num_entities = entities.len() as i32;
                std::mem::forget(entities); // Prevent deallocation

                BertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities,
                }
            }
            Err(e) => {
                eprintln!("Error classifying tokens with Candle BERT: {e}");
                default_result
            }
        },
        None => {
            eprintln!("Candle BERT token classifier not initialized");
            default_result
        }
    }
}

/// Classify tokens using official Candle BERT token classifier (called from Go)
#[no_mangle]
pub extern "C" fn classify_candle_bert_tokens(
    text: *const c_char,
) -> BertTokenClassificationResult {
    let default_result = BertTokenClassificationResult {
        entities: std::ptr::null_mut(),
        num_entities: 0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = CANDLE_BERT_TOKEN_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_tokens_with_spans(text) {
            Ok(results) => {
                // Convert results to C-compatible format with proper spans
                let mut entities = Vec::new();

                for (token, class_idx, confidence, start_char, end_char) in results {
                    // Skip special tokens and O labels
                    if class_idx == 0
                        || token.starts_with("##")
                        || token == "[CLS]"
                        || token == "[SEP]"
                    {
                        continue;
                    }

                    // Extract actual text from original text using character spans
                    let actual_text = if start_char < end_char && end_char <= text.len() {
                        text[start_char..end_char].to_string()
                    } else {
                        token.clone()
                    };

                    let entity = BertTokenEntity {
                        entity_type: CString::new(format!("CLASS_{}", class_idx))
                            .unwrap()
                            .into_raw(),
                        start: start_char as i32,
                        end: end_char as i32,
                        text: CString::new(actual_text).unwrap().into_raw(),
                        confidence,
                    };
                    entities.push(entity);
                }

                if entities.is_empty() {
                    return default_result;
                }

                let entities_ptr = entities.as_mut_ptr();
                let num_entities = entities.len() as i32;
                std::mem::forget(entities); // Prevent deallocation

                BertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities,
                }
            }
            Err(e) => {
                eprintln!("Error classifying tokens with Candle BERT: {e}");
                default_result
            }
        },
        None => {
            eprintln!("Candle BERT token classifier not initialized");
            default_result
        }
    }
}

/// Classify text for sequence classification using official Candle BERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_candle_bert_text(text: *const c_char) -> ClassificationResult {
    let default_result = ClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = CANDLE_BERT_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying text with Candle BERT: {e}");
                default_result
            }
        },
        None => {
            eprintln!("Candle BERT classifier not initialized");
            default_result
        }
    }
}

/// Classify text for sequence classification using BERT (called from Go)
#[no_mangle]
pub extern "C" fn classify_bert_text(text: *const c_char) -> ClassificationResult {
    let default_result = ClassificationResult {
        class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = BERT_TOKEN_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ClassificationResult {
                class: class_idx as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Error classifying text: {e}");
                default_result
            }
        },
        None => {
            eprintln!("BERT classifier not initialized");
            default_result
        }
    }
}

// ================================================================================================
// END OF BERT TOKEN CLASSIFICATION C INTERFACE
// ================================================================================================

// ================================================================================================
// LORA UNIFIED CLASSIFIER C INTERFACE
// ================================================================================================

// UnifiedClassifier and BatchClassificationResult already imported above

// Global LoRA Unified Classifier instance
static LORA_UNIFIED_CLASSIFIER: Mutex<Option<UnifiedClassifier>> = Mutex::new(None);

/// Initialize LoRA Unified Classifier with high-confidence models
#[no_mangle]
pub extern "C" fn init_lora_unified_classifier(
    intent_model_path: *const c_char,
    pii_model_path: *const c_char,
    security_model_path: *const c_char,
    architecture: *const c_char, // "bert", "roberta", or "modernbert"
    use_cpu: bool,
) -> bool {
    let intent_path = unsafe {
        match CStr::from_ptr(intent_model_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let pii_path = unsafe {
        match CStr::from_ptr(pii_model_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let security_path = unsafe {
        match CStr::from_ptr(security_model_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let arch = unsafe {
        match CStr::from_ptr(architecture).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match UnifiedClassifier::new_with_lora_models(
        intent_path,
        pii_path,
        security_path,
        arch,
        use_cpu,
    ) {
        Ok(classifier) => {
            let mut classifier_opt = LORA_UNIFIED_CLASSIFIER.lock().unwrap();
            *classifier_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize unified classifier: {}", e);
            false
        }
    }
}

/// High-confidence batch classification result for C interface
#[repr(C)]
pub struct LoRABatchResult {
    pub intent_results: *mut LoRAIntentResult,
    pub pii_results: *mut LoRAPIIResult,
    pub security_results: *mut LoRASecurityResult,
    pub batch_size: i32,
    pub avg_confidence: f32, // Expected: 0.99+
}

/// High-confidence intent result for C interface
#[repr(C)]
pub struct LoRAIntentResult {
    pub category: *mut c_char,
    pub confidence: f32, // Expected: 0.99+
}

/// High-confidence PII result for C interface
#[repr(C)]
pub struct LoRAPIIResult {
    pub has_pii: bool,
    pub pii_types: *mut *mut c_char,
    pub num_pii_types: i32,
    pub confidence: f32, // Expected: 0.99+
}

/// High-confidence security result for C interface
#[repr(C)]
pub struct LoRASecurityResult {
    pub is_jailbreak: bool,
    pub threat_type: *mut c_char,
    pub confidence: f32, // Expected: 0.99+
}

/// High-confidence batch classification using LoRA models
#[no_mangle]
pub extern "C" fn classify_batch_with_lora(
    texts: *const *const c_char,
    num_texts: i32,
) -> LoRABatchResult {
    let default_result = LoRABatchResult {
        intent_results: std::ptr::null_mut(),
        pii_results: std::ptr::null_mut(),
        security_results: std::ptr::null_mut(),
        batch_size: 0,
        avg_confidence: 0.0,
    };

    if num_texts <= 0 {
        return default_result;
    }

    // Convert C strings to Rust strings
    let mut text_vec = Vec::new();
    for i in 0..num_texts {
        let text_ptr = unsafe { *texts.offset(i as isize) };
        let text = unsafe {
            match CStr::from_ptr(text_ptr).to_str() {
                Ok(s) => s,
                Err(_) => return default_result,
            }
        };
        text_vec.push(text);
    }

    let classifier_opt = LORA_UNIFIED_CLASSIFIER.lock().unwrap();
    match &*classifier_opt {
        Some(classifier) => {
            match classifier.classify_batch(&text_vec) {
                Ok(batch_result) => {
                    // Convert Rust results to C-compatible format
                    let mut intent_results = Vec::new();
                    let mut pii_results = Vec::new();
                    let mut security_results = Vec::new();
                    let mut total_confidence = 0.0f32;

                    for (_i, (intent, pii, security)) in batch_result
                        .intent_results
                        .iter()
                        .zip(batch_result.pii_results.iter())
                        .zip(batch_result.security_results.iter())
                        .map(|((a, b), c)| (a, b, c))
                        .enumerate()
                    {
                        // Intent result
                        let intent_c = LoRAIntentResult {
                            category: CString::new(intent.category.clone()).unwrap().into_raw(),
                            confidence: intent.confidence,
                        };
                        intent_results.push(intent_c);

                        // PII result
                        let pii_types_c: Vec<*mut c_char> = pii
                            .pii_types
                            .iter()
                            .map(|s| CString::new(s.clone()).unwrap().into_raw())
                            .collect();
                        let pii_types_ptr = if pii_types_c.is_empty() {
                            std::ptr::null_mut()
                        } else {
                            let ptr = pii_types_c.as_ptr() as *mut *mut c_char;
                            std::mem::forget(pii_types_c);
                            ptr
                        };

                        let pii_c = LoRAPIIResult {
                            has_pii: pii.has_pii,
                            pii_types: pii_types_ptr,
                            num_pii_types: pii.pii_types.len() as i32,
                            confidence: pii.confidence,
                        };
                        pii_results.push(pii_c);

                        // Security result
                        let security_c = LoRASecurityResult {
                            is_jailbreak: security.is_jailbreak,
                            threat_type: CString::new(security.threat_type.clone())
                                .unwrap()
                                .into_raw(),
                            confidence: security.confidence,
                        };
                        security_results.push(security_c);

                        // Calculate average confidence
                        total_confidence +=
                            (intent.confidence + pii.confidence + security.confidence) / 3.0;
                    }

                    let avg_confidence = total_confidence / num_texts as f32;

                    // Prepare final result
                    let intent_ptr = intent_results.as_mut_ptr();
                    let pii_ptr = pii_results.as_mut_ptr();
                    let security_ptr = security_results.as_mut_ptr();

                    std::mem::forget(intent_results);
                    std::mem::forget(pii_results);
                    std::mem::forget(security_results);

                    LoRABatchResult {
                        intent_results: intent_ptr,
                        pii_results: pii_ptr,
                        security_results: security_ptr,
                        batch_size: num_texts,
                        avg_confidence,
                    }
                }
                Err(_e) => default_result,
            }
        }
        None => default_result,
    }
}

/// Free LoRA batch classification result
#[no_mangle]
pub extern "C" fn free_lora_batch_result(result: LoRABatchResult) {
    if result.batch_size <= 0 {
        return;
    }

    // Free intent results
    if !result.intent_results.is_null() {
        let intent_slice = unsafe {
            std::slice::from_raw_parts_mut(result.intent_results, result.batch_size as usize)
        };
        for intent in intent_slice {
            if !intent.category.is_null() {
                unsafe {
                    let _ = CString::from_raw(intent.category);
                }
            }
        }
        unsafe {
            let _ = Vec::from_raw_parts(
                result.intent_results,
                result.batch_size as usize,
                result.batch_size as usize,
            );
        }
    }

    // Free PII results
    if !result.pii_results.is_null() {
        let pii_slice = unsafe {
            std::slice::from_raw_parts_mut(result.pii_results, result.batch_size as usize)
        };
        for pii in pii_slice {
            if !pii.pii_types.is_null() && pii.num_pii_types > 0 {
                let pii_types_slice = unsafe {
                    std::slice::from_raw_parts_mut(pii.pii_types, pii.num_pii_types as usize)
                };
                for pii_type in pii_types_slice {
                    if !pii_type.is_null() {
                        unsafe {
                            let _ = CString::from_raw(*pii_type);
                        }
                    }
                }
                unsafe {
                    let _ = Vec::from_raw_parts(
                        pii.pii_types,
                        pii.num_pii_types as usize,
                        pii.num_pii_types as usize,
                    );
                }
            }
        }
        unsafe {
            let _ = Vec::from_raw_parts(
                result.pii_results,
                result.batch_size as usize,
                result.batch_size as usize,
            );
        }
    }

    // Free security results
    if !result.security_results.is_null() {
        let security_slice = unsafe {
            std::slice::from_raw_parts_mut(result.security_results, result.batch_size as usize)
        };
        for security in security_slice {
            if !security.threat_type.is_null() {
                unsafe {
                    let _ = CString::from_raw(security.threat_type);
                }
            }
        }
        unsafe {
            let _ = Vec::from_raw_parts(
                result.security_results,
                result.batch_size as usize,
                result.batch_size as usize,
            );
        }
    }
}

// ================================================================================================
// END OF LORA UNIFIED CLASSIFIER C INTERFACE
// ================================================================================================
