// This file is a binding for the candle-core and candle-transformers libraries.
// It is based on https://github.com/huggingface/candle/tree/main/candle-examples/examples/bert
use std::ffi::{c_char, CStr, CString};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

pub mod modernbert;

// Re-export ModernBERT functions and structures
pub use modernbert::{
    classify_modernbert_jailbreak_text, classify_modernbert_pii_text, classify_modernbert_text,
    init_modernbert_classifier, init_modernbert_jailbreak_classifier,
    init_modernbert_pii_classifier, ModernBertClassificationResult,
};

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
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
    model: BertModel,
    tokenizer: Tokenizer,
    classification_head: Linear,
    num_classes: usize,
    device: Device,
}

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
                println!("Loading model from local directory: {model_id}");
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
                println!("Loading model from HuggingFace Hub: {model_id}");
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
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Use the approximate GELU for better performance
        config.hidden_act = HiddenAct::GeluApproximate;

        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
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

        if is_sentence_transformer {
            println!("Detected SentenceTransformer model with linear classifier head");
        }

        let (config_filename, tokenizer_filename, weights_filename, use_pth) =
            if Path::new(model_id).exists() {
                // Local model path
                println!("Loading model from local directory: {model_id}");
                let config_path = Path::new(model_id).join("config.json");
                let tokenizer_path = Path::new(model_id).join("tokenizer.json");

                // For SentenceTransformer models, check both the root and 0_Transformer
                let weights_path = if is_sentence_transformer {
                    // First check if model weights are at the root level (most common for sentence-transformers)
                    if Path::new(model_id).join("model.safetensors").exists() {
                        println!("Found model weights at root level");
                        (
                            Path::new(model_id)
                                .join("model.safetensors")
                                .to_string_lossy()
                                .to_string(),
                            false,
                        )
                    } else if Path::new(model_id).join("pytorch_model.bin").exists() {
                        println!("Found PyTorch model at root level");
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
                println!("Loading model from HuggingFace Hub: {model_id}");
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
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Use approximate GELU for better performance
        config.hidden_act = HiddenAct::GeluApproximate;

        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };

        println!("Successfully loaded transformer model");
        let model = BertModel::load(vb.clone(), &config)?;
        println!("Successfully initialized BERT model instance");

        // Create a classification head
        // For SentenceTransformer models, we need to load the Dense layer weights from 2_Dense
        let (w, b) = if is_sentence_transformer {
            // Load the dense layer weights from 2_Dense
            let dense_dir = Path::new(model_id).join("2_Dense");
            println!("Looking for dense weights in {}", dense_dir.display());

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
                    println!("Found dense safetensors weights");
                    (
                        dense_dir
                            .join("model.safetensors")
                            .to_string_lossy()
                            .to_string(),
                        false,
                    )
                } else if dense_dir.join("pytorch_model.bin").exists() {
                    println!("Found dense PyTorch weights");
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
                println!("Successfully loaded dense layer weights");

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
            // Regular BERT model: create random weights
            let hidden_size = config.hidden_size;
            let w = Tensor::randn(0.0, 0.02, (hidden_size, num_classes), &device)?;
            let b = Tensor::zeros((num_classes,), DType::F32, &device)?;
            (w, b)
        };

        let classification_head = Linear::new(w, Some(b));
        println!("Linear classification head created");

        Ok(Self {
            model,
            tokenizer,
            classification_head,
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

        // Implement proper mean pooling for SentenceTransformer
        // Sum over token dimension (dim=1) and divide by attention mask sum to get mean
        let embedding_sum = embeddings.sum(1)?;
        let attention_mask_sum = attention_mask_tensor.to_dtype(embeddings.dtype())?.sum(1)?;
        let pooled_embedding = embedding_sum.broadcast_div(&attention_mask_sum)?;

        // Get the dimensions and convert to the right type
        let pooled_embedding = pooled_embedding.to_dtype(DType::F32)?;

        // Apply the linear layer (classification head) manually
        let weights = self.classification_head.weight().to_dtype(DType::F32)?;
        let bias = self
            .classification_head
            .bias()
            .unwrap()
            .to_dtype(DType::F32)?;

        // Use matmul with the weights matrix
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
        Some(classifier) => match classifier.classify_text_with_probs(text) {
            Ok((class_idx, confidence, probabilities)) => {
                // Allocate memory for probabilities array
                let prob_len = probabilities.len();
                let prob_ptr = Box::into_raw(probabilities.into_boxed_slice()) as *mut f32;

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
