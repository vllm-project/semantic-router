//! Semantic Similarity Core Module

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::Path;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// Structure to hold BERT model and tokenizer for semantic similarity
///
/// This is the core similarity computation engine that provides embedding
/// generation and similarity calculation capabilities for both traditional
/// and LoRA model paths.
pub struct BertSimilarity {
    /// The BERT model for generating embeddings
    model: BertModel,
    /// Tokenizer for text preprocessing
    tokenizer: Tokenizer,
    /// Computing device (CPU or CUDA)
    device: Device,
}

impl BertSimilarity {
    /// Create a new BertSimilarity instance
    ///
    /// ## Arguments
    /// * `model_id` - Model identifier (HuggingFace Hub ID or local path)
    /// * `use_cpu` - Whether to force CPU usage (false for GPU when available)
    ///
    /// ## Returns
    /// * `Result<Self>` - Initialized BertSimilarity instance
    ///
    /// ## Examples
    /// ```rust
    /// let similarity = BertSimilarity::new("sentence-transformers/all-MiniLM-L6-v2", false)?;
    /// ```
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
                        Err(_) => (api.get("pytorch_model.bin")?, true),
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

    /// Tokenize a text string into token IDs and token strings
    ///
    /// ## Arguments
    /// * `text` - Input text to tokenize
    /// * `max_length` - Maximum sequence length (default: 512)
    ///
    /// ## Returns
    /// * `Result<(Vec<i32>, Vec<String>)>` - Tuple of (token_ids, tokens)
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

    /// Get embedding for a text
    ///
    /// ## Arguments
    /// * `text` - Input text to embed
    /// * `max_length` - Maximum sequence length (default: 512)
    ///
    /// ## Returns
    /// * `Result<Tensor>` - Normalized embedding tensor
    ///
    /// ## Notes
    /// Uses mean pooling over token embeddings with attention mask weighting,
    /// followed by L2 normalization for cosine similarity compatibility.
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

    /// Calculate cosine similarity between two texts
    ///
    /// ## Arguments
    /// * `text1` - First text for comparison
    /// * `text2` - Second text for comparison
    /// * `max_length` - Maximum sequence length (default: 512)
    ///
    /// ## Returns
    /// * `Result<f32>` - Cosine similarity score between -1.0 and 1.0
    ///
    /// ## Notes
    /// For normalized embeddings, dot product equals cosine similarity.
    /// Higher values indicate greater similarity.
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

    /// Find most similar text from a list of candidates
    ///
    /// ## Arguments
    /// * `query_text` - Query text to find matches for
    /// * `candidates` - List of candidate texts to compare against
    /// * `max_length` - Maximum sequence length (default: 512)
    ///
    /// ## Returns
    /// * `Result<(usize, f32)>` - Tuple of (best_index, similarity_score)
    ///
    /// ## Errors
    /// * Returns error if candidates list is empty
    ///
    /// ## Performance
    /// This method computes embeddings for each candidate individually,
    /// which is suitable for small candidate lists. For large lists,
    /// consider batch processing.
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

    /// Get the device this model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get a reference to the tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Check if the model is running on GPU
    pub fn is_gpu(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }
}

/// Normalize a tensor using L2 normalization
///
/// ## Arguments
/// * `v` - Input tensor to normalize
///
/// ## Returns
/// * `Result<Tensor>` - L2 normalized tensor
///
/// ## Notes
/// This function computes L2 norm along the last dimension and normalizes
/// the input tensor by dividing by the norm. This ensures unit vectors
/// suitable for cosine similarity calculations.
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    let norm = v.sqr()?.sum_keepdim(1)?.sqrt()?;
    Ok(v.broadcast_div(&norm)?)
}
