//! Qwen3Guard Model for Safety Classification
//!
//! This module implements the Qwen3Guard-Gen model for detecting unsafe content,
//! jailbreak attempts, and other safety concerns using generative inference.
//!
//! Based on: https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B
//!
//! Key features:
//! - Text generation with safety classification
//! - Regex parsing of structured output
//! - Support for multiple severity levels (Safe, Controversial, Unsafe)
//! - Category-specific detection (Jailbreak, Violence, Hate Speech, etc.)
//!
//! Example output format:
//! ```text
//! Reasoning: The user is attempting to bypass safety guidelines...
//! Category: Jailbreak
//! Severity level: Unsafe
//! ```

use crate::core::{ConfigErrorType, UnifiedError, UnifiedResult};
use crate::model_architectures::generative::qwen3_with_lora::{
    Config as Qwen3Config, ModelForCausalLM as Qwen3Model,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Guard generation result (just raw text, no parsing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardGenerationResult {
    /// Raw generated text from the model
    pub raw_output: String,
}

/// Qwen3Guard model configuration
#[derive(Debug, Clone)]
pub struct Qwen3GuardConfig {
    /// Temperature for generation (0.0 = deterministic)
    pub temperature: f64,

    /// Top-p sampling threshold
    pub top_p: f64,

    /// Maximum tokens to generate
    pub max_tokens: usize,

    /// Repeat penalty
    pub repeat_penalty: f32,

    /// Context size for repeat penalty
    pub repeat_last_n: usize,
}

impl Default for Qwen3GuardConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0, // Deterministic by default
            top_p: 0.95,
            max_tokens: 512,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

/// Qwen3Guard model for safety classification
pub struct Qwen3GuardModel {
    /// Base Qwen3 model
    model: Qwen3Model,

    /// Tokenizer
    tokenizer: Arc<Tokenizer>,

    /// Device
    device: Device,

    /// Data type
    dtype: DType,

    /// Generation config
    config: Qwen3GuardConfig,

    /// EOS token ID
    eos_token_id: u32,

    /// IM_END token ID (for chat models)
    im_end_token_id: u32,
}

impl Qwen3GuardModel {
    /// Create a new Qwen3Guard model
    ///
    /// # Arguments
    /// - `model_path`: Path to Qwen3Guard model directory
    /// - `device`: Device to run on
    /// - `config`: Optional generation configuration
    pub fn new(
        model_path: &str,
        device: &Device,
        config: Option<Qwen3GuardConfig>,
    ) -> UnifiedResult<Self> {
        println!("🛡️  Initializing Qwen3Guard Model");
        println!("  Model path: {}", model_path);

        let base_dir = Path::new(model_path);

        // Load config
        let config_path = base_dir.join("config.json");
        let model_config: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)
            .map_err(|e| UnifiedError::Configuration {
                operation: "parse config".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            })?;

        println!(
            "  Config: hidden_size={}, layers={}, vocab={}",
            model_config.hidden_size, model_config.num_hidden_layers, model_config.vocab_size
        );

        // Load tokenizer
        let tokenizer_path = base_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| UnifiedError::Configuration {
                operation: "load tokenizer".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            })?;

        // Get special token IDs
        let eos_token_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(151643);
        let im_end_token_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);

        // Determine dtype
        let dtype = if device.is_cuda() || device.is_metal() {
            DType::BF16
        } else {
            DType::F32
        };
        println!("  Using dtype: {:?}", dtype);

        // Load model weights
        println!("  Loading model weights...");
        let vb = {
            let single_weights_path = base_dir.join("model.safetensors");
            let index_path = base_dir.join("model.safetensors.index.json");

            if single_weights_path.exists() {
                // Single file
                unsafe {
                    VarBuilder::from_mmaped_safetensors(&[single_weights_path], dtype, device)
                        .map_err(|e| UnifiedError::Processing {
                            operation: "load weights".to_string(),
                            source: e.to_string(),
                            input_context: None,
                        })?
                }
            } else if index_path.exists() {
                // Sharded model
                let index_data = std::fs::read(&index_path)?;
                let index: serde_json::Value =
                    serde_json::from_slice(&index_data).map_err(|e| {
                        UnifiedError::Configuration {
                            operation: "parse index".to_string(),
                            source: ConfigErrorType::ParseError(e.to_string()),
                            context: None,
                        }
                    })?;

                let weight_map =
                    index["weight_map"]
                        .as_object()
                        .ok_or_else(|| UnifiedError::Configuration {
                            operation: "parse weight_map".to_string(),
                            source: ConfigErrorType::ParseError("Missing weight_map".to_string()),
                            context: None,
                        })?;

                let mut weight_files = std::collections::HashSet::new();
                for file in weight_map.values() {
                    if let Some(f) = file.as_str() {
                        weight_files.insert(base_dir.join(f));
                    }
                }

                let weight_files: Vec<_> = weight_files.into_iter().collect();
                unsafe {
                    VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device).map_err(
                        |e| UnifiedError::Processing {
                            operation: "load weights".to_string(),
                            source: e.to_string(),
                            input_context: None,
                        },
                    )?
                }
            } else {
                return Err(UnifiedError::Configuration {
                    operation: "find weights".to_string(),
                    source: ConfigErrorType::ParseError(
                        "No model.safetensors or model.safetensors.index.json found".to_string(),
                    ),
                    context: None,
                });
            }
        };

        // Load model
        let model = Qwen3Model::new(&model_config, vb).map_err(|e| UnifiedError::Processing {
            operation: "load model".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        println!("✅ Qwen3Guard model loaded successfully");

        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            device: device.clone(),
            dtype,
            config: config.unwrap_or_default(),
            eos_token_id,
            im_end_token_id,
        })
    }

    /// Format input prompt for guard classification
    /// This uses the exact chat template from Qwen3Guard's tokenizer
    fn format_prompt(&self, text: &str, mode: &str) -> String {
        // Use the official Qwen3Guard chat template
        // This template is from tokenizer.apply_chat_template in the HuggingFace model
        let user_label = if mode == "output" {
            "ASSISTANT"
        } else {
            "USER"
        };

        format!(
            "<|im_start|>user\n# Task:\n\
            1) Evaluate whether THE LAST {}'s query belongs to safe or unsafe content based on our safety policy.\n\
            2) Identify the category of unsafe content found.\n\n\
            <BEGIN SAFETY POLICY>\n\
            **Safety: Safe**: The {}'s query does not contain any prohibited content or content that could lead to prohibited content.\n\
            **Safety: Unsafe**: The {}'s query contains prohibited content.\n\
            **Safety: Controversial**: The {}'s query does not explicitly contain prohibited content, but its intent, context, or potential responses could be misinterpreted, misused, or exploited to generate unsafe content under certain conditions.\n\
            <END SAFETY POLICY>\n\n\
            <BEGIN UNSAFE CONTENT CATEGORIES>\n\
            Violent.\n\
            Non-violent Illegal Acts.\n\
            Sexual Content or Sexual Acts.\n\
            PII.\n\
            Suicide & Self-Harm.\n\
            Unethical Acts.\n\
            Politically Sensitive Topics.\n\
            Copyright Violation.\n\
            Jailbreak.\n\
            <END UNSAFE CONTENT CATEGORIES>\n\n\
            <BEGIN CONVERSATION>\n\n\
            {}: {}\n\n\
            <END CONVERSATION>\n\n\
            Provide your safety assessment for ONLY THE LAST **{}'s query** in the above conversation:\n \
            - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n \
            - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>\n\
            <|im_start|>assistant\n\
            <think>\n\n\
            </think>\n\n",
            user_label, user_label, user_label, user_label, user_label, text, user_label
        )
    }

    /// Generate text using the model
    fn generate(&mut self, prompt: &str) -> UnifiedResult<String> {
        // Clear KV cache
        self.model.clear_kv_cache();

        // Tokenize
        let encoding =
            self.tokenizer
                .encode(prompt, true)
                .map_err(|e| UnifiedError::Configuration {
                    operation: "tokenize".to_string(),
                    source: ConfigErrorType::ParseError(e.to_string()),
                    context: None,
                })?;

        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let mut generated_text = String::new();

        // Generation loop
        for step in 0..self.config.max_tokens {
            let context_size = if step > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];

            // Forward pass
            let input = Tensor::new(ctxt, &self.device)
                .map_err(|e| UnifiedError::Processing {
                    operation: "create tensor".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?
                .unsqueeze(0)
                .map_err(|e| UnifiedError::Processing {
                    operation: "unsqueeze".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;

            let logits =
                self.model
                    .forward(&input, start_pos)
                    .map_err(|e| UnifiedError::Processing {
                        operation: "forward pass".to_string(),
                        source: e.to_string(),
                        input_context: None,
                    })?;

            // Extract last token logits
            let logits = logits
                .squeeze(0)
                .map_err(|e| UnifiedError::Processing {
                    operation: "squeeze".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?
                .squeeze(0)
                .map_err(|e| UnifiedError::Processing {
                    operation: "squeeze".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?
                .to_dtype(DType::F32)
                .map_err(|e| UnifiedError::Processing {
                    operation: "to_dtype".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;

            // Apply repeat penalty
            let logits = if self.config.repeat_penalty != 1.0 {
                let start_at = tokens.len().saturating_sub(self.config.repeat_last_n);
                apply_repeat_penalty(&logits, self.config.repeat_penalty, &tokens[start_at..])?
            } else {
                logits
            };

            // Sample next token
            let next_token = if self.config.temperature == 0.0 {
                // Greedy sampling
                sample_argmax(&logits)?
            } else {
                // Temperature + top-p sampling
                sample_topp(&logits, self.config.temperature, self.config.top_p)?
            };

            // Check for EOS
            if next_token == self.eos_token_id || next_token == self.im_end_token_id {
                break;
            }

            tokens.push(next_token);

            // Decode token
            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                generated_text.push_str(&piece);
            }
        }

        Ok(generated_text)
    }

    /// Generate guard output for safety classification
    ///
    /// # Arguments
    /// - `text`: Input text to classify
    /// - `mode`: Classification mode ("input" for user prompts, "output" for model responses)
    ///
    /// Returns raw generated text that should be parsed by the caller
    pub fn generate_guard(
        &mut self,
        text: &str,
        mode: &str,
    ) -> UnifiedResult<GuardGenerationResult> {
        let prompt = self.format_prompt(text, mode);
        let output = self.generate(&prompt)?;
        Ok(GuardGenerationResult { raw_output: output })
    }
}

// ================================================================================================
// Sampling Functions
// ================================================================================================

/// Apply repeat penalty to logits
fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> UnifiedResult<Tensor> {
    let logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| UnifiedError::Processing {
            operation: "to_vec1".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

    let mut modified_logits = logits_vec.clone();
    for &token_id in context {
        let idx = token_id as usize;
        if idx < modified_logits.len() {
            if modified_logits[idx] < 0.0 {
                modified_logits[idx] *= penalty;
            } else {
                modified_logits[idx] /= penalty;
            }
        }
    }

    Tensor::new(modified_logits, logits.device()).map_err(|e| UnifiedError::Processing {
        operation: "create tensor".to_string(),
        source: e.to_string(),
        input_context: None,
    })
}

/// Greedy sampling (argmax)
fn sample_argmax(logits: &Tensor) -> UnifiedResult<u32> {
    let logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| UnifiedError::Processing {
            operation: "to_vec1".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

    let max_idx = logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    Ok(max_idx as u32)
}

/// Top-p (nucleus) sampling
fn sample_topp(logits: &Tensor, temperature: f64, top_p: f64) -> UnifiedResult<u32> {
    let logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| UnifiedError::Processing {
            operation: "to_vec1".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

    // Apply temperature
    let scaled_logits: Vec<f32> = logits_vec.iter().map(|&x| x / temperature as f32).collect();

    // Compute softmax
    let max_logit = scaled_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = scaled_logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

    // Sort by probability
    let mut indexed_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Compute cumulative probabilities and apply top-p
    let mut cumulative_prob = 0.0;
    let mut sampled_probs = Vec::new();

    for (idx, prob) in indexed_probs.iter() {
        sampled_probs.push((*idx, *prob));
        cumulative_prob += prob;
        if cumulative_prob >= top_p as f32 {
            break;
        }
    }

    // Normalize filtered probabilities
    let filtered_sum: f32 = sampled_probs.iter().map(|(_, p)| p).sum();
    let normalized_probs: Vec<(usize, f32)> = sampled_probs
        .iter()
        .map(|(idx, p)| (*idx, p / filtered_sum))
        .collect();

    // Sample from distribution
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let random_value: f32 = rng.gen();

    let mut cumulative = 0.0;
    for (idx, prob) in normalized_probs.iter() {
        cumulative += prob;
        if random_value <= cumulative {
            return Ok(*idx as u32);
        }
    }

    // Fallback to last token
    Ok(normalized_probs
        .last()
        .map(|(idx, _)| *idx as u32)
        .unwrap_or(0))
}
