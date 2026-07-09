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
use crate::model_architectures::prefix_cache::{PrefixCache, PrefixCacheConfig};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

mod qwen3_guard_generation;
mod qwen3_guard_loading;
mod qwen3_guard_sampling;

use qwen3_guard_loading::load_guard_var_builder;
use qwen3_guard_sampling::{apply_repeat_penalty, sample_argmax, sample_topp};

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

    /// Prefix cache configuration
    pub prefix_cache: PrefixCacheConfig,
}

impl Default for Qwen3GuardConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0, // Deterministic by default
            top_p: 0.95,
            max_tokens: 512,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            prefix_cache: PrefixCacheConfig {
                enabled: true, // Enable by default for 2-3x speedup!
                verbose: false,
            },
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

    /// Prefix cache for "input" mode (USER prompt)
    prefix_cache_input: Option<PrefixCache>,

    /// Prefix cache for "output" mode (ASSISTANT response)
    prefix_cache_output: Option<PrefixCache>,

    /// Snapshot of KV cache state after prefilling the input prefix.
    /// Restored per-request instead of re-prefilling, saving ~640ms/request on CPU.
    kv_snapshot_input: Option<Vec<candle_nn::kv_cache::KvCache>>,

    /// Snapshot of KV cache state after prefilling the output prefix.
    kv_snapshot_output: Option<Vec<candle_nn::kv_cache::KvCache>>,
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
        println!("Initializing Qwen3Guard Model");
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
        let vb = load_guard_var_builder(base_dir, dtype, device)?;

        // Load model
        let model = Qwen3Model::new(&model_config, vb).map_err(|e| UnifiedError::Processing {
            operation: "load model".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        println!("Qwen3Guard model loaded successfully");

        let config = config.unwrap_or_default();

        let mut instance = Self {
            model,
            tokenizer: Arc::new(tokenizer),
            device: device.clone(),
            dtype,
            config,
            eos_token_id,
            im_end_token_id,
            prefix_cache_input: None,
            prefix_cache_output: None,
            kv_snapshot_input: None,
            kv_snapshot_output: None,
        };

        // Initialize prefix caches if enabled
        if instance.config.prefix_cache.enabled {
            if instance.config.prefix_cache.verbose {
                println!("🚀 Initializing prefix cache for Qwen3Guard...");
            }
            instance.initialize_prefix_caches()?;
            if instance.config.prefix_cache.verbose {
                println!("Prefix cache initialized successfully");
            }
        }

        Ok(instance)
    }

    /// Initialize prefix caches for both input and output modes
    ///
    /// This extracts the fixed part of the prompt template and tokenizes it once.
    /// The tokens are cached for reuse across all requests.
    ///
    /// Additionally, this pre-computes the KV cache state for each prefix by
    /// running a single forward pass (prefill) and snapshotting the result.
    /// Per-request inference then restores this snapshot via `kv_cache_restore`
    /// instead of re-prefilling the same fixed prefix every time, saving ~640ms
    /// per request on CPU. The snapshot/restore methods already exist on
    /// `ModelForCausalLM` (qwen3_with_lora.rs:509,514) and are used by the
    /// MultiLoRA classifier for the same optimization pattern.
    fn initialize_prefix_caches(&mut self) -> UnifiedResult<()> {
        // Initialize cache for "input" mode (USER prompt)
        let input_prefix = self.extract_fixed_prefix("input");
        let encoding = self
            .tokenizer
            .encode(input_prefix.as_str(), true)
            .map_err(|e| UnifiedError::Configuration {
                operation: "tokenize prefix (input)".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            })?;
        let input_tokens: Vec<u32> = encoding.get_ids().to_vec();

        if self.config.prefix_cache.verbose {
            println!("   Input mode prefix: {} tokens", input_tokens.len());
        }

        // Prefill the prefix once and snapshot the KV cache state.
        // This is the key optimization: the prefix is a fixed template, so
        // every process_prefix() call produces an identical KV state.
        // By snapshotting once at init and restoring per-request, we avoid
        // re-running a full transformer forward pass (~640ms on CPU) on every
        // single guard invocation.
        self.model.clear_kv_cache();
        self.model
            .process_prefix(&input_tokens)
            .map_err(|e| UnifiedError::Processing {
                operation: "prefill input prefix for KV snapshot".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        self.kv_snapshot_input = Some(self.model.kv_cache_snapshot());

        self.prefix_cache_input = Some(PrefixCache::new(input_tokens));

        // Initialize cache for "output" mode (ASSISTANT response)
        let output_prefix = self.extract_fixed_prefix("output");
        let encoding = self
            .tokenizer
            .encode(output_prefix.as_str(), true)
            .map_err(|e| UnifiedError::Configuration {
                operation: "tokenize prefix (output)".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            })?;
        let output_tokens: Vec<u32> = encoding.get_ids().to_vec();

        if self.config.prefix_cache.verbose {
            println!("   Output mode prefix: {} tokens", output_tokens.len());
        }

        // Same prefill + snapshot for output mode
        self.model.clear_kv_cache();
        self.model
            .process_prefix(&output_tokens)
            .map_err(|e| UnifiedError::Processing {
                operation: "prefill output prefix for KV snapshot".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        self.kv_snapshot_output = Some(self.model.kv_cache_snapshot());

        self.prefix_cache_output = Some(PrefixCache::new(output_tokens));

        // Clear the KV cache so the model starts clean for actual requests
        self.model.clear_kv_cache();

        Ok(())
    }

    /// Extract the fixed prefix part of the prompt (everything before user text)
    ///
    /// The format_prompt has this structure:
    /// [FIXED PREFIX] + {user_label}: {text} + [FIXED SUFFIX]
    ///
    /// We cache: [FIXED PREFIX] + {user_label}:
    /// Then append user text at runtime.
    fn extract_fixed_prefix(&self, mode: &str) -> String {
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
            {}: ",
            user_label, user_label, user_label, user_label, user_label
        )
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
        // Check if prefix cache is available and enabled
        let use_cache = match mode {
            "input" => self.prefix_cache_input.is_some(),
            "output" => self.prefix_cache_output.is_some(),
            _ => false,
        };

        let output = if use_cache {
            // Fast path: Use prefix cache
            self.generate_with_prefix_cache(text, mode)?
        } else {
            // Slow path: Full prompt processing
            let prompt = self.format_prompt(text, mode);
            self.generate(&prompt)?
        };

        Ok(GuardGenerationResult { raw_output: output })
    }
}
