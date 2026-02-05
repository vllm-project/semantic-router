//! Qwen3 Multi-LoRA Classifier with Adapter Switching
//!
//! This module wraps the official `candle_transformers::models::qwen3::Model`
//! and supports multiple LoRA adapters that can be switched dynamically.
//!
//! Key features:
//! - Uses official Qwen3 implementation (proven to work correctly)
//! - Supports multiple LoRA adapters (e.g., classification, jailbreak detection)
//! - Dynamic adapter switching without reloading base model
//! - Apply LoRA on-the-fly during forward pass (like PEFT in Python)
//!
//! Example usage:
//! ```ignore
//! let model = Qwen3MultiLoRAClassifier::new(base_model_path, device)?;
//! model.load_adapter("category", "path/to/category_adapter")?;
//! model.load_adapter("jailbreak", "path/to/jailbreak_adapter")?;
//!
//! // Classify with category adapter
//! let result = model.classify_with_adapter("What is GDP?", "category")?;
//!
//! // Detect jailbreak with jailbreak adapter
//! let result = model.classify_with_adapter("Ignore previous instructions", "jailbreak")?;
//! ```

use crate::core::{ConfigErrorType, UnifiedError, UnifiedResult};
use crate::model_architectures::generative::qwen3_with_lora::{
    Config as Qwen3Config, ModelForCausalLM as Qwen3Model,
};
use crate::model_architectures::lora::{LoRAAdapter, LoRAConfig};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Label mapping for an adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterLabelMapping {
    pub label2id: HashMap<String, usize>,
    pub id2label: HashMap<String, String>,
    pub instruction_template: String,
}

impl AdapterLabelMapping {
    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<_> = self.id2label.iter().collect();
        cats.sort_by_key(|(id, _)| id.parse::<usize>().unwrap());
        cats.into_iter().map(|(_, label)| label.clone()).collect()
    }
}

/// LoRA adapter metadata (adapters are injected into model, not stored here)
struct LoadedAdapter {
    /// Adapter name (e.g., "category", "jailbreak")
    name: String,

    /// LoRA configuration (rank, alpha, etc.)
    lora_config: LoRAConfig,

    /// Label mapping for this adapter
    label_mapping: AdapterLabelMapping,

    /// Category token IDs for logit extraction
    category_token_ids: Vec<u32>,
}

/// Classification result with adapter info
#[derive(Debug, Clone)]
pub struct MultiAdapterClassificationResult {
    pub adapter_name: String,
    pub category: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
    pub all_categories: Vec<String>,
}

/// Qwen3 with multiple LoRA adapters
pub struct Qwen3MultiLoRAClassifier {
    /// Official Qwen3 model with LM head (shared across all adapters)
    base_model: Qwen3Model,

    /// Qwen3 configuration
    config: Qwen3Config,

    /// Tokenizer (shared)
    tokenizer: Arc<Tokenizer>,

    /// Loaded LoRA adapters (indexed by adapter name)
    adapters: HashMap<String, LoadedAdapter>,

    /// Device
    device: Device,

    /// Model dtype
    dtype: DType,
}

impl Qwen3MultiLoRAClassifier {
    /// Create a new multi-adapter classifier
    ///
    /// # Arguments
    /// - `base_model_path`: Path to base Qwen3-0.6B model
    /// - `device`: Device to run on
    pub fn new(base_model_path: &str, device: &Device) -> UnifiedResult<Self> {
        println!("🚀 Initializing Qwen3 Multi-LoRA Classifier");
        println!("  Base model: {}", base_model_path);

        let base_dir = Path::new(base_model_path);

        // Load config
        let config_path = base_dir.join("config.json");
        let config: Qwen3Config =
            serde_json::from_slice(&std::fs::read(config_path)?).map_err(|e| {
                UnifiedError::Configuration {
                    operation: "parse config".to_string(),
                    source: ConfigErrorType::ParseError(e.to_string()),
                    context: None,
                }
            })?;

        println!(
            "  Config: hidden_size={}, layers={}, vocab={}",
            config.hidden_size, config.num_hidden_layers, config.vocab_size
        );

        // Load tokenizer
        let tokenizer_path = base_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| UnifiedError::Configuration {
                operation: "load tokenizer".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            })?;

        // Determine dtype
        let dtype = if device.is_cuda() || device.is_metal() {
            DType::BF16
        } else {
            DType::F32
        };
        println!("  Using dtype: {:?}", dtype);

        // Load base model weights (support both single and sharded models)
        println!("  Loading base model weights...");

        let vb = {
            let single_weights_path = base_dir.join("model.safetensors");
            let index_path = base_dir.join("model.safetensors.index.json");

            if single_weights_path.exists() {
                // Single file model
                println!("  Using single model file");
                unsafe {
                    VarBuilder::from_mmaped_safetensors(&[single_weights_path], dtype, device)
                }
            } else if index_path.exists() {
                // Sharded model - read index to get all shard files
                println!("  Using sharded model files");
                let index_content =
                    std::fs::read_to_string(&index_path).map_err(|e| UnifiedError::Model {
                        model_type: crate::core::ModelErrorType::Embedding,
                        operation: "read model index".to_string(),
                        source: e.to_string(),
                        context: None,
                    })?;

                let index: serde_json::Value =
                    serde_json::from_str(&index_content).map_err(|e| {
                        UnifiedError::Configuration {
                            operation: "parse model index".to_string(),
                            source: ConfigErrorType::ParseError(e.to_string()),
                            context: None,
                        }
                    })?;

                // Extract unique shard filenames from weight_map
                let mut shard_files = std::collections::HashSet::new();
                if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                    for shard_name in weight_map.values() {
                        if let Some(name) = shard_name.as_str() {
                            shard_files.insert(name);
                        }
                    }
                }

                // Convert to sorted paths
                let mut shard_paths: Vec<PathBuf> = shard_files
                    .into_iter()
                    .map(|name| base_dir.join(name))
                    .collect();
                shard_paths.sort();

                println!("  Loading {} shard files", shard_paths.len());

                unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, device) }
            } else {
                return Err(UnifiedError::Configuration {
                    operation: "find model weights".to_string(),
                    source: ConfigErrorType::FileNotFound(format!(
                        "Neither {:?} nor {:?} found",
                        single_weights_path, index_path
                    )),
                    context: None,
                });
            }
        }
        .map_err(|e| UnifiedError::Model {
            model_type: crate::core::ModelErrorType::Embedding,
            operation: "load base weights".to_string(),
            source: e.to_string(),
            context: None,
        })?;

        // Build ModelForCausalLM (includes LM head)
        println!("  🏗️  Building Qwen3 model with LM head...");
        let base_model = Qwen3Model::new(&config, vb).map_err(|e| UnifiedError::Model {
            model_type: crate::core::ModelErrorType::Embedding,
            operation: "build Qwen3 ModelForCausalLM".to_string(),
            source: e.to_string(),
            context: None,
        })?;

        println!("✅ Qwen3 model loaded successfully\n");

        Ok(Self {
            base_model,
            config,
            tokenizer: Arc::new(tokenizer),
            adapters: HashMap::new(),
            device: device.clone(),
            dtype,
        })
    }

    /// Load a LoRA adapter
    ///
    /// # Arguments
    /// - `adapter_name`: Name for this adapter (e.g., "category", "jailbreak")
    /// - `adapter_path`: Path to adapter directory containing adapter_model.safetensors
    pub fn load_adapter(&mut self, adapter_name: &str, adapter_path: &str) -> UnifiedResult<()> {
        println!("📦 Loading LoRA adapter '{}'", adapter_name);
        println!("  Path: {}", adapter_path);

        let adapter_dir = Path::new(adapter_path);

        // Load adapter config
        let adapter_config_path = adapter_dir.join("adapter_config.json");
        let adapter_config_json = std::fs::read_to_string(&adapter_config_path)?;
        let adapter_config: serde_json::Value = serde_json::from_str(&adapter_config_json)?;

        let r = adapter_config["r"].as_u64().unwrap_or(16) as usize;
        let alpha = adapter_config["lora_alpha"].as_f64().unwrap_or(32.0);
        let dropout = adapter_config["lora_dropout"].as_f64().unwrap_or(0.05);

        println!(
            "  LoRA config: r={}, alpha={}, dropout={}",
            r, alpha, dropout
        );

        let lora_config = LoRAConfig {
            rank: r,
            alpha,
            dropout,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            use_bias: false,
            init_method: crate::model_architectures::lora::LoRAInitMethod::Kaiming,
        };

        // Load label mapping
        let label_mapping_path = adapter_dir.join("label_mapping.json");
        let label_mapping: AdapterLabelMapping = if label_mapping_path.exists() {
            serde_json::from_str(&std::fs::read_to_string(label_mapping_path)?)?
        } else {
            println!("  ⚠️  No label_mapping.json found, using default");
            AdapterLabelMapping {
                label2id: HashMap::new(),
                id2label: HashMap::new(),
                instruction_template: String::new(),
            }
        };

        println!("  Categories: {}", label_mapping.categories().len());

        // Load LoRA weights
        let adapter_weights_path = adapter_dir.join("adapter_model.safetensors");
        if !adapter_weights_path.exists() {
            return Err(UnifiedError::Configuration {
                operation: "find adapter weights".to_string(),
                source: ConfigErrorType::FileNotFound(format!("{:?}", adapter_weights_path)),
                context: Some("Expected adapter_model.safetensors".to_string()),
            });
        }

        let lora_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[adapter_weights_path], self.dtype, &self.device)
        }
        .map_err(|e| UnifiedError::Model {
            model_type: crate::core::ModelErrorType::Embedding,
            operation: "load LoRA weights".to_string(),
            source: e.to_string(),
            context: None,
        })?;

        // Load LoRA adapters for each layer
        let mut adapters = HashMap::new();
        let base_prefix = "base_model.model.model";

        // Load adapters for each transformer layer
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer_prefix = format!("{}.layers.{}", base_prefix, layer_idx);
            let layer_vb = lora_vb.pp(&layer_prefix);

            // Load attention projections
            for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                let key = format!("layers.{}.self_attn.{}", layer_idx, proj);
                if let Ok(adapter) = self.try_load_projection_adapter(
                    &layer_vb.pp(&format!("self_attn.{}", proj)),
                    &lora_config,
                    proj,
                ) {
                    adapters.insert(key, adapter);
                }
            }

            // Load MLP projections
            for proj in &["gate_proj", "up_proj", "down_proj"] {
                let key = format!("layers.{}.mlp.{}", layer_idx, proj);
                if let Ok(adapter) = self.try_load_mlp_adapter(
                    &layer_vb.pp(&format!("mlp.{}", proj)),
                    &lora_config,
                    proj,
                ) {
                    adapters.insert(key, adapter);
                }
            }
        }

        println!(
            "  Loaded {} LoRA adapters across {} layers",
            adapters.len(),
            self.config.num_hidden_layers
        );

        // Prepare category tokens
        let category_token_ids = Self::prepare_category_tokens(&self.tokenizer, &label_mapping)?;

        // Inject LoRA adapters into the model
        println!("  🔧 Injecting LoRA adapters into base model...");
        let adapters_arc: HashMap<String, Arc<LoRAAdapter>> = adapters
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();
        self.base_model.inject_lora_adapters(adapters_arc);

        // Store adapter metadata (adapters are now in the model)
        self.adapters.insert(
            adapter_name.to_string(),
            LoadedAdapter {
                name: adapter_name.to_string(),
                lora_config,
                label_mapping,
                category_token_ids,
            },
        );

        println!(
            "✅ Adapter '{}' loaded and injected successfully\n",
            adapter_name
        );

        Ok(())
    }

    /// Try to load a projection adapter (helper method)
    fn try_load_projection_adapter(
        &self,
        vb: &VarBuilder,
        lora_config: &LoRAConfig,
        _proj_name: &str,
    ) -> UnifiedResult<LoRAAdapter> {
        // Use head_dim from config (NOT calculated from hidden_size / num_heads)
        // Qwen3-0.6B has head_dim=128 explicitly set in config, not 64!
        let head_dim = self.config.head_dim;
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        // Determine input/output dimensions based on projection type
        let (input_dim, output_dim) = match _proj_name {
            "q_proj" => (hidden_size, num_heads * head_dim),
            "k_proj" | "v_proj" => (hidden_size, num_kv_heads * head_dim),
            "o_proj" => (num_heads * head_dim, hidden_size),
            _ => {
                return Err(UnifiedError::Configuration {
                    operation: "determine dimensions".to_string(),
                    source: ConfigErrorType::ParseError(format!(
                        "Unknown projection: {}",
                        _proj_name
                    )),
                    context: None,
                })
            }
        };

        LoRAAdapter::new(input_dim, output_dim, lora_config, vb.clone(), &self.device).map_err(
            |e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: format!("load {} LoRA", _proj_name),
                source: e.to_string(),
                context: None,
            },
        )
    }

    /// Try to load an MLP adapter (helper method)
    fn try_load_mlp_adapter(
        &self,
        vb: &VarBuilder,
        lora_config: &LoRAConfig,
        proj_name: &str,
    ) -> UnifiedResult<LoRAAdapter> {
        let intermediate_size = self.config.intermediate_size;
        let hidden_size = self.config.hidden_size;

        let (input_dim, output_dim) = match proj_name {
            "gate_proj" | "up_proj" => (hidden_size, intermediate_size),
            "down_proj" => (intermediate_size, hidden_size),
            _ => {
                return Err(UnifiedError::Configuration {
                    operation: "determine MLP dimensions".to_string(),
                    source: ConfigErrorType::ParseError(format!(
                        "Unknown MLP projection: {}",
                        proj_name
                    )),
                    context: None,
                })
            }
        };

        LoRAAdapter::new(input_dim, output_dim, lora_config, vb.clone(), &self.device).map_err(
            |e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: format!("load {} LoRA", proj_name),
                source: e.to_string(),
                context: None,
            },
        )
    }

    /// Prepare category token IDs for an adapter
    fn prepare_category_tokens(
        tokenizer: &Tokenizer,
        label_mapping: &AdapterLabelMapping,
    ) -> UnifiedResult<Vec<u32>> {
        let categories = label_mapping.categories();
        let mut token_ids = Vec::new();

        for category in &categories {
            let tokens = tokenizer
                .encode(format!(" {}", category), false)
                .map_err(|e| UnifiedError::Configuration {
                    operation: "tokenize category".to_string(),
                    source: ConfigErrorType::ParseError(e.to_string()),
                    context: Some(format!("category: {}", category)),
                })?;

            if let Some(&token_id) = tokens.get_ids().first() {
                token_ids.push(token_id);
            }
        }

        println!("  Prepared {} category tokens", token_ids.len());
        Ok(token_ids)
    }

    /// Format prompt for an adapter
    fn format_prompt(&self, text: &str, adapter_name: &str) -> UnifiedResult<String> {
        let adapter =
            self.adapters
                .get(adapter_name)
                .ok_or_else(|| UnifiedError::Configuration {
                    operation: "find adapter".to_string(),
                    source: ConfigErrorType::ParseError(format!(
                        "Adapter '{}' not found",
                        adapter_name
                    )),
                    context: None,
                })?;

        let instruction = if !adapter.label_mapping.instruction_template.is_empty() {
            adapter
                .label_mapping
                .instruction_template
                .replace("{question}", text)
        } else {
            let categories = adapter.label_mapping.categories().join(", ");
            format!(
                "You are an expert classifier. Classify the following into exactly ONE category. Respond with ONLY the category name.\n\nCategories: {}\n\nClassify:\n{}\nAnswer:",
                categories, text
            )
        };

        // ChatML format with thinking tags
        Ok(format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            instruction
        ))
    }

    /// Classify multiple texts with a specific adapter (batched inference)
    ///
    /// # Arguments
    /// - `texts`: Input texts to classify
    /// - `adapter_name`: Name of adapter to use (e.g., "category", "jailbreak")
    ///
    /// # Returns
    /// Classification results for all texts
    ///
    /// # Example
    /// ```ignore
    /// let texts = vec!["What is GDP?", "How to code?"];
    /// let results = model.classify_batch_with_adapter(&texts, "category")?;
    /// ```
    pub fn classify_batch_with_adapter(
        &mut self,
        texts: &[String],
        adapter_name: &str,
    ) -> UnifiedResult<Vec<MultiAdapterClassificationResult>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Get adapter
        let adapter =
            self.adapters
                .get(adapter_name)
                .ok_or_else(|| UnifiedError::Configuration {
                    operation: "find adapter".to_string(),
                    source: ConfigErrorType::ParseError(format!(
                        "Adapter '{}' not found",
                        adapter_name
                    )),
                    context: Some(format!("Available adapters: {:?}", self.list_adapters())),
                })?;

        // Clear KV cache before new classification
        self.base_model.clear_kv_cache();

        // Format prompts for all texts
        let mut prompts = Vec::new();
        for text in texts {
            prompts.push(self.format_prompt(text, adapter_name)?);
        }

        // Tokenize all prompts
        let mut tokenized_prompts = Vec::new();
        let mut max_len = 0;

        for prompt in &prompts {
            let encoding = self.tokenizer.encode(prompt.as_str(), true).map_err(|e| {
                UnifiedError::Configuration {
                    operation: "tokenize".to_string(),
                    source: ConfigErrorType::ParseError(e.to_string()),
                    context: None,
                }
            })?;

            let token_ids = encoding.get_ids().to_vec();
            max_len = max_len.max(token_ids.len());
            tokenized_prompts.push(token_ids);
        }

        // Pad all sequences to max_len
        let batch_size = texts.len();
        let mut padded_tokens = Vec::with_capacity(batch_size * max_len);

        for tokens in &tokenized_prompts {
            padded_tokens.extend_from_slice(tokens);
            // Pad with 0s (or use proper pad_token_id if available)
            for _ in tokens.len()..max_len {
                padded_tokens.push(0);
            }
        }

        // Create batched tensor [batch_size, seq_len]
        let input_ids = Tensor::from_vec(padded_tokens, (batch_size, max_len), &self.device)
            .map_err(|e| UnifiedError::Processing {
                operation: "create batched tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // Forward pass with batched input
        let logits = self
            .base_model
            .forward(&input_ids, 0)
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "batched forward".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        // Squeeze to [batch_size, vocab_size]
        let logits = logits.squeeze(1).map_err(|e| UnifiedError::Processing {
            operation: "squeeze logits".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        // Process each result in the batch
        let mut results = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            // Extract logits for this batch item
            let last_logits = logits
                .i(batch_idx)
                .map_err(|e| UnifiedError::Processing {
                    operation: "index batch".to_string(),
                    source: e.to_string(),
                    input_context: Some(format!("batch_idx={}", batch_idx)),
                })?
                .to_dtype(DType::F32)
                .map_err(|e| UnifiedError::Processing {
                    operation: "convert to f32".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;

            // Extract category logits
            let mut category_logits = Vec::new();
            for &token_id in &adapter.category_token_ids {
                let logit = last_logits
                    .i(token_id as usize)
                    .map_err(|e| UnifiedError::Processing {
                        operation: "extract logit".to_string(),
                        source: e.to_string(),
                        input_context: Some(format!("token_id={}", token_id)),
                    })?
                    .to_scalar::<f32>()
                    .map_err(|e| UnifiedError::Processing {
                        operation: "convert to scalar".to_string(),
                        source: e.to_string(),
                        input_context: None,
                    })?;
                category_logits.push(logit);
            }

            // Apply softmax
            let probabilities = softmax(&category_logits);

            // Find best category
            if probabilities.is_empty() {
                return Err(UnifiedError::Processing {
                    operation: "find best category".to_string(),
                    source: format!("No probabilities computed for batch item {}", batch_idx),
                    input_context: Some(format!(
                        "category_token_ids: {:?}",
                        adapter.category_token_ids
                    )),
                });
            }

            let max_idx = probabilities
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| UnifiedError::Processing {
                    operation: "find max probability".to_string(),
                    source: "Failed to find maximum probability".to_string(),
                    input_context: None,
                })?;

            let categories = adapter.label_mapping.categories();

            results.push(MultiAdapterClassificationResult {
                adapter_name: adapter_name.to_string(),
                category: categories[max_idx].clone(),
                confidence: probabilities[max_idx],
                probabilities: probabilities.clone(),
                all_categories: categories.clone(),
            });
        }

        Ok(results)
    }

    /// Classify with a specific adapter (single text)
    ///
    /// # Arguments
    /// - `text`: Input text to classify
    /// - `adapter_name`: Name of adapter to use (e.g., "category", "jailbreak")
    ///
    /// # Returns
    /// Classification result with probabilities for all categories
    ///
    /// # Example
    /// ```ignore
    /// let result = model.classify_with_adapter("What is GDP?", "category")?;
    /// println!("Category: {} ({:.1}%)", result.category, result.confidence * 100.0);
    /// ```
    pub fn classify_with_adapter(
        &mut self,
        text: &str,
        adapter_name: &str,
    ) -> UnifiedResult<MultiAdapterClassificationResult> {
        // Get adapter
        let adapter =
            self.adapters
                .get(adapter_name)
                .ok_or_else(|| UnifiedError::Configuration {
                    operation: "find adapter".to_string(),
                    source: ConfigErrorType::ParseError(format!(
                        "Adapter '{}' not found",
                        adapter_name
                    )),
                    context: Some(format!("Available adapters: {:?}", self.list_adapters())),
                })?;

        // Clear KV cache before new classification
        self.base_model.clear_kv_cache();

        // Format prompt
        let prompt = self.format_prompt(text, adapter_name)?;

        // Tokenize
        let encoding = self.tokenizer.encode(prompt.as_str(), true).map_err(|e| {
            UnifiedError::Configuration {
                operation: "tokenize".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            }
        })?;

        let token_ids = encoding.get_ids();
        let input_ids = Tensor::new(token_ids, &self.device)
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

        // Forward pass to get logits at last position
        // ModelForCausalLM::forward returns [batch, 1, vocab_size]
        let logits = self
            .base_model
            .forward(&input_ids, 0)
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "base forward".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        // Squeeze to [batch, vocab_size]
        let logits = logits.squeeze(1).map_err(|e| UnifiedError::Processing {
            operation: "squeeze logits".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        // TODO: Apply LoRA adapters to logits
        // For now, just use base model logits
        // In the future, we need to:
        // 1. Hook into the base model's forward pass
        // 2. Apply LoRA deltas to each layer's outputs
        // 3. Accumulate the effects through the model

        // Extract logits at batch=0
        let last_logits = logits
            .i(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "index batch".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .to_dtype(DType::F32)
            .map_err(|e| UnifiedError::Processing {
                operation: "convert to f32".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // Extract category logits
        let mut category_logits = Vec::new();
        for &token_id in &adapter.category_token_ids {
            let logit = last_logits
                .i(token_id as usize)
                .map_err(|e| UnifiedError::Processing {
                    operation: "extract logit".to_string(),
                    source: e.to_string(),
                    input_context: Some(format!("token_id={}", token_id)),
                })?
                .to_scalar::<f32>()
                .map_err(|e| UnifiedError::Processing {
                    operation: "convert to scalar".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;
            category_logits.push(logit);
        }

        // Apply softmax
        let probabilities = softmax(&category_logits);

        // Find best category
        let max_idx = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let categories = adapter.label_mapping.categories();

        Ok(MultiAdapterClassificationResult {
            adapter_name: adapter_name.to_string(),
            category: categories[max_idx].clone(),
            confidence: probabilities[max_idx],
            probabilities,
            all_categories: categories,
        })
    }

    /// List all loaded adapters
    pub fn list_adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    /// Get categories for a specific adapter
    pub fn get_adapter_categories(&self, adapter_name: &str) -> Option<Vec<String>> {
        self.adapters
            .get(adapter_name)
            .map(|adapter| adapter.label_mapping.categories())
    }

    /// Zero-shot classification with base model only (no adapter required)
    ///
    /// This method allows classification without a pre-trained adapter by:
    /// 1. Accepting categories as runtime input
    /// 2. Extracting category token IDs on-the-fly
    /// 3. Running inference with just the base model
    ///
    /// # Arguments
    /// - `text`: Input text to classify
    /// - `categories`: List of category names (e.g., ["positive", "negative", "neutral"])
    ///
    /// # Returns
    /// Classification result with probabilities for provided categories
    ///
    /// # Example
    /// ```ignore
    /// let categories = vec!["positive".to_string(), "negative".to_string()];
    /// let result = model.classify_zero_shot("I love this!", categories)?;
    /// println!("Sentiment: {} ({:.1}%)", result.category, result.confidence * 100.0);
    /// ```
    ///
    /// # Note
    /// - No LoRA weights applied (base model only)
    /// - Lower accuracy than fine-tuned adapters
    /// - Useful for quick testing or when no adapter is available
    /// - This method only works if the labels have only one token
    pub fn classify_zero_shot(
        &mut self,
        text: &str,
        categories: Vec<String>,
    ) -> UnifiedResult<MultiAdapterClassificationResult> {
        if categories.is_empty() {
            return Err(UnifiedError::Configuration {
                operation: "validate categories".to_string(),
                source: ConfigErrorType::ParseError("Categories list cannot be empty".to_string()),
                context: None,
            });
        }

        // Clear KV cache before new classification
        self.base_model.clear_kv_cache();

        // Format prompt for zero-shot classification
        let categories_str = categories.join(", ");
        let instruction = format!(
            "You are an expert classifier. Classify the following into exactly ONE category. Respond with ONLY the category name.\n\nCategories: {}\n\nClassify:\n{}\nAnswer:",
            categories_str, text
        );

        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            instruction
        );

        // Tokenize
        let encoding = self.tokenizer.encode(prompt.as_str(), true).map_err(|e| {
            UnifiedError::Configuration {
                operation: "tokenize".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            }
        })?;

        let token_ids = encoding.get_ids();
        let input_ids = Tensor::new(token_ids, &self.device)
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

        // Forward pass
        let logits = self
            .base_model
            .forward(&input_ids, 0)
            .map_err(|e| UnifiedError::Model {
                model_type: crate::core::ModelErrorType::Embedding,
                operation: "base forward".to_string(),
                source: e.to_string(),
                context: None,
            })?;

        let logits = logits.squeeze(1).map_err(|e| UnifiedError::Processing {
            operation: "squeeze logits".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        // Extract logits at batch=0
        let last_logits = logits
            .i(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "index batch".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| UnifiedError::Processing {
                operation: "convert to f32".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // Extract category token IDs on-the-fly (first token of each label)
        let mut category_token_ids = Vec::new();
        for category in &categories {
            let tokens = self
                .tokenizer
                .encode(format!(" {}", category), false)
                .map_err(|e| UnifiedError::Configuration {
                    operation: "tokenize category".to_string(),
                    source: ConfigErrorType::ParseError(e.to_string()),
                    context: Some(format!("category: {}", category)),
                })?;

            if let Some(&token_id) = tokens.get_ids().first() {
                category_token_ids.push(token_id);
            }
        }

        // Extract category logits
        let mut category_logits = Vec::new();
        for &token_id in &category_token_ids {
            let logit = last_logits
                .i(token_id as usize)
                .map_err(|e| UnifiedError::Processing {
                    operation: "extract logit".to_string(),
                    source: e.to_string(),
                    input_context: Some(format!("token_id={}", token_id)),
                })?
                .to_scalar::<f32>()
                .map_err(|e| UnifiedError::Processing {
                    operation: "convert to scalar".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;
            category_logits.push(logit);
        }

        // Apply softmax
        let probabilities = softmax(&category_logits);

        // Find best category
        let (best_idx, &best_confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let best_category = categories[best_idx].clone();

        Ok(MultiAdapterClassificationResult {
            adapter_name: "zero-shot".to_string(),
            category: best_category,
            confidence: best_confidence,
            probabilities,
            all_categories: categories,
        })
    }

    /// Zero-shot classification that properly handles multi-token labels by
    /// reusing the prompt KV cache and stepping through each label token.
    /// Keeping it in a separate method since it introduces more overhead.
    pub fn classify_zero_shot_multi_tokens(
        &mut self,
        text: &str,
        categories: Vec<String>,
    ) -> UnifiedResult<MultiAdapterClassificationResult> {
        if categories.is_empty() {
            return Err(UnifiedError::Configuration {
                operation: "validate categories".to_string(),
                source: ConfigErrorType::ParseError("Categories list cannot be empty".to_string()),
                context: None,
            });
        }

        // Clear KV cache before new classification
        self.base_model.clear_kv_cache();

        // Format prompt for zero-shot classification
        let categories_str = categories.join(", ");
        let instruction = format!(
            "You are an expert classifier. Classify the following into exactly ONE category. Respond with ONLY the category name.\n\nCategories: {}\n\nClassify:\n{}\nAnswer:",
            categories_str, text
        );

        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            instruction
        );

        // Tokenize prompt
        let prompt_encoding = self.tokenizer.encode(prompt.as_str(), true).map_err(|e| {
            UnifiedError::Configuration {
                operation: "tokenize".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            }
        })?;

        // Keep prompt tokens to reuse across category scoring
        // TODO: remove prompt_ids since it's only used to get length
        let prompt_ids: Vec<u32> = prompt_encoding.get_ids().to_vec();
        let prompt_token_ids = Tensor::new(prompt_encoding.get_ids(), &self.device)
            .map_err(|e| UnifiedError::Processing {
                operation: "create prompt tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .unsqueeze(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "unsqueeze prompt tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // base_logits is needed to predict the first category token
        let base_logits =
            self.base_model
                .forward(&prompt_token_ids, 0)
                .map_err(|e| UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "forward prompt".to_string(),
                    source: e.to_string(),
                    context: None,
                })?;
        let prompt_len = prompt_ids.len();
        // Snapshot prompt KV cache so each category can start from the same state
        let prompt_cache = self.base_model.kv_cache_snapshot();

        // Tokenize each category to full token sequences (handles multi-token labels)
        let mut category_token_seqs: Vec<Vec<u32>> = Vec::with_capacity(categories.len());
        for category in &categories {
            let tokens = self
                .tokenizer
                .encode(format!(" {}", category), false)
                .map_err(|e| UnifiedError::Configuration {
                    operation: "tokenize category".to_string(),
                    source: ConfigErrorType::ParseError(e.to_string()),
                    context: Some(format!("category: {}", category)),
                })?;
            let ids = tokens.get_ids();
            if ids.is_empty() {
                return Err(UnifiedError::Configuration {
                    operation: "tokenize category".to_string(),
                    source: ConfigErrorType::ParseError("Category produced no tokens".to_string()),
                    context: Some(format!("category: {}", category)),
                });
            }

            category_token_seqs.push(ids.to_vec());
        }
        // Compute log-probability for each category with a single forward per category,
        // reusing the prompt KV cache.
        let mut category_log_probs = Vec::with_capacity(categories.len());

        for (cat_idx, token_seq) in category_token_seqs.iter().enumerate() {
            // Restore prompt cache so each category starts from identical prefix state
            self.base_model.kv_cache_restore(&prompt_cache);
            let label_ids = Tensor::from_vec(token_seq.clone(), (1, token_seq.len()), &self.device)
                .map_err(|e| UnifiedError::Processing {
                    operation: "create label tensor".to_string(),
                    source: e.to_string(),
                    input_context: Some(format!("category={}", categories[cat_idx])),
                })?;

            // Forward all label tokens with prefix cache already populated
            let logits = self
                .base_model
                .forward_all(&label_ids, prompt_len)
                .map_err(|e| UnifiedError::Model {
                    model_type: crate::core::ModelErrorType::Embedding,
                    operation: "full forward".to_string(),
                    source: e.to_string(),
                    context: Some(format!("category={}", categories[cat_idx])),
                })?;
            // logits shape: [1, label_len, vocab]; score each label token
            let mut total_log_prob = 0f32;
            for (tok_idx, &token_id) in token_seq.iter().enumerate() {
                // for first token, use base_logits from prompt forward
                // for subsequent tokens, use logits from full forward, offset by 1
                // TODO: clean this up
                let step_logits = if tok_idx == 0 {
                    base_logits
                        .clone()
                        .i((0, 0))
                        .map_err(|e| UnifiedError::Processing {
                            operation: "index batch".to_string(),
                            source: e.to_string(),
                            input_context: None,
                        })?
                        .to_dtype(DType::F32)
                        .map_err(|e| UnifiedError::Processing {
                            operation: "convert to f32".to_string(),
                            source: e.to_string(),
                            input_context: None,
                        })?
                } else {
                    logits
                        .i((0, tok_idx - 1))
                        .map_err(|e| UnifiedError::Processing {
                            operation: "index logits".to_string(),
                            source: e.to_string(),
                            input_context: Some(format!("category={}", categories[cat_idx])),
                        })?
                        .to_dtype(DType::F32)
                        .map_err(|e| UnifiedError::Processing {
                            operation: "convert logits to f32".to_string(),
                            source: e.to_string(),
                            input_context: Some(format!("category={}", categories[cat_idx])),
                        })?
                };
                let logits_vec =
                    step_logits
                        .to_vec1::<f32>()
                        .map_err(|e| UnifiedError::Processing {
                            operation: "logits to vec".to_string(),
                            source: e.to_string(),
                            input_context: Some(format!("category={}", categories[cat_idx])),
                        })?;
                if (token_id as usize) >= logits_vec.len() {
                    return Err(UnifiedError::Processing {
                        operation: "token index out of bounds".to_string(),
                        source: format!(
                            "token_id {} out of vocab range {}",
                            token_id,
                            logits_vec.len()
                        ),
                        input_context: Some(format!("category={}", categories[cat_idx])),
                    });
                }
                // for each label token, compute log-probability
                let probs = softmax(&logits_vec);
                let prob = probs[token_id as usize].max(f32::MIN_POSITIVE);
                total_log_prob += prob.ln();
            }
            let norm_log_prob = total_log_prob / (token_seq.len() as f32);
            category_log_probs.push(norm_log_prob);
        }
        // Normalize log-probabilities across categories
        let probabilities = softmax(&category_log_probs);
        // Find best category
        let (best_idx, &best_confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let best_category = categories[best_idx].clone();

        Ok(MultiAdapterClassificationResult {
            adapter_name: "zero-shot".to_string(),
            category: best_category,
            confidence: best_confidence,
            probabilities,
            all_categories: categories,
        })
    }
}

/// Apply softmax
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum).collect()
}
