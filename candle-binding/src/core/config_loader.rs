//! Unified Configuration Loader

use crate::core::unified_error::{config_errors, UnifiedError};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// Unified configuration loader for all model types
pub struct UnifiedConfigLoader;

impl UnifiedConfigLoader {
    /// Load and parse JSON configuration file from model path
    pub fn load_json_config(model_path: &str) -> Result<Value, UnifiedError> {
        let config_path = Path::new(model_path).join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|_e| config_errors::file_not_found(&config_path.to_string_lossy()))?;

        serde_json::from_str(&config_content).map_err(|e| {
            config_errors::invalid_json(&config_path.to_string_lossy(), &e.to_string())
        })
    }

    /// Load and parse JSON configuration file from specific path
    pub fn load_json_config_from_path(config_path: &str) -> Result<Value, UnifiedError> {
        let config_content = std::fs::read_to_string(config_path)
            .map_err(|_e| config_errors::file_not_found(config_path))?;

        serde_json::from_str(&config_content)
            .map_err(|e| config_errors::invalid_json(config_path, &e.to_string()))
    }

    /// Extract id2label mapping as HashMap<usize, String>
    pub fn extract_id2label_map(
        config_json: &Value,
    ) -> Result<HashMap<usize, String>, UnifiedError> {
        let id2label_json = config_json
            .get("id2label")
            .ok_or_else(|| config_errors::missing_field("id2label", "config.json"))?;

        let mut id2label = HashMap::new();
        if let Some(obj) = id2label_json.as_object() {
            for (id_str, label_value) in obj {
                let id: usize = id_str.parse().map_err(|e| {
                    config_errors::invalid_json(
                        "config.json",
                        &format!("Invalid id in id2label: {}", e),
                    )
                })?;

                let label = label_value
                    .as_str()
                    .ok_or_else(|| {
                        config_errors::invalid_json("config.json", "Label value is not a string")
                    })?
                    .to_string();

                id2label.insert(id, label);
            }
            Ok(id2label)
        } else {
            Err(config_errors::invalid_json(
                "config.json",
                "id2label is not an object",
            ))
        }
    }

    /// Extract id2label mapping as HashMap<String, String> (for string-based IDs)
    pub fn extract_id2label_string_map(
        config_json: &Value,
    ) -> Result<HashMap<String, String>, UnifiedError> {
        let id2label_json = config_json
            .get("id2label")
            .ok_or_else(|| config_errors::missing_field("id2label", "config.json"))?;

        let mut id2label = HashMap::new();
        if let Some(obj) = id2label_json.as_object() {
            for (id_str, label_value) in obj {
                if let Some(label) = label_value.as_str() {
                    id2label.insert(id_str.clone(), label.to_string());
                }
            }
            Ok(id2label)
        } else {
            Err(config_errors::invalid_json(
                "config.json",
                "id2label is not an object",
            ))
        }
    }

    /// Extract labels as sorted Vec<String> (sorted by ID)
    pub fn extract_sorted_labels(config_json: &Value) -> Result<Vec<String>, UnifiedError> {
        let id2label_json = config_json
            .get("id2label")
            .ok_or_else(|| config_errors::missing_field("id2label", "config.json"))?;

        if let Some(obj) = id2label_json.as_object() {
            let mut labels: Vec<(usize, String)> = Vec::new();

            for (id_str, label_value) in obj {
                if let (Ok(id), Some(label)) = (id_str.parse::<usize>(), label_value.as_str()) {
                    labels.push((id, label.to_string()));
                }
            }

            labels.sort_by_key(|&(id, _)| id);
            Ok(labels.into_iter().map(|(_, label)| label).collect())
        } else {
            Err(config_errors::invalid_json(
                "config.json",
                "id2label is not an object",
            ))
        }
    }

    /// Extract labels as Vec<String> with index-based ordering
    pub fn extract_indexed_labels(config_json: &Value) -> Result<Vec<String>, UnifiedError> {
        let id2label_json = config_json
            .get("id2label")
            .ok_or_else(|| config_errors::missing_field("id2label", "config.json"))?;

        if let Some(obj) = id2label_json.as_object() {
            // Try numeric IDs first
            let mut numeric_labels: Vec<(usize, String)> = Vec::new();
            for (id_str, label_value) in obj {
                if let (Ok(id), Some(label)) = (id_str.parse::<usize>(), label_value.as_str()) {
                    numeric_labels.push((id, label.to_string()));
                }
            }

            if !numeric_labels.is_empty() {
                numeric_labels.sort_by_key(|&(id, _)| id);
                return Ok(numeric_labels.into_iter().map(|(_, label)| label).collect());
            }

            // Fallback to string keys
            let labels: Vec<String> = obj
                .values()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect();

            if !labels.is_empty() {
                Ok(labels)
            } else {
                Err(config_errors::invalid_json(
                    "config.json",
                    "No valid id2label found",
                ))
            }
        } else {
            Err(config_errors::invalid_json(
                "config.json",
                "id2label is not an object",
            ))
        }
    }

    /// Extract number of classes from config
    pub fn extract_num_classes(config_json: &Value) -> usize {
        if let Some(id2label) = config_json.get("id2label").and_then(|v| v.as_object()) {
            id2label.len()
        } else {
            2 // Default fallback
        }
    }

    /// Extract hidden size from config
    pub fn extract_hidden_size(config_json: &Value) -> usize {
        config_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(768) as usize
    }

    /// Load LoRA configuration data
    pub fn load_lora_config(model_path: &str) -> Result<LoRAConfigData, UnifiedError> {
        let lora_config_path = Path::new(model_path).join("lora_config.json");
        let lora_config_content = std::fs::read_to_string(&lora_config_path)
            .map_err(|_e| config_errors::file_not_found(&lora_config_path.to_string_lossy()))?;

        let lora_config_json: Value = serde_json::from_str(&lora_config_content).map_err(|e| {
            config_errors::invalid_json(&lora_config_path.to_string_lossy(), &e.to_string())
        })?;

        LoRAConfigData::from_json(&lora_config_json)
    }
}

/// LoRA configuration data structure
#[derive(Debug, Clone)]
pub struct LoRAConfigData {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub task_type: String,
}

impl LoRAConfigData {
    /// Create LoRAConfigData from JSON value
    pub fn from_json(config_json: &Value) -> Result<Self, UnifiedError> {
        Ok(LoRAConfigData {
            rank: config_json.get("r").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
            alpha: config_json
                .get("lora_alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(32.0) as f32,
            dropout: config_json
                .get("lora_dropout")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32,
            target_modules: config_json
                .get("target_modules")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_else(|| vec!["query".to_string(), "value".to_string()]),
            task_type: config_json
                .get("task_type")
                .and_then(|v| v.as_str())
                .unwrap_or("FEATURE_EXTRACTION")
                .to_string(),
        })
    }
}

/// Model configuration structure
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub id2label: HashMap<usize, String>,
    pub label2id: HashMap<String, usize>,
    pub num_labels: usize,
    pub hidden_size: usize,
}

/// ModernBERT configuration structure
#[derive(Debug, Clone)]
pub struct ModernBertConfig {
    pub num_classes: usize,
    pub hidden_size: usize,
}

/// mmBERT configuration structure (multilingual ModernBERT)
///
/// mmBERT is a multilingual encoder built on ModernBERT architecture with:
/// - 256k vocabulary for 1800+ language support
/// - 8192 max position embeddings
/// - RoPE positional embeddings (sans_pos)
#[derive(Debug, Clone)]
pub struct MmBertConfig {
    pub num_classes: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub local_attention: usize,
    pub global_attn_every_n_layers: usize,
    pub is_multilingual: bool,
}

/// Token configuration structure
#[derive(Debug, Clone)]
pub struct TokenConfig {
    pub id2label: HashMap<usize, String>,
    pub label2id: HashMap<String, usize>,
    pub num_labels: usize,
    pub hidden_size: usize,
}

/// Configuration loader trait
pub trait ConfigLoader {
    type Output;

    fn load_from_path(path: &Path) -> Result<Self::Output, UnifiedError>;
}

/// Intent configuration loader
pub struct IntentConfigLoader;
impl ConfigLoader for IntentConfigLoader {
    type Output = Vec<String>;

    fn load_from_path(path: &Path) -> Result<Self::Output, UnifiedError> {
        let config_json = UnifiedConfigLoader::load_json_config(&path.to_string_lossy())?;
        UnifiedConfigLoader::extract_sorted_labels(&config_json)
    }
}

/// PII configuration loader
pub struct PIIConfigLoader;
impl ConfigLoader for PIIConfigLoader {
    type Output = Vec<String>;

    fn load_from_path(path: &Path) -> Result<Self::Output, UnifiedError> {
        let config_json = UnifiedConfigLoader::load_json_config(&path.to_string_lossy())?;
        UnifiedConfigLoader::extract_sorted_labels(&config_json)
    }
}

/// Security configuration loader
pub struct SecurityConfigLoader;
impl ConfigLoader for SecurityConfigLoader {
    type Output = Vec<String>;

    fn load_from_path(path: &Path) -> Result<Self::Output, UnifiedError> {
        let config_json = UnifiedConfigLoader::load_json_config(&path.to_string_lossy())?;
        UnifiedConfigLoader::extract_sorted_labels(&config_json)
    }
}

/// Token configuration loader
pub struct TokenConfigLoader;
impl ConfigLoader for TokenConfigLoader {
    type Output = TokenConfig;

    fn load_from_path(path: &Path) -> Result<Self::Output, UnifiedError> {
        let config_json = UnifiedConfigLoader::load_json_config(&path.to_string_lossy())?;
        let id2label = UnifiedConfigLoader::extract_id2label_map(&config_json)?;
        let label2id: HashMap<String, usize> = id2label
            .iter()
            .map(|(&id, label)| (label.clone(), id))
            .collect();
        let num_labels = id2label.len();
        let hidden_size = UnifiedConfigLoader::extract_hidden_size(&config_json);

        Ok(TokenConfig {
            id2label,
            label2id,
            num_labels,
            hidden_size,
        })
    }
}

/// LoRA configuration loader
pub struct LoRAConfigLoader;
impl ConfigLoader for LoRAConfigLoader {
    type Output = LoRAConfigData;

    fn load_from_path(path: &Path) -> Result<Self::Output, UnifiedError> {
        UnifiedConfigLoader::load_lora_config(&path.to_string_lossy())
    }
}

/// ModernBERT configuration loader
pub struct ModernBertConfigLoader;
impl ConfigLoader for ModernBertConfigLoader {
    type Output = ModernBertConfig;

    fn load_from_path(path: &Path) -> Result<Self::Output, UnifiedError> {
        let config_json = UnifiedConfigLoader::load_json_config(&path.to_string_lossy())?;
        let num_classes = UnifiedConfigLoader::extract_num_classes(&config_json);
        let hidden_size = UnifiedConfigLoader::extract_hidden_size(&config_json);

        Ok(ModernBertConfig {
            num_classes,
            hidden_size,
        })
    }
}

/// mmBERT configuration loader (multilingual ModernBERT)
pub struct MmBertConfigLoader;
impl ConfigLoader for MmBertConfigLoader {
    type Output = MmBertConfig;

    fn load_from_path(path: &Path) -> Result<Self::Output, UnifiedError> {
        let config_json = UnifiedConfigLoader::load_json_config(&path.to_string_lossy())?;
        let num_classes = UnifiedConfigLoader::extract_num_classes(&config_json);
        let hidden_size = UnifiedConfigLoader::extract_hidden_size(&config_json);

        // Extract mmBERT-specific configuration
        let vocab_size = config_json
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(256000) as usize;

        let max_position_embeddings = config_json
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(8192) as usize;

        let num_hidden_layers = config_json
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(22) as usize;

        let num_attention_heads = config_json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize;

        let intermediate_size = config_json
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(1152) as usize;

        let local_attention = config_json
            .get("local_attention")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;

        let global_attn_every_n_layers = config_json
            .get("global_attn_every_n_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;

        // Check if this is a multilingual model (vocab_size >= 200000)
        let is_multilingual = vocab_size >= 200000;

        Ok(MmBertConfig {
            num_classes,
            hidden_size,
            vocab_size,
            max_position_embeddings,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            local_attention,
            global_attn_every_n_layers,
            is_multilingual,
        })
    }
}

/// Detect if a model is mmBERT from its config.json
pub fn is_mmbert_model(model_path: &str) -> Result<bool, UnifiedError> {
    let config_json = UnifiedConfigLoader::load_json_config(model_path)?;

    let vocab_size = config_json
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let model_type = config_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let position_embedding_type = config_json
        .get("position_embedding_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // mmBERT has large vocab (256000) and uses modernbert architecture with sans_pos
    Ok(vocab_size >= 200000 && model_type == "modernbert" && position_embedding_type == "sans_pos")
}

/// Load mmBERT configuration from model path
pub fn load_mmbert_config(model_path: &str) -> Result<MmBertConfig, UnifiedError> {
    MmBertConfigLoader::load_from_path(std::path::Path::new(model_path))
}

/// Model configuration loader
pub struct ModelConfigLoader;
impl ConfigLoader for ModelConfigLoader {
    type Output = ModelConfig;

    fn load_from_path(path: &Path) -> Result<Self::Output, UnifiedError> {
        let config_json = UnifiedConfigLoader::load_json_config(&path.to_string_lossy())?;
        let id2label = UnifiedConfigLoader::extract_id2label_map(&config_json)?;
        let label2id: HashMap<String, usize> = id2label
            .iter()
            .map(|(&id, label)| (label.clone(), id))
            .collect();
        let num_labels = id2label.len();
        let hidden_size = UnifiedConfigLoader::extract_hidden_size(&config_json);

        Ok(ModelConfig {
            id2label,
            label2id,
            num_labels,
            hidden_size,
        })
    }
}

/// Load config for intent classification (replaces intent_lora.rs logic)
pub fn load_intent_labels(model_path: &str) -> Result<Vec<String>, UnifiedError> {
    let config_json = UnifiedConfigLoader::load_json_config(model_path)?;
    UnifiedConfigLoader::extract_sorted_labels(&config_json)
}

/// Load config for PII detection (replaces pii_lora.rs logic)
pub fn load_pii_labels(model_path: &str) -> Result<Vec<String>, UnifiedError> {
    let config_json = UnifiedConfigLoader::load_json_config(model_path)?;
    UnifiedConfigLoader::extract_sorted_labels(&config_json)
}

/// Load config for security detection (replaces security_lora.rs logic)
pub fn load_security_labels(model_path: &str) -> Result<Vec<String>, UnifiedError> {
    let config_json = UnifiedConfigLoader::load_json_config(model_path)?;
    UnifiedConfigLoader::extract_sorted_labels(&config_json)
}

/// Load id2label mapping from config file (replaces token_lora.rs logic)
pub fn load_id2label_from_config(
    config_path: &str,
) -> Result<HashMap<String, String>, UnifiedError> {
    let config_json = UnifiedConfigLoader::load_json_config_from_path(config_path)?;
    UnifiedConfigLoader::extract_id2label_string_map(&config_json)
}

/// Load labels from model config (replaces modernbert.rs logic)
pub fn load_labels_from_model_config(model_path: &str) -> Result<Vec<String>, UnifiedError> {
    let config_json = UnifiedConfigLoader::load_json_config(model_path)?;
    UnifiedConfigLoader::extract_indexed_labels(&config_json)
}

/// Load token config (replaces token_lora.rs logic)
pub fn load_token_config(
    model_path: &str,
) -> Result<(HashMap<usize, String>, HashMap<String, usize>, usize, usize), UnifiedError> {
    let config_json = UnifiedConfigLoader::load_json_config(model_path)?;
    let id2label = UnifiedConfigLoader::extract_id2label_map(&config_json)?;
    let label2id: HashMap<String, usize> = id2label
        .iter()
        .map(|(&id, label)| (label.clone(), id))
        .collect();
    let num_labels = id2label.len();
    let hidden_size = UnifiedConfigLoader::extract_hidden_size(&config_json);

    Ok((id2label, label2id, num_labels, hidden_size))
}

/// Load ModernBERT number of classes (replaces modernbert.rs logic)
pub fn load_modernbert_num_classes(model_path: &str) -> Result<usize, UnifiedError> {
    let config_json = UnifiedConfigLoader::load_json_config(model_path)?;
    Ok(UnifiedConfigLoader::extract_num_classes(&config_json))
}

/// Global configuration loader for main config.yaml
pub struct GlobalConfigLoader;

impl GlobalConfigLoader {
    /// Load threshold for intent classifier from config/config.yaml
    pub fn load_intent_threshold() -> Result<f32, UnifiedError> {
        let config_path = "config/config.yaml";
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|_| config_errors::file_not_found(config_path))?;

        // Parse YAML to find classifier.category_model.threshold
        Self::extract_yaml_threshold(&config_str, &["classifier", "category_model", "threshold"])
            .or_else(|| Self::extract_yaml_threshold(&config_str, &["bert_model", "threshold"]))
            .ok_or_else(|| {
                config_errors::missing_field("classifier.category_model.threshold", config_path)
            })
    }

    /// Load threshold for security classifier from config/config.yaml
    pub fn load_security_threshold() -> Result<f32, UnifiedError> {
        let config_path = "config/config.yaml";
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|_| config_errors::file_not_found(config_path))?;

        // Parse YAML to find prompt_guard.threshold
        Self::extract_yaml_threshold(&config_str, &["prompt_guard", "threshold"])
            .ok_or_else(|| config_errors::missing_field("prompt_guard.threshold", config_path))
    }

    /// Load threshold for PII classifier from config/config.yaml
    pub fn load_pii_threshold() -> Result<f32, UnifiedError> {
        let config_path = "config/config.yaml";
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|_| config_errors::file_not_found(config_path))?;

        // Parse YAML to find classifier.pii_model.threshold
        Self::extract_yaml_threshold(&config_str, &["classifier", "pii_model", "threshold"])
            .ok_or_else(|| {
                config_errors::missing_field("classifier.pii_model.threshold", config_path)
            })
    }

    /// Extract threshold value from YAML content using hierarchical path
    fn extract_yaml_threshold(yaml_content: &str, path: &[&str]) -> Option<f32> {
        let lines: Vec<&str> = yaml_content.lines().collect();
        let mut current_level = 0;
        let mut found_sections = vec![false; path.len()];

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let indent_level = (line.len() - line.trim_start().len()) / 2;

            // Reset found sections if we're at a higher level
            if indent_level <= current_level {
                for i in (indent_level / 2 + 1)..found_sections.len() {
                    found_sections[i] = false;
                }
            }

            current_level = indent_level;

            // Check if this line matches our current section
            if let Some(section_end) = trimmed.find(':') {
                let section_name = trimmed[..section_end].trim();
                let section_level = indent_level / 2;

                if section_level < path.len() && section_name == path[section_level] {
                    found_sections[section_level] = true;

                    // If this is the threshold line and all parent sections are found
                    if section_level == path.len() - 1
                        && found_sections[..path.len() - 1].iter().all(|&x| x)
                    {
                        if let Some(value_str) = trimmed.split(':').nth(1) {
                            if let Ok(threshold) = value_str.trim().parse::<f32>() {
                                if threshold > 0.0 && threshold <= 1.0 {
                                    return Some(threshold);
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }
}

/// Router configuration structure
#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub high_confidence_threshold: f32, // For high confidence requirement detection
    pub low_latency_threshold_ms: u64,  // For low latency requirement detection
    pub lora_baseline_score: f32,       // LoRA path baseline score
    pub traditional_baseline_score: f32, // Traditional path baseline score
    pub embedding_baseline_score: f32,  // Embedding model (Qwen3/Gemma) baseline score
    pub success_confidence_threshold: f32, // Success rate calculation threshold
    pub large_batch_threshold: usize,   // Large batch size threshold
    pub lora_default_execution_time_ms: u64, // LoRA default execution time
    pub traditional_default_execution_time_ms: u64, // Traditional default execution time
    pub default_confidence_threshold: f32, // Default confidence requirement
    pub default_max_latency_ms: u64,    // Default max latency
    pub default_batch_size: usize,      // Default batch size
    pub default_avg_execution_time_ms: u64, // Default average execution time
    pub lora_default_confidence: f32,   // LoRA default confidence
    pub traditional_default_confidence: f32, // Traditional default confidence
    pub lora_default_success_rate: f32, // LoRA default success rate
    pub traditional_default_success_rate: f32, // Traditional default success rate
    // Scoring weights for intelligent path selection
    pub multi_task_lora_weight: f32, // LoRA advantage for multi-task
    pub single_task_traditional_weight: f32, // Traditional advantage for single task
    pub large_batch_lora_weight: f32, // LoRA advantage for large batch
    pub small_batch_traditional_weight: f32, // Traditional advantage for small batch
    pub medium_batch_weight: f32,    // Weight for medium batch (neutral)
    pub high_confidence_lora_weight: f32, // LoRA advantage for high confidence
    pub low_confidence_traditional_weight: f32, // Traditional advantage for low confidence
    pub low_latency_lora_weight: f32, // LoRA advantage for low latency
    pub high_latency_traditional_weight: f32, // Traditional advantage for relaxed latency
    pub performance_history_weight: f32, // Weight for historical performance factor
    // Traditional model specific configurations
    pub traditional_bert_confidence_threshold: f32, // Traditional BERT confidence threshold
    pub traditional_modernbert_confidence_threshold: f32, // Traditional ModernBERT confidence threshold
    pub traditional_pii_detection_threshold: f32,         // Traditional PII detection threshold
    pub traditional_token_classification_threshold: f32, // Traditional token classification threshold
    pub traditional_dropout_prob: f32,                   // Traditional model dropout probability
    pub traditional_attention_dropout_prob: f32, // Traditional model attention dropout probability
    pub tie_break_confidence: f32,               // Confidence value for tie-breaking situations
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            high_confidence_threshold: 0.99,
            low_latency_threshold_ms: 2000,
            lora_baseline_score: 0.8,
            traditional_baseline_score: 0.7,
            embedding_baseline_score: 0.75, // Higher quality than Traditional, versatile
            success_confidence_threshold: 0.8,
            large_batch_threshold: 4,
            lora_default_execution_time_ms: 1345,
            traditional_default_execution_time_ms: 4567,
            default_confidence_threshold: 0.95,
            default_max_latency_ms: 5000,
            default_batch_size: 4,
            default_avg_execution_time_ms: 3000,
            lora_default_confidence: 0.99,
            traditional_default_confidence: 0.95,
            lora_default_success_rate: 0.98,
            traditional_default_success_rate: 0.95,
            // Balanced scoring weights (total weight per factor should be similar)
            multi_task_lora_weight: 0.3, // LoRA excels at parallel processing
            single_task_traditional_weight: 0.3, // Traditional stable for single tasks
            large_batch_lora_weight: 0.25, // LoRA good for large batches
            small_batch_traditional_weight: 0.25, // Traditional good for small batches
            medium_batch_weight: 0.1,    // Neutral weight for medium batches
            high_confidence_lora_weight: 0.25, // LoRA provides high confidence
            low_confidence_traditional_weight: 0.25, // Traditional sufficient for low confidence
            low_latency_lora_weight: 0.3, // LoRA is faster
            high_latency_traditional_weight: 0.1, // Traditional acceptable for relaxed timing
            performance_history_weight: 0.2, // Historical performance factor
            // Traditional model configurations
            traditional_bert_confidence_threshold: 0.95, // BERT confidence threshold
            traditional_modernbert_confidence_threshold: 0.8, // ModernBERT confidence threshold
            traditional_pii_detection_threshold: 0.5,    // PII detection threshold
            traditional_token_classification_threshold: 0.9, // Token classification threshold
            traditional_dropout_prob: 0.1,               // Dropout probability
            traditional_attention_dropout_prob: 0.1,     // Attention dropout probability
            tie_break_confidence: 0.5,                   // Neutral confidence for tie situations
        }
    }
}

impl GlobalConfigLoader {
    /// Load router configuration from config/config.yaml
    pub fn load_router_config() -> Result<RouterConfig, UnifiedError> {
        let config_path = "config/config.yaml";
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|_| config_errors::file_not_found(config_path))?;

        let mut router_config = RouterConfig::default();

        // Load router-specific configurations from YAML
        if let Some(value) =
            Self::extract_yaml_value(&config_str, &["router", "high_confidence_threshold"])
        {
            if let Ok(threshold) = value.parse::<f32>() {
                router_config.high_confidence_threshold = threshold;
            }
        }

        if let Some(value) =
            Self::extract_yaml_value(&config_str, &["router", "low_latency_threshold_ms"])
        {
            if let Ok(threshold) = value.parse::<u64>() {
                router_config.low_latency_threshold_ms = threshold;
            }
        }

        if let Some(value) =
            Self::extract_yaml_value(&config_str, &["router", "lora_baseline_score"])
        {
            if let Ok(score) = value.parse::<f32>() {
                router_config.lora_baseline_score = score;
            }
        }

        if let Some(value) =
            Self::extract_yaml_value(&config_str, &["router", "traditional_baseline_score"])
        {
            if let Ok(score) = value.parse::<f32>() {
                router_config.traditional_baseline_score = score;
            }
        }

        if let Some(value) =
            Self::extract_yaml_value(&config_str, &["router", "embedding_baseline_score"])
        {
            if let Ok(score) = value.parse::<f32>() {
                router_config.embedding_baseline_score = score;
            }
        }

        // Load success threshold
        if let Some(value) =
            Self::extract_yaml_value(&config_str, &["router", "success_confidence_threshold"])
        {
            if let Ok(threshold) = value.parse::<f32>() {
                router_config.success_confidence_threshold = threshold;
            }
        }

        Ok(router_config)
    }

    /// Load router configuration with fallback to defaults
    pub fn load_router_config_safe() -> RouterConfig {
        Self::load_router_config().unwrap_or_default()
    }

    /// Extract YAML value as string from hierarchical path
    fn extract_yaml_value(yaml_content: &str, path: &[&str]) -> Option<String> {
        let lines: Vec<&str> = yaml_content.lines().collect();
        let mut current_level = 0;
        let mut found_sections = vec![false; path.len()];

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let indent_level = (line.len() - line.trim_start().len()) / 2;

            // Reset found sections if we're at a higher level
            if indent_level <= current_level {
                for i in (indent_level / 2 + 1)..found_sections.len() {
                    found_sections[i] = false;
                }
            }

            current_level = indent_level;

            // Check if this line matches our current section
            if let Some(section_end) = trimmed.find(':') {
                let section_name = trimmed[..section_end].trim();
                let section_level = indent_level / 2;

                if section_level < path.len() && section_name == path[section_level] {
                    found_sections[section_level] = true;

                    // If this is the target line and all parent sections are found
                    if section_level == path.len() - 1
                        && found_sections[..path.len() - 1].iter().all(|&x| x)
                    {
                        if let Some(value_str) = trimmed.split(':').nth(1) {
                            return Some(value_str.trim().to_string());
                        }
                    }
                }
            }
        }

        None
    }
}
