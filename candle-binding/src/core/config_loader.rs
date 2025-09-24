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
