//! Unified Jailbreak Model Factory
//!
//! This module provides automatic model type detection and initialization
//! for jailbreak/prompt injection detection models. It supports:
//!
//! - **ModernBERT**: Fast sequence classification models
//! - **DeBERTa V3**: High-accuracy models like ProtectAI prompt injection detector
//! - **Qwen3Guard**: Generative safety classification models
//!
//! ## Auto-Detection
//!
//! The factory automatically detects the model architecture by reading config.json:
//! - `model_type`: "bert", "deberta-v2" (for DeBERTa v3), "qwen3"
//! - `architectures`: ["BertForSequenceClassification"], ["DebertaV2ForSequenceClassification"], ["Qwen3ForCausalLM"]
//!
//! ## Performance & Concurrency
//!
//! - Uses `LazyLock` for static data (default labels) to avoid repeated allocations
//! - Uses `parking_lot::Mutex` instead of `std::sync::Mutex` for Qwen3Guard
//! - ModernBERT and DeBERTa models are lock-free after initialization (wrapped in Arc)
//! - Qwen3Guard requires a mutex because generation modifies internal state (prefix cache)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use candle_semantic_router::model_architectures::jailbreak_factory::JailbreakModelFactory;
//!
//! // Auto-detect and load from model ID
//! let classifier = JailbreakModelFactory::from_model_id(
//!     "protectai/deberta-v3-base-prompt-injection",
//!     false  // use_cpu
//! )?;
//!
//! // Classify text
//! let result = classifier.classify("Ignore previous instructions")?;
//! println!("Class: {}, Confidence: {}", result.label, result.confidence);
//! ```

use crate::core::{ConfigErrorType, ModelErrorType, UnifiedError, UnifiedResult};
use crate::model_architectures::generative::qwen3_guard::Qwen3GuardModel;
use crate::model_architectures::traditional::deberta_v3::DebertaV3Classifier;
use crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier;
use candle_core::Device;
use parking_lot::Mutex;
use serde_json::Value;
use std::path::Path;
use std::sync::{Arc, LazyLock};

/// Default fallback labels for ModernBERT models (initialized once using LazyLock)
static DEFAULT_MODERNBERT_LABELS: LazyLock<Vec<String>> =
    LazyLock::new(|| vec!["benign".to_string(), "jailbreak".to_string()]);

/// Default fallback labels for DeBERTa models (initialized once using LazyLock)
static DEFAULT_DEBERTA_LABELS: LazyLock<Vec<String>> =
    LazyLock::new(|| vec!["SAFE".to_string(), "INJECTION".to_string()]);

/// Jailbreak classification result
#[derive(Debug, Clone)]
pub struct JailbreakResult {
    /// Predicted class index (0 = benign/safe, 1 = jailbreak)
    pub class: usize,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Label name (e.g., "benign", "jailbreak", "INJECTION", "SAFE")
    pub label: String,
}

/// Unified jailbreak classifier trait
pub trait JailbreakClassifier: Send + Sync {
    /// Classify text for jailbreak/prompt injection
    fn classify(&self, text: &str) -> UnifiedResult<JailbreakResult>;

    /// Get model type name
    fn model_type_name(&self) -> &str;
}

/// ModernBERT jailbreak classifier wrapper
pub struct ModernBertJailbreakClassifier {
    model: Arc<TraditionalModernBertClassifier>,
    labels: Vec<String>,
}

impl JailbreakClassifier for ModernBertJailbreakClassifier {
    fn classify(&self, text: &str) -> UnifiedResult<JailbreakResult> {
        let (class, confidence) =
            self.model
                .classify_text(text)
                .map_err(|e| UnifiedError::Model {
                    model_type: ModelErrorType::ModernBERT,
                    operation: "classify_text".to_string(),
                    source: format!("ModernBERT classification failed: {}", e),
                    context: None,
                })?;

        // Get label from class index
        let label = self
            .labels
            .get(class)
            .cloned()
            .unwrap_or_else(|| format!("class_{}", class));

        Ok(JailbreakResult {
            class,
            confidence,
            label,
        })
    }

    fn model_type_name(&self) -> &str {
        "modernbert"
    }
}

/// DeBERTa V3 jailbreak classifier wrapper
pub struct DebertaJailbreakClassifier {
    model: Arc<DebertaV3Classifier>,
    labels: Vec<String>,
}

impl JailbreakClassifier for DebertaJailbreakClassifier {
    fn classify(&self, text: &str) -> UnifiedResult<JailbreakResult> {
        let (label, confidence) =
            self.model
                .classify_text(text)
                .map_err(|e| UnifiedError::Model {
                    model_type: ModelErrorType::Classifier,
                    operation: "classify_text".to_string(),
                    source: format!("DeBERTa classification failed: {}", e),
                    context: None,
                })?;

        // Find class index from label
        let class = self.labels.iter().position(|l| l == &label).unwrap_or(0);

        Ok(JailbreakResult {
            class,
            confidence,
            label,
        })
    }

    fn model_type_name(&self) -> &str {
        "deberta-v3"
    }
}

/// Qwen3Guard jailbreak classifier wrapper
///
/// Note: Uses parking_lot::Mutex instead of std::sync::Mutex.
/// The mutex is necessary because generate_guard() modifies internal state
/// (prefix cache, generation state).
pub struct Qwen3GuardJailbreakClassifier {
    model: Arc<Mutex<Qwen3GuardModel>>,
}

impl JailbreakClassifier for Qwen3GuardJailbreakClassifier {
    fn classify(&self, text: &str) -> UnifiedResult<JailbreakResult> {
        // Use "input" mode for jailbreak detection
        // parking_lot::Mutex has no poisoning, so we can just lock directly
        let mut model = self.model.lock();

        let result = model
            .generate_guard(text, "input")
            .map_err(|e| UnifiedError::Model {
                model_type: ModelErrorType::Classifier,
                operation: "generate_guard".to_string(),
                source: format!("Qwen3Guard generation failed: {}", e),
                context: None,
            })?;

        // Release lock before parsing (parsing doesn't need the model)
        drop(model);

        // Parse the raw output for safety classification
        let (class, label, confidence) = parse_qwen3_guard_output(&result.raw_output);

        Ok(JailbreakResult {
            class,
            confidence,
            label,
        })
    }

    fn model_type_name(&self) -> &str {
        "qwen3-guard"
    }
}

/// Parse Qwen3Guard output to extract classification
fn parse_qwen3_guard_output(output: &str) -> (usize, String, f32) {
    // Look for "Severity level:" line
    // Format: "Severity level: Safe" or "Severity level: Unsafe"
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Severity level:") {
            let severity = trimmed
                .strip_prefix("Severity level:")
                .unwrap_or("")
                .trim()
                .to_lowercase();

            match severity.as_str() {
                "safe" => return (0, "SAFE".to_string(), 0.95),
                "unsafe" => return (1, "UNSAFE".to_string(), 0.95),
                "controversial" => return (0, "SAFE".to_string(), 0.6),
                _ => {}
            }
        }
    }

    // Default to safe if parsing fails
    (0, "SAFE".to_string(), 0.5)
}

/// Model architecture type detected from config.json
#[derive(Debug, Clone, PartialEq)]
pub enum JailbreakModelArchitecture {
    ModernBert,
    DebertaV3,
    Qwen3Guard,
    Unknown,
}

/// Jailbreak model factory for auto-detection and loading
pub struct JailbreakModelFactory;

impl JailbreakModelFactory {
    /// Detect model architecture from config.json
    ///
    /// Reads the config.json file and examines:
    /// - `model_type` field
    /// - `architectures` array
    ///
    /// Supports both local paths and HuggingFace model IDs.
    ///
    /// ## Returns
    /// - `JailbreakModelArchitecture` enum value
    pub fn detect_architecture(model_path: &str) -> UnifiedResult<JailbreakModelArchitecture> {
        let config_path = Path::new(model_path).join("config.json");

        // Try to read config.json - either locally or download from HuggingFace
        let config_content = if config_path.exists() {
            // Local path exists, read directly
            std::fs::read_to_string(&config_path).map_err(|e| UnifiedError::Configuration {
                operation: "read_config".to_string(),
                source: ConfigErrorType::ParseError(format!("Failed to read config: {}", e)),
                context: Some(config_path.display().to_string()),
            })?
        } else if model_path.contains('/')
            && !model_path.starts_with('.')
            && !model_path.starts_with('/')
        {
            // Looks like a HuggingFace model ID (e.g., "org/model")
            // Try to fetch config.json from HuggingFace Hub
            use hf_hub::api::sync::Api;

            let api = Api::new().map_err(|e| UnifiedError::Configuration {
                operation: "hf_hub_api".to_string(),
                source: ConfigErrorType::ParseError(format!("Failed to create HF Hub API: {}", e)),
                context: Some(model_path.to_string()),
            })?;

            let repo = api.model(model_path.to_string());
            let config_file = repo
                .get("config.json")
                .map_err(|e| UnifiedError::Configuration {
                    operation: "fetch_config".to_string(),
                    source: ConfigErrorType::FileNotFound(format!(
                        "Failed to fetch config.json from HuggingFace: {}",
                        e
                    )),
                    context: Some(model_path.to_string()),
                })?;

            std::fs::read_to_string(&config_file).map_err(|e| UnifiedError::Configuration {
                operation: "read_cached_config".to_string(),
                source: ConfigErrorType::ParseError(format!("Failed to read cached config: {}", e)),
                context: Some(format!("{:?}", config_file)),
            })?
        } else {
            // Neither local path nor HuggingFace ID
            return Err(UnifiedError::Configuration {
                operation: "detect_architecture".to_string(),
                source: ConfigErrorType::FileNotFound(config_path.display().to_string()),
                context: Some("Not a valid local path or HuggingFace model ID".to_string()),
            });
        };

        let config: Value =
            serde_json::from_str(&config_content).map_err(|e| UnifiedError::Configuration {
                operation: "parse_config_json".to_string(),
                source: ConfigErrorType::ParseError(format!("Failed to parse JSON: {}", e)),
                context: Some(config_path.display().to_string()),
            })?;

        // Check model_type field
        if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
            match model_type.to_lowercase().as_str() {
                "bert" | "modernbert" => return Ok(JailbreakModelArchitecture::ModernBert),
                "deberta" | "deberta-v2" => return Ok(JailbreakModelArchitecture::DebertaV3),
                "qwen2" | "qwen3" => return Ok(JailbreakModelArchitecture::Qwen3Guard),
                _ => {}
            }
        }

        // Check architectures array
        if let Some(architectures) = config.get("architectures").and_then(|v| v.as_array()) {
            for arch in architectures {
                if let Some(arch_str) = arch.as_str() {
                    let arch_lower = arch_str.to_lowercase();
                    if arch_lower.contains("modernbert")
                        || (arch_lower.contains("bert") && !arch_lower.contains("deberta"))
                    {
                        return Ok(JailbreakModelArchitecture::ModernBert);
                    } else if arch_lower.contains("deberta") {
                        return Ok(JailbreakModelArchitecture::DebertaV3);
                    } else if arch_lower.contains("qwen") {
                        return Ok(JailbreakModelArchitecture::Qwen3Guard);
                    }
                }
            }
        }

        Ok(JailbreakModelArchitecture::Unknown)
    }

    /// Load jailbreak classifier from model ID with auto-detection
    ///
    /// ## Arguments
    /// - `model_id`: HuggingFace model ID or local path
    /// - `use_cpu`: Force CPU inference
    ///
    /// ## Returns
    /// - Boxed trait object implementing `JailbreakClassifier`
    ///
    /// ## Example
    /// ```ignore
    /// let classifier = JailbreakModelFactory::from_model_id(
    ///     "protectai/deberta-v3-base-prompt-injection",
    ///     false
    /// )?;
    /// ```
    pub fn from_model_id(
        model_id: &str,
        use_cpu: bool,
    ) -> UnifiedResult<Box<dyn JailbreakClassifier>> {
        println!(
            "🔍 Auto-detecting jailbreak model architecture: {}",
            model_id
        );

        let architecture = Self::detect_architecture(model_id)?;

        println!("✅ Detected architecture: {:?}", architecture);

        match architecture {
            JailbreakModelArchitecture::ModernBert => Self::load_modernbert(model_id, use_cpu),
            JailbreakModelArchitecture::DebertaV3 => Self::load_deberta_v3(model_id, use_cpu),
            JailbreakModelArchitecture::Qwen3Guard => Self::load_qwen3_guard(model_id, use_cpu),
            JailbreakModelArchitecture::Unknown => Err(UnifiedError::Model {
                model_type: ModelErrorType::Classifier,
                operation: "detect_architecture".to_string(),
                source: format!(
                    "Unknown or unsupported model architecture for: {}",
                    model_id
                ),
                context: None,
            }),
        }
    }

    /// Load ModernBERT jailbreak classifier
    fn load_modernbert(
        model_id: &str,
        use_cpu: bool,
    ) -> UnifiedResult<Box<dyn JailbreakClassifier>> {
        println!("📦 Loading ModernBERT jailbreak classifier...");

        let model = TraditionalModernBertClassifier::load_from_directory(model_id, use_cpu)
            .map_err(|e| UnifiedError::Model {
                model_type: ModelErrorType::ModernBERT,
                operation: "load_from_directory".to_string(),
                source: format!("Failed to load ModernBERT: {}", e),
                context: Some(model_id.to_string()),
            })?;

        // Load labels from config, fallback to static default labels (no allocation if not needed)
        let labels = crate::core::config_loader::load_labels_from_model_config(model_id)
            .unwrap_or_else(|_| DEFAULT_MODERNBERT_LABELS.clone());

        println!(
            "✅ ModernBERT jailbreak classifier loaded with {} classes",
            labels.len()
        );

        Ok(Box::new(ModernBertJailbreakClassifier {
            model: Arc::new(model),
            labels,
        }))
    }

    /// Load DeBERTa V3 jailbreak classifier
    fn load_deberta_v3(
        model_id: &str,
        use_cpu: bool,
    ) -> UnifiedResult<Box<dyn JailbreakClassifier>> {
        println!("📦 Loading DeBERTa V3 jailbreak classifier...");

        let model =
            DebertaV3Classifier::new(model_id, use_cpu).map_err(|e| UnifiedError::Model {
                model_type: ModelErrorType::Classifier,
                operation: "new".to_string(),
                source: format!("Failed to load DeBERTa V3: {}", e),
                context: Some(model_id.to_string()),
            })?;

        // Load labels from config, fallback to static default labels (no allocation if not needed)
        let labels = crate::core::config_loader::load_labels_from_model_config(model_id)
            .unwrap_or_else(|_| DEFAULT_DEBERTA_LABELS.clone());

        println!(
            "✅ DeBERTa V3 jailbreak classifier loaded with {} classes",
            labels.len()
        );

        Ok(Box::new(DebertaJailbreakClassifier {
            model: Arc::new(model),
            labels,
        }))
    }

    /// Load Qwen3Guard jailbreak classifier
    fn load_qwen3_guard(
        model_id: &str,
        use_cpu: bool,
    ) -> UnifiedResult<Box<dyn JailbreakClassifier>> {
        println!("📦 Loading Qwen3Guard jailbreak classifier...");

        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };

        let model =
            Qwen3GuardModel::new(model_id, &device, None).map_err(|e| UnifiedError::Model {
                model_type: ModelErrorType::Classifier,
                operation: "new".to_string(),
                source: format!("Failed to load Qwen3Guard: {}", e),
                context: Some(model_id.to_string()),
            })?;

        println!("✅ Qwen3Guard jailbreak classifier loaded");

        Ok(Box::new(Qwen3GuardJailbreakClassifier {
            model: Arc::new(Mutex::new(model)),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_qwen3_guard_safe() {
        let output = "Reasoning: This is a normal query.\nCategory: None\nSeverity level: Safe";
        let (class, label, confidence) = parse_qwen3_guard_output(output);
        assert_eq!(class, 0);
        assert_eq!(label, "SAFE");
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_parse_qwen3_guard_unsafe() {
        let output = "Reasoning: Jailbreak attempt.\nCategory: Jailbreak\nSeverity level: Unsafe";
        let (class, label, confidence) = parse_qwen3_guard_output(output);
        assert_eq!(class, 1);
        assert_eq!(label, "UNSAFE");
        assert!(confidence > 0.9);
    }
}
