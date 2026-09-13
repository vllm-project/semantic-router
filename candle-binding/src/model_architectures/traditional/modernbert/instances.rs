//! Explicit head binding over a loaded ModernBERT backbone.
//!
//! The caller deliberately selects the source backbone. Only head tensors are
//! read from the adapter artifact; a different directory is never assumed to
//! contain the same backbone. Each binding captures its tokenizer and labels.

use super::*;

/// A loaded encoder without any classifier tensors or label requirements.
/// Task-specific heads are independently owned views of this immutable model.
pub struct ModernBertBackbone {
    model: Arc<ModernBert>,
    config: Config,
    device: Device,
    variant: ModernBertVariant,
}

impl ModernBertBackbone {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let config_path = format!("{path}/config.json");
        let variant = ModernBertVariant::detect_from_config(&config_path)?;
        let raw = std::fs::read_to_string(&config_path)?;
        let mut config = TraditionalModernBertClassifier::parse_model_config(&raw)?;
        config.classifier_config = None;
        let weights = format!("{path}/model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
        let model = ModernBert::load(vb.clone(), &config)
            .or_else(|_| ModernBert::load(vb.pp("_orig_mod"), &config))?;
        drain_loader_queue(device);
        Ok(Self {
            model: Arc::new(model),
            config,
            device: device.clone(),
            variant,
        })
    }
}

struct HeadBinding {
    head: Option<FixedModernBertHead>,
    config: Config,
    tokenizer: Box<dyn DualPathTokenizer>,
    classifier: candle_nn::Linear,
    labels: HashMap<String, String>,
    pooling: ClassifierPooling,
}

fn load_head(
    path: &str,
    source: &Config,
    device: &Device,
    variant: ModernBertVariant,
) -> Result<HeadBinding> {
    let raw = std::fs::read_to_string(format!("{path}/config.json"))?;
    let raw_config: serde_json::Value = serde_json::from_str(&raw)?;
    let pooling = match raw_config["classifier_pooling"].as_str() {
        Some("cls") => ClassifierPooling::CLS,
        _ => ClassifierPooling::MEAN,
    };
    let config = TraditionalModernBertClassifier::parse_model_config(&raw)?;
    // Classifier metadata may differ; every backbone execution parameter must match.
    let mut actual = config.clone();
    let mut expected = source.clone();
    actual.classifier_config = None;
    expected.classifier_config = None;
    anyhow::ensure!(
        actual == expected,
        "head is incompatible with the selected backbone configuration"
    );
    let labels = crate::ffi::classify::load_id2label_from_config(&format!("{path}/config.json"))?;
    anyhow::ensure!(
        !labels.is_empty(),
        "head requires an explicit nonempty id2label mapping"
    );
    let tokenizer = Tokenizer::from_file(format!("{path}/tokenizer.json")).map_err(E::msg)?;
    anyhow::ensure!(
        tokenizer.get_vocab_size(true) <= config.vocab_size,
        "head tokenizer exceeds backbone vocabulary"
    );
    let max_length = TraditionalModernBertClassifier::resolve_sequence_length(&config, None)?;
    let tokenizer = TraditionalModernBertClassifier::tokenizer_for_config(
        tokenizer,
        &config,
        variant,
        device.clone(),
        max_length,
    )?;
    let weights = format!("{path}/model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
    let vb = if vb.contains_tensor("classifier.weight") {
        vb
    } else {
        vb.pp("_orig_mod")
    };
    let head = if vb.contains_tensor("head.dense.weight") {
        Some(FixedModernBertHead::load(vb.pp("head"), &config)?)
    } else {
        None
    };
    let classifier = candle_nn::Linear::new(
        vb.get((labels.len(), config.hidden_size), "classifier.weight")?,
        Some(vb.get((labels.len(),), "classifier.bias")?),
    );
    drain_loader_queue(device);
    Ok(HeadBinding {
        head,
        config,
        tokenizer,
        classifier,
        labels,
        pooling,
    })
}

macro_rules! impl_bindings {
    ($source:ty) => {
        impl $source {
            pub fn device(&self) -> &Device {
                &self.device
            }

            /// Load a sequence head using this instance's actual backbone weights.
            pub fn bind_sequence_head(
                &self,
                path: &str,
            ) -> Result<TraditionalModernBertClassifier> {
                let binding = load_head(path, &self.config, &self.device, self.variant)?;
                Ok(TraditionalModernBertClassifier {
                    model: Arc::clone(&self.model),
                    head: binding.head,
                    classifier: FixedModernBertClassifier {
                        classifier: binding.classifier,
                    },
                    classifier_pooling: binding.pooling,
                    tokenizer: binding.tokenizer,
                    device: self.device.clone(),
                    config: binding.config,
                    num_classes: binding.labels.len(),
                    variant: self.variant,
                })
            }

            /// Load a token head using this instance's actual backbone weights.
            pub fn bind_token_head(
                &self,
                path: &str,
            ) -> Result<TraditionalModernBertTokenClassifier> {
                let binding = load_head(path, &self.config, &self.device, self.variant)?;
                Ok(TraditionalModernBertTokenClassifier {
                    model: Arc::clone(&self.model),
                    head: binding.head,
                    classifier: FixedModernBertTokenClassifier {
                        classifier: binding.classifier,
                    },
                    tokenizer: binding.tokenizer,
                    device: self.device.clone(),
                    config: binding.config,
                    num_classes: binding.labels.len(),
                    labels: binding.labels,
                    variant: self.variant,
                })
            }
        }
    };
}

impl_bindings!(TraditionalModernBertClassifier);
impl_bindings!(TraditionalModernBertTokenClassifier);

impl_bindings!(ModernBertBackbone);
