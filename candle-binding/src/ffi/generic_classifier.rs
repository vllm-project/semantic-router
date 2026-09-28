use crate::core::config_loader::UnifiedConfigLoader;
use crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier;
use crate::BertClassifier;
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

#[derive(Debug)]
pub enum GenericClassifier {
    Bert(Box<BertClassifier>),
    ModernBert(Box<TraditionalModernBertClassifier>),
}

impl GenericClassifier {
    pub(super) fn new(model_id: &str, num_classes: usize, use_cpu: bool) -> Result<Self> {
        let (config_path, _, _, _) = BertClassifier::resolve_model_files(model_id)?;
        let config = UnifiedConfigLoader::load_json_config_from_path(&config_path)?;
        let model_type = match config.get("model_type") {
            None | Some(Value::Null) => "bert",
            Some(Value::String(model_type)) => model_type.as_str(),
            Some(_) => bail!("model_type must be a string"),
        };
        match model_type {
            "bert" => Ok(Self::Bert(Box::new(BertClassifier::new(
                model_id,
                num_classes,
                use_cpu,
            )?))),
            "modernbert" | "mmbert" | "mmbert32k" | "mmbert-32k" => {
                let directory = Path::new(&config_path)
                    .parent()
                    .and_then(Path::to_str)
                    .context("model configuration directory must be UTF-8")?;
                let classifier =
                    TraditionalModernBertClassifier::load_from_directory(directory, use_cpu)?;
                ensure!(
                    classifier.get_num_classes() == num_classes,
                    "configured number of classes {num_classes} does not match model head ({})",
                    classifier.get_num_classes()
                );
                Ok(Self::ModernBert(Box::new(classifier)))
            }
            _ => bail!("unsupported generic classifier model_type: {model_type}"),
        }
    }

    pub(super) fn classify_text(&self, text: &str) -> Result<(usize, f32)> {
        match self {
            Self::Bert(classifier) => classifier.classify_text(text),
            Self::ModernBert(classifier) => Ok(classifier.classify_text(text)?),
        }
    }

    pub(super) fn classify_text_with_probabilities(
        &self,
        text: &str,
    ) -> Result<(usize, f32, Vec<f32>)> {
        match self {
            Self::Bert(classifier) => classifier.classify_text_with_probabilities(text),
            Self::ModernBert(classifier) => Ok(classifier.classify_text_with_probabilities(text)?),
        }
    }
}
