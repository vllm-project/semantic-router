//! The exported embedding representation is part of the model, not a backend option.

use serde::Deserialize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IntermediateNormalization {
    None,
    // Older Candle artifacts applied final_norm at every exit.
    #[default]
    FinalNorm,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingContract {
    version: u32,
    intermediate_normalization: String,
    final_normalization: String,
    pooling: String,
    pooling_accumulation_dtype: String,
    truncate_before_l2_normalize: bool,
}

impl IntermediateNormalization {
    pub fn from_model_config(config: &serde_json::Value) -> Result<Self, String> {
        let Some(value) = config.get("representation_contract") else {
            return Ok(Self::default());
        };
        let contract: EmbeddingContract =
            serde_json::from_value(value.clone()).map_err(|err| err.to_string())?;
        if contract.version != 1
            || contract.final_normalization != "final_norm"
            || contract.pooling != "attention_mask_mean"
            || contract.pooling_accumulation_dtype != "float32"
            || !contract.truncate_before_l2_normalize
        {
            return Err("unsupported embedding representation_contract".into());
        }
        match contract.intermediate_normalization.as_str() {
            "none" => Ok(Self::None),
            "final_norm" => Ok(Self::FinalNorm),
            _ => Err("unsupported intermediate_normalization".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn preserves_legacy_and_rejects_unknown_representation() {
        assert_eq!(
            IntermediateNormalization::from_model_config(&json!({})).unwrap(),
            IntermediateNormalization::FinalNorm
        );
        let mut config = json!({"representation_contract": {
            "version": 1,
            "intermediate_normalization": "none",
            "final_normalization": "final_norm",
            "pooling": "attention_mask_mean",
            "pooling_accumulation_dtype": "float32",
            "truncate_before_l2_normalize": true
        }});
        assert_eq!(
            IntermediateNormalization::from_model_config(&config).unwrap(),
            IntermediateNormalization::None
        );
        for (field, value) in [
            ("version", json!(2)),
            ("pooling", json!("cls")),
            ("intermediate_normalization", json!("typo")),
            ("final_normalization", json!("none")),
            ("pooling_accumulation_dtype", json!("float16")),
            ("truncate_before_l2_normalize", json!(false)),
            ("unknown", json!(true)),
        ] {
            let mut invalid = config.clone();
            invalid["representation_contract"][field] = value;
            assert!(IntermediateNormalization::from_model_config(&invalid).is_err());
        }
        config["representation_contract"] = json!(null);
        assert!(IntermediateNormalization::from_model_config(&config).is_err());
    }
}
