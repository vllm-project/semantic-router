//! Representation identities for embedding models.
pub(crate) use crate::core::artifact_identity::{json_digest, ArtifactDigest, ArtifactSnapshot};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeIdentity {
    pub version: u32,
    pub model_type: &'static str,
    pub runtime: String,
    pub effective_config_sha256: String,
    pub tokenizer_sha256: String,
    pub artifacts: Vec<ArtifactDigest>,
    pub layer: usize,
    pub dimension: usize,
    pub max_sequence_length: usize,
    pub pooling_contract: &'static str,
}

pub fn tokenizer_digest(tokenizer: &tokenizers::Tokenizer) -> anyhow::Result<String> {
    let serialized = tokenizer
        .to_string(false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    json_digest(&serde_json::from_str::<serde_json::Value>(&serialized)?)
}

impl RuntimeIdentity {
    pub fn for_exit(&self, layer: usize, dimension: usize) -> Self {
        let mut result = self.clone();
        result.layer = layer;
        result.dimension = dimension;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn content_not_location_and_no_mutable_reread() {
        let directory = tempfile::tempdir().unwrap();
        let a = directory.path().join("a");
        let b = directory.path().join("b");
        std::fs::write(&a, b"weights").unwrap();
        std::fs::copy(&a, &b).unwrap();
        let captured = ArtifactSnapshot::capture(&a, "weights").unwrap();
        assert_eq!(
            captured.digest.sha256,
            ArtifactSnapshot::capture(&b, "weights")
                .unwrap()
                .digest
                .sha256
        );
        std::fs::write(&a, b"replacement weights").unwrap();
        assert!(captured.verify().is_err());
        assert_ne!(
            captured.digest.sha256,
            ArtifactSnapshot::capture(&a, "weights")
                .unwrap()
                .digest
                .sha256
        );
    }
    #[test]
    fn json_object_order_does_not_change_identity() {
        assert_eq!(
            json_digest(&serde_json::json!({"a":1,"b":2})).unwrap(),
            json_digest(&serde_json::json!({"b":2,"a":1})).unwrap()
        );
    }
}
