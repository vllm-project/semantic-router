//! Content captured at model initialization; no repository/path based identities.
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{File, Metadata};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactDigest {
    pub role: String,
    pub sha256: String,
}

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

pub fn json_digest(value: &impl Serialize) -> anyhow::Result<String> {
    // Value uses sorted object keys, including tokenizer vocabulary maps.
    let value = serde_json::to_value(value)?;
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?)))
}

pub fn tokenizer_digest(tokenizer: &tokenizers::Tokenizer) -> anyhow::Result<String> {
    let serialized = tokenizer
        .to_string(false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    json_digest(&serde_json::from_str::<serde_json::Value>(&serialized)?)
}

// The files are immutable model inputs. Catch replacement or writes during load;
// a runtime descriptor is not a monitor for unsupported in-place weight mutation.
pub struct ArtifactSnapshot {
    path: PathBuf,
    metadata: Metadata,
    pub digest: ArtifactDigest,
}

impl ArtifactSnapshot {
    pub fn capture(path: &Path, role: impl Into<String>) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        anyhow::ensure!(
            metadata.is_file(),
            "embedding artifact is not a regular file"
        );
        let mut reader = BufReader::new(file);
        let mut hash = Sha256::new();
        let mut buffer = [0_u8; 65536];
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        let result = Self {
            path: path.to_path_buf(),
            metadata,
            digest: ArtifactDigest {
                role: role.into(),
                sha256: format!("{:x}", hash.finalize()),
            },
        };
        result.verify()?;
        Ok(result)
    }

    pub fn verify(&self) -> anyhow::Result<()> {
        let now = std::fs::metadata(&self.path)?;
        let stable =
            now.len() == self.metadata.len() && now.modified()? == self.metadata.modified()?;
        #[cfg(unix)]
        let stable = {
            use std::os::unix::fs::MetadataExt;
            stable
                && now.ino() == self.metadata.ino()
                && now.dev() == self.metadata.dev()
                && now.ctime() == self.metadata.ctime()
                && now.ctime_nsec() == self.metadata.ctime_nsec()
        };
        anyhow::ensure!(
            stable,
            "embedding artifact changed while the model was loading"
        );
        Ok(())
    }
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
