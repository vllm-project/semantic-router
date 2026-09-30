//! Immutable content identities shared by owned ONNX sessions.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, Metadata};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactDigest {
    pub role: String,
    pub sha256: String,
}

pub fn json_digest(value: &impl Serialize) -> anyhow::Result<String> {
    // Value uses sorted object keys, including tokenizer vocabulary maps.
    let value = serde_json::to_value(value)?;
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?)))
}

// The files are immutable model inputs. Catch replacement or writes during load;
// a runtime descriptor is not a monitor for unsupported in-place weight mutation.
#[derive(Clone)]
pub struct ArtifactSnapshot {
    path: PathBuf,
    metadata: Metadata,
    pub digest: ArtifactDigest,
}

impl ArtifactSnapshot {
    pub fn capture(path: &Path, role: impl Into<String>) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        anyhow::ensure!(metadata.is_file(), "artifact is not a regular file");
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
        anyhow::ensure!(stable, "artifact changed while the model was loading");
        Ok(())
    }
}
