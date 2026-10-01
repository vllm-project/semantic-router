//! Versioned, content-addressed deployment contract. No model-name heuristics.
use crate::core::artifact_identity::ArtifactSnapshot;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum Axis {
    Fixed(usize),
    Symbol(String),
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Port {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<Axis>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub file: String,
    pub inputs: Vec<Port>,
    pub output: Port,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub repo_id: String,
    pub revision: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Embedding {
    pub dimension: usize,
    pub normalization: String,
    pub dimensions: Vec<usize>,
    pub text_pooling: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TextProcessor {
    pub padding_side: String,
    pub pad_token_id: u32,
    pub strip_whitespace: bool,
    pub reject_overflow: bool,
    pub instruction_api: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageProcessor {
    pub size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub resample: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Resample {
    pub method: String,
    pub lowpass_filter_width: usize,
    pub rolloff: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AudioProcessor {
    pub file: String,
    pub max_seconds: usize,
    pub sample_rates: Vec<usize>,
    pub max_sample_rate: usize,
    pub resample: Resample,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Processors {
    pub text: TextProcessor,
    pub image: ImageProcessor,
    pub audio: AudioProcessor,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Parity {
    pub file: String,
    pub passed: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format_version: u32,
    pub adapter: String,
    pub variant: String,
    pub source: Source,
    pub embedding: Embedding,
    pub tokenizer: String,
    pub max_text_length: usize,
    pub graphs: BTreeMap<String, Graph>,
    pub processors: Processors,
    pub files: BTreeMap<String, String>,
    pub reference_parity: Parity,
}

pub fn contained(root: &Path, file: &str) -> anyhow::Result<PathBuf> {
    let relative = Path::new(file);
    anyhow::ensure!(
        !file.is_empty()
            && !file.contains('\\')
            && relative
                .components()
                .all(|p| matches!(p, Component::Normal(_))),
        "artifact path must be relative and contained"
    );
    let path = root.join(relative).canonicalize()?;
    anyhow::ensure!(
        path.starts_with(root.canonicalize()?),
        "artifact symlink escapes model directory"
    );
    Ok(path)
}
fn port(name: &str, dtype: &str, shape: &[usize]) -> Port {
    Port {
        name: name.into(),
        dtype: dtype.into(),
        shape: shape.iter().map(|&v| Axis::Fixed(v)).collect(),
    }
}
impl Manifest {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.format_version == 1 && self.adapter == "vela_omni",
            "unsupported Omni artifact format"
        );
        let (dimension, limit, size, resample, pooling) = match self.variant.as_str() {
            "nano" => (384, 512, 512, "bicubic", "cls"),
            "mini" => (768, 32768, 384, "bilinear", "last_token_full_l2_prefix_l2"),
            _ => anyhow::bail!("unsupported Omni variant"),
        };
        anyhow::ensure!(
            !self.source.repo_id.is_empty()
                && self.source.revision.len() == 40
                && self.source.revision.bytes().all(|b| b.is_ascii_hexdigit()),
            "source must pin an immutable revision"
        );
        anyhow::ensure!(
            self.embedding.dimension == dimension
                && self.embedding.dimensions == [dimension]
                && self.embedding.normalization == "l2"
                && self.embedding.text_pooling == pooling
                && self.max_text_length == limit,
            "unsupported Omni representation"
        );
        let text = &self.processors.text;
        anyhow::ensure!(
            text.reject_overflow
                && text.strip_whitespace == (self.variant == "mini")
                && text.padding_side == "right",
            "unsupported text input policy"
        );
        anyhow::ensure!(
            text.instruction_api.as_deref()
                == if self.variant == "mini" {
                    Some("qwen_optional_instruction_v1")
                } else {
                    None
                },
            "unsupported instruction contract"
        );
        let image = &self.processors.image;
        anyhow::ensure!(
            image.size == size
                && image.resample == resample
                && image.mean == [0.5; 3]
                && image.std == [0.5; 3],
            "unsupported image processor"
        );
        let audio = &self.processors.audio;
        anyhow::ensure!(
            audio.max_seconds == 30
                && audio.max_sample_rate == 384000
                && (if self.variant == "nano" {
                    audio.sample_rates == [16000, 44100, 48000]
                } else {
                    audio.sample_rates.is_empty()
                })
                && audio.resample.method == "sinc_interp_hann"
                && audio.resample.lowpass_filter_width == 6
                && audio.resample.rolloff == 0.99,
            "unsupported audio processor"
        );
        anyhow::ensure!(
            self.graphs.len() == 4,
            "exactly four Omni graphs are required"
        );
        let text_ports = ["input_ids", "attention_mask"].map(|name| Port {
            name: name.into(),
            dtype: "int64".into(),
            shape: vec![Axis::Fixed(1), Axis::Symbol("sequence".into())],
        });
        let contracts = [
            ("text", text_ports.to_vec(), dimension),
            (
                "image",
                vec![port("pixel_values", "float32", &[1, 3, size, size])],
                dimension,
            ),
            (
                "clap",
                vec![port("input_features", "float32", &[1, 1, 1001, 64])],
                512,
            ),
            (
                "audio",
                vec![
                    port("input_features", "float32", &[1, 80, 3000]),
                    port("clap_embedding", "float32", &[1, 512]),
                ],
                dimension,
            ),
        ];
        for (name, inputs, dim) in contracts {
            let graph = self
                .graphs
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("missing {name} graph"))?;
            anyhow::ensure!(
                graph.inputs == inputs && graph.output == port("embedding", "float32", &[1, dim]),
                "invalid {name} graph contract"
            );
            anyhow::ensure!(
                self.files.contains_key(&graph.file),
                "graph absent from integrity manifest"
            );
        }
        anyhow::ensure!(
            self.reference_parity.passed,
            "artifact has no passed reference parity"
        );
        for path in [&self.tokenizer, &audio.file, &self.reference_parity.file] {
            anyhow::ensure!(
                self.files.contains_key(path),
                "required file absent from integrity manifest"
            );
        }
        for hash in self.files.values() {
            anyhow::ensure!(
                hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid file digest"
            );
        }
        Ok(())
    }
    pub fn load(root: &Path) -> anyhow::Result<(Self, Vec<ArtifactSnapshot>)> {
        let path = contained(root, "vela_omni_manifest.json")?;
        let mut snapshots = vec![ArtifactSnapshot::capture(&path, "manifest")?];
        let manifest: Self = serde_json::from_slice(&std::fs::read(&path)?)?;
        manifest.validate()?;
        for (file, hash) in &manifest.files {
            anyhow::ensure!(
                file != "vela_omni_manifest.json",
                "manifest cannot digest itself"
            );
            let snapshot = ArtifactSnapshot::capture(&contained(root, file)?, file)?;
            anyhow::ensure!(
                &snapshot.digest.sha256 == hash,
                "artifact digest differs: {file}"
            );
            snapshots.push(snapshot);
        }
        #[derive(Deserialize)]
        struct ParityCase {
            passed: bool,
        }
        #[derive(Deserialize)]
        struct Receipt {
            passed: bool,
            source: Source,
            variant: String,
            tests: Vec<ParityCase>,
        }
        let receipt: Receipt = serde_json::from_slice(&std::fs::read(contained(
            root,
            &manifest.reference_parity.file,
        )?)?)?;
        anyhow::ensure!(
            receipt.passed
                && receipt.source == manifest.source
                && receipt.variant == manifest.variant
                && !receipt.tests.is_empty()
                && receipt.tests.iter().all(|test| test.passed),
            "reference parity receipt does not validate this artifact source"
        );
        Ok((manifest, snapshots))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prevents_external_artifacts() {
        let root = tempfile::tempdir().unwrap();
        for name in ["../secret", "/tmp/secret", "a\\b", ""] {
            assert!(contained(root.path(), name).is_err());
        }
    }
    #[test]
    fn port_contract_preserves_dynamic_symbols() {
        let port: Port =
            serde_json::from_str(r#"{"name":"input_ids","dtype":"int64","shape":[1,"sequence"]}"#)
                .unwrap();
        assert_eq!(
            port.shape,
            vec![Axis::Fixed(1), Axis::Symbol("sequence".into())]
        );
        assert!(serde_json::from_str::<Port>(
            r#"{"name":"x","dtype":"float32","shape":[1],"fallback":true}"#
        )
        .is_err());
    }
}
