//! Manifest-driven Vela Omni deployment, independent of Hub repository names.
//! The artifact owns graph ports and preprocessing; the provider owns execution.
mod audio;
mod image;
mod manifest;

use super::runtime_identity::{tokenizer_digest, RuntimeIdentity};
use crate::core::{
    artifact_identity::json_digest,
    compilation_cache::CompilationCacheLease,
    execution_contract::ExecutionInput,
    instance_options::{InstanceOptions, Overflow},
    unified_error::{errors, UnifiedResult},
};
use manifest::{Axis, Manifest};
use ort::{
    session::Session,
    tensor::TensorElementType,
    value::{DynValue, Tensor, ValueType},
};
use parking_lot::Mutex;
use serde::Serialize;
use std::{collections::BTreeMap, path::Path};
use tokenizers::Tokenizer;

struct Graph {
    session: Mutex<Session>,
    _cache: Option<CompilationCacheLease>,
    dimension: usize,
}
fn dtype(value: &str) -> Option<TensorElementType> {
    match value {
        "float32" => Some(TensorElementType::Float32),
        "int64" => Some(TensorElementType::Int64),
        _ => None,
    }
}
fn validate_port(
    port: &manifest::Port,
    value: &ValueType,
    fixed: Option<usize>,
) -> anyhow::Result<()> {
    let ValueType::Tensor { ty, shape, .. } = value else {
        anyhow::bail!("graph port is not a tensor")
    };
    anyhow::ensure!(
        Some(*ty) == dtype(&port.dtype) && shape.len() == port.shape.len(),
        "graph port dtype/rank differs from manifest"
    );
    for (actual, expected) in shape.iter().zip(&port.shape) {
        let expected = match expected {
            Axis::Fixed(n) => *n as i64,
            Axis::Symbol(_) => fixed.map_or(-1, |n| n as i64),
        };
        anyhow::ensure!(
            *actual == expected,
            "graph shape differs from manifest: {actual} vs {expected}"
        );
    }
    Ok(())
}
impl Graph {
    fn load(
        root: &Path,
        manifest: &Manifest,
        key: &str,
        options: &InstanceOptions,
        fixed: Option<usize>,
    ) -> anyhow::Result<Self> {
        let graph = &manifest.graphs[key];
        let path = manifest::contained(root, &graph.file)?;
        let concrete: Option<Vec<ExecutionInput>> = graph
            .inputs
            .iter()
            .map(|p| {
                Some(ExecutionInput {
                    name: p.name.clone(),
                    dtype: p.dtype.clone(),
                    shape: p
                        .shape
                        .iter()
                        .map(|a| match a {
                            Axis::Fixed(n) => Some(*n as i64),
                            Axis::Symbol(_) => fixed.map(|n| n as i64),
                        })
                        .collect::<Option<_>>()?,
                })
            })
            .collect();
        let prepared = match concrete {
            Some(inputs) if fixed.is_some() => {
                options.create_session_with_fixed_contract(&path, &inputs)?
            }
            Some(inputs) => options.create_session_with_contract(&path, &inputs)?,
            None => options.prepare_session(&path)?,
        };
        let session = &prepared.session;
        anyhow::ensure!(
            session.inputs.len() == graph.inputs.len() && session.outputs.len() == 1,
            "graph ports differ from manifest"
        );
        for expected in &graph.inputs {
            let actual = session
                .inputs
                .iter()
                .find(|p| p.name == expected.name)
                .ok_or_else(|| anyhow::anyhow!("missing input {}", expected.name))?;
            validate_port(expected, &actual.input_type, fixed)?;
        }
        anyhow::ensure!(
            session.outputs[0].name == graph.output.name,
            "graph output name differs from manifest"
        );
        validate_port(&graph.output, &session.outputs[0].output_type, None)?;
        for artifact in &prepared.artifacts {
            if let Some(location) = artifact.digest.role.strip_prefix("external:") {
                let parent = Path::new(&graph.file).parent().unwrap_or(Path::new(""));
                let name = parent.join(location).to_string_lossy().to_string();
                anyhow::ensure!(
                    manifest.files.get(&name) == Some(&artifact.digest.sha256),
                    "external graph data missing from manifest"
                );
            }
            artifact.verify()?;
        }
        let Axis::Fixed(dimension) = graph.output.shape[1] else {
            anyhow::bail!("dynamic output dimension")
        };
        Ok(Self {
            session: Mutex::new(prepared.session),
            _cache: prepared.cache_lease,
            dimension,
        })
    }
    fn run(&self, inputs: Vec<(&str, DynValue)>) -> anyhow::Result<Vec<f32>> {
        let mut session = self.session.lock();
        let outputs = session.run(inputs)?;
        let output = outputs
            .get("embedding")
            .ok_or_else(|| anyhow::anyhow!("missing embedding output"))?;
        let (shape, values) = output.try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.as_ref() == [1, self.dimension as i64] && values.len() == self.dimension,
            "invalid embedding output shape"
        );
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        anyhow::ensure!(
            values.iter().all(|v| v.is_finite()) && (norm - 1.0).abs() < 0.005,
            "graph must return a finite unit embedding"
        );
        Ok(values.to_vec())
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct AudioCapability {
    pub sample_rates: Vec<usize>,
    pub max_sample_rate: usize,
    pub max_seconds: usize,
    pub max_channels: usize,
    pub layout: &'static str,
}
pub struct OmniModel {
    manifest: Manifest,
    tokenizer: Tokenizer,
    graphs: BTreeMap<String, Graph>,
    audio: audio::AudioConfig,
    limit: usize,
    fixed_text: Option<usize>,
    identity: RuntimeIdentity,
}
impl OmniModel {
    pub fn load(options: &InstanceOptions) -> UnifiedResult<Self> {
        options.validate()?;
        Self::load_impl(options)
            .map_err(|e| errors::model_load(&options.model_path, &e.to_string()))
    }
    fn load_impl(options: &InstanceOptions) -> anyhow::Result<Self> {
        options.validate()?;
        anyhow::ensure!(
            options.model_file.is_none(),
            "Omni graph selection belongs to its manifest"
        );
        anyhow::ensure!(
            options.overflow == Overflow::Reject,
            "Omni requires explicit rejection of overlength inputs"
        );
        let root = Path::new(&options.model_path);
        let (manifest, snapshots) = Manifest::load(root)?;
        let audio: audio::AudioConfig = serde_json::from_slice(&std::fs::read(
            manifest::contained(root, &manifest.processors.audio.file)?,
        )?)?;
        audio.validate()?;
        let mut tokenizer = Tokenizer::from_file(manifest::contained(root, &manifest.tokenizer)?)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        tokenizer.with_padding(None);
        anyhow::ensure!(
            (manifest.processors.text.pad_token_id as usize) < tokenizer.get_vocab_size(true),
            "pad token is outside tokenizer vocabulary"
        );
        let limit = options.effective_limit(manifest.max_text_length)?;
        let fixed = options.execution_max_input_tokens;
        if let Some(fixed) = fixed {
            anyhow::ensure!(
                fixed > 0 && fixed <= limit,
                "execution text length exceeds input budget"
            );
        }
        let limit = fixed.unwrap_or(limit);
        let mut graphs = BTreeMap::new();
        for name in ["text", "image", "clap", "audio"] {
            graphs.insert(
                name.into(),
                Graph::load(
                    root,
                    &manifest,
                    name,
                    options,
                    if name == "text" { fixed } else { None },
                )?,
            );
        }
        for snapshot in &snapshots {
            snapshot.verify()?;
        }
        let identity = RuntimeIdentity {
            version: 1,
            model_type: "vela_omni",
            runtime: format!(
                "ort-omni-v1:{}",
                json_digest(
                    &options
                        .evidence
                        .lock()
                        .iter()
                        .map(|e| (
                            &e.runtime_build,
                            e.provider,
                            e.device_id,
                            e.precision,
                            e.cpu_fallback_disabled,
                            e.custom_ops_profile,
                            &e.custom_ops_sha256,
                            &e.execution_inputs,
                            &e.input_schema,
                            &e.compiler_flags,
                            options.intra_threads
                        ))
                        .collect::<Vec<_>>()
                )?
            ),
            effective_config_sha256: json_digest(&(&manifest, limit, fixed))?,
            tokenizer_sha256: tokenizer_digest(&tokenizer)?,
            artifacts: snapshots.iter().map(|s| s.digest.clone()).collect(),
            layer: 0,
            dimension: manifest.embedding.dimension,
            max_sequence_length: limit,
            pooling_contract: if manifest.variant == "nano" {
                "cls_l2"
            } else {
                "last_token_full_l2_prefix_l2"
            },
        };
        let model = Self {
            manifest,
            tokenizer,
            graphs,
            audio,
            limit,
            fixed_text: fixed,
            identity,
        };
        // Prepare every declared branch, including CLAP. No text-only readiness.
        model.encode_text("warmup", None)?;
        let size = model.manifest.processors.image.size;
        model.graphs["image"].run(vec![(
            "pixel_values",
            Tensor::from_array(([1, 3, size, size], vec![0f32; 3 * size * size]))?.into_dyn(),
        )])?;
        model.audio_forward(&vec![0f32; 16000], 16000, 1)?;
        Ok(model)
    }
    pub fn prepare_text<'a>(&self, text: &'a str) -> &'a str {
        if self.manifest.processors.text.strip_whitespace {
            text.trim()
        } else {
            text
        }
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn max_text_length(&self) -> usize {
        self.manifest.max_text_length
    }
    pub fn effective_limit(&self) -> usize {
        self.limit
    }
    pub fn dimension(&self) -> usize {
        self.manifest.embedding.dimension
    }
    pub fn pooling(&self) -> &str {
        &self.manifest.embedding.text_pooling
    }
    pub fn audio_capability(&self) -> AudioCapability {
        let a = &self.manifest.processors.audio;
        AudioCapability {
            sample_rates: a.sample_rates.clone(),
            max_sample_rate: a.max_sample_rate,
            max_seconds: a.max_seconds,
            max_channels: 8,
            layout: "channels_first",
        }
    }
    fn validate_dimension(&self, dimension: Option<usize>) -> anyhow::Result<()> {
        anyhow::ensure!(
            dimension.is_none_or(|d| self.manifest.embedding.dimensions.contains(&d)),
            "Omni does not support requested output dimension"
        );
        Ok(())
    }
    pub fn descriptor(&self, layer: usize, dimension: usize) -> UnifiedResult<RuntimeIdentity> {
        if layer != 0 {
            return Err(errors::config_error(
                "target_layer",
                "Omni does not support layer early exit",
            ));
        }
        self.validate_dimension((dimension != 0).then_some(dimension))
            .map_err(|e| errors::config_error("target_dimension", &e.to_string()))?;
        Ok(self.identity.clone())
    }
    pub fn encode_text(&self, text: &str, dimension: Option<usize>) -> UnifiedResult<Vec<f32>> {
        let call = || -> anyhow::Result<Vec<f32>> {
            self.validate_dimension(dimension)?;
            let text = if self.manifest.processors.text.strip_whitespace {
                text.trim()
            } else {
                text
            };
            let encoded = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            anyhow::ensure!(
                !encoded.is_empty() && encoded.len() <= self.limit,
                "text exceeds prepared token budget"
            );
            let length = self.fixed_text.unwrap_or(encoded.len());
            let mut ids = vec![i64::from(self.manifest.processors.text.pad_token_id); length];
            let mut mask = vec![0i64; length];
            let start = if self.manifest.processors.text.padding_side == "left" {
                length - encoded.len()
            } else {
                0
            };
            for (i, (id, m)) in encoded
                .get_ids()
                .iter()
                .zip(encoded.get_attention_mask())
                .enumerate()
            {
                ids[start + i] = i64::from(*id);
                mask[start + i] = i64::from(*m);
            }
            self.graphs["text"].run(vec![
                (
                    "input_ids",
                    Tensor::from_array(([1, length], ids))?.into_dyn(),
                ),
                (
                    "attention_mask",
                    Tensor::from_array(([1, length], mask))?.into_dyn(),
                ),
            ])
        };
        call().map_err(|e| errors::inference_error("embedding", &e.to_string()))
    }
    pub fn encode_image(&self, bytes: &[u8], dimension: Option<usize>) -> UnifiedResult<Vec<f32>> {
        let call = || -> anyhow::Result<Vec<f32>> {
            self.validate_dimension(dimension)?;
            let p = &self.manifest.processors.image;
            let values = image::pixels(bytes, p)?;
            self.graphs["image"].run(vec![(
                "pixel_values",
                Tensor::from_array(([1, 3, p.size, p.size], values))?.into_dyn(),
            )])
        };
        call().map_err(|e| errors::inference_error("embedding", &e.to_string()))
    }
    fn audio_forward(&self, pcm: &[f32], rate: usize, channels: usize) -> anyhow::Result<Vec<f32>> {
        audio::validate_pcm(pcm, rate, channels, &self.manifest.processors.audio)?;
        let original16 = audio::native_rate(pcm, rate, channels, 16000);
        let original48 = audio::native_rate(pcm, rate, channels, 48000);
        let windows = audio::windows(original48.len());
        let mut clap = vec![0f32; 512];
        for (start, end) in &windows {
            let features = audio::features(&original48[*start..*end], &self.audio.clap, true);
            let vector = self.graphs["clap"].run(vec![(
                "input_features",
                Tensor::from_array(([1, 1, 1001, 64], features))?.into_dyn(),
            )])?;
            for (value, add) in clap.iter_mut().zip(vector) {
                *value += add;
            }
        }
        if windows.len() > 1 {
            for value in &mut clap {
                *value /= windows.len() as f32;
            }
            audio::normalize(&mut clap)?;
        }
        let whisper = audio::features(&original16, &self.audio.whisper, false);
        self.graphs["audio"].run(vec![
            (
                "input_features",
                Tensor::from_array(([1, 80, 3000], whisper))?.into_dyn(),
            ),
            (
                "clap_embedding",
                Tensor::from_array(([1, 512], clap))?.into_dyn(),
            ),
        ])
    }
    pub fn encode_audio(
        &self,
        pcm: &[f32],
        rate: usize,
        channels: usize,
        dimension: Option<usize>,
    ) -> UnifiedResult<Vec<f32>> {
        let call = || {
            self.validate_dimension(dimension)?;
            self.audio_forward(pcm, rate, channels)
        };
        call().map_err(|e| errors::inference_error("embedding", &e.to_string()))
    }
    pub fn finish_profiling(&self) -> UnifiedResult<Vec<String>> {
        self.graphs
            .values()
            .map(|g| {
                g.session
                    .lock()
                    .end_profiling()
                    .map_err(|e| errors::ort_error(&e.to_string()))
            })
            .collect()
    }
}

#[cfg(test)]
mod dsp_reference_test;
