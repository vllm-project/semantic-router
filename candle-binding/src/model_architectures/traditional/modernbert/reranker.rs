//! A cross-encoder with a distinct trained MLP at each declared layer/width.
//! Pair tokenization belongs to the owned task; scores are raw relevance logits.

use super::{ModernBert, TraditionalModernBertClassifier};
use anyhow::{ensure, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

#[derive(Clone, Copy, Default, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairScorerSelection {
    #[serde(default)]
    pub layer: usize,
    #[serde(default)]
    pub dimension: usize,
}

#[derive(Deserialize)]
struct MatryoshkaConfig {
    layer_indices: Vec<usize>,
    dim_indices: Vec<usize>,
    hidden_size: usize,
    num_layers: usize,
    pooling_strategy: String,
    // Older layouts repeat the encoder's normalization declaration.
    has_final_norm: Option<bool>,
    representation_contract: Option<RepresentationContract>,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RepresentationContract {
    version: usize,
    pooling: String,
    intermediate_normalization: String,
    final_normalization: String,
    head_dtype: String,
}

pub struct MatryoshkaReranker {
    encoder: ModernBert,
    dense: Linear,
    output: Linear,
    device: Device,
    selection: PairScorerSelection,
}

impl MatryoshkaReranker {
    pub fn load(path: &str, device: &Device, mut selection: PairScorerSelection) -> Result<Self> {
        let root = Path::new(path);
        let text = std::fs::read_to_string(root.join("config.json"))?;
        let raw: serde_json::Value = serde_json::from_str(&text)?;
        ensure!(
            raw["architectures"] == serde_json::json!(["ModernBertModel"]),
            "capability: reranking requires a ModernBertModel encoder artifact"
        );
        let config = TraditionalModernBertClassifier::parse_model_config(&text)?;
        let representation: RepresentationContract =
            serde_json::from_value(raw["representation_contract"].clone())?;
        ensure!(
            representation.version == 1
                && representation.pooling == "cls"
                && representation.intermediate_normalization == "final_norm"
                && representation.final_normalization == "final_norm"
                && representation.head_dtype == "float32",
            "capability: unsupported reranker representation contract"
        );
        let layout: MatryoshkaConfig =
            serde_json::from_slice(&std::fs::read(root.join("matryoshka_config.json"))?)?;
        ensure!(
            layout.hidden_size == config.hidden_size
                && layout.num_layers == config.num_hidden_layers
                && layout.pooling_strategy == "cls"
                && layout.has_final_norm != Some(false)
                && layout
                    .representation_contract
                    .as_ref()
                    .is_none_or(|declared| declared == &representation),
            "configuration: reranker head layout disagrees with its encoder"
        );
        ensure!(
            !layout.layer_indices.is_empty()
                && !layout.dim_indices.is_empty()
                && layout
                    .layer_indices
                    .iter()
                    .all(|&n| n > 0 && n <= layout.num_layers)
                && layout
                    .dim_indices
                    .iter()
                    .all(|&n| n >= 2 && n <= layout.hidden_size)
                && layout.layer_indices.iter().collect::<HashSet<_>>().len()
                    == layout.layer_indices.len()
                && layout.dim_indices.iter().collect::<HashSet<_>>().len()
                    == layout.dim_indices.len(),
            "configuration: invalid trained reranker exits"
        );
        if selection.layer == 0 {
            selection.layer = layout.num_layers;
        }
        if selection.dimension == 0 {
            selection.dimension = layout.hidden_size;
        }
        ensure!(
            layout.layer_indices.contains(&selection.layer)
                && layout.dim_indices.contains(&selection.dimension),
            "capability: no trained reranker head for the selected layer and dimension"
        );
        let weights = root.join("model.safetensors");
        let encoder_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
        let encoder = ModernBert::load_backbone(encoder_vb, &config)?;
        let head_bytes = std::fs::read(root.join("classification_heads.safetensors"))?;
        let head_tensors = safetensors::SafeTensors::deserialize(&head_bytes)?;
        let prefix = format!("{}.{}", selection.layer, selection.dimension);
        for suffix in ["0.weight", "0.bias", "3.weight", "3.bias"] {
            ensure!(
                head_tensors.tensor(&format!("{prefix}.{suffix}"))?.dtype()
                    == safetensors::Dtype::F32,
                "configuration: reranker heads must store FP32 tensors"
            );
        }
        let heads =
            VarBuilder::from_buffered_safetensors(head_bytes, DType::F32, device)?.pp(prefix);
        let width = selection.dimension / 2;
        let dense = linear(selection.dimension, width, heads.pp("0"))?;
        let output = linear(width, 1, heads.pp("3"))?;
        Ok(Self {
            encoder,
            dense,
            output,
            device: device.clone(),
            selection,
        })
    }

    pub fn selection(&self) -> PairScorerSelection {
        self.selection
    }

    pub fn score_tokens(&self, ids: &[u32]) -> Result<f32> {
        ensure!(!ids.is_empty(), "configuration: empty pair token sequence");
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let mask = Tensor::ones(input.shape(), DType::U32, &self.device)?;
        let hidden = self
            .encoder
            .forward_to_layer(&input, &mask, self.selection.layer)?;
        let cls = hidden
            .i((.., 0, ..))?
            .narrow(1, 0, self.selection.dimension)?
            .to_dtype(DType::F32)?;
        let score = self
            .output
            .forward(&self.dense.forward(&cls)?.gelu_erf()?)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        ensure!(
            score.is_finite(),
            "result_invalid: reranker returned a nonfinite score"
        );
        Ok(score)
    }
}
