//! Owned query/document cross-encoder execution. The loaded graph declares its
//! trained exit; neither its filename nor a classifier transform defines it.

use crate::core::instance_options::{InstanceOptions, Overflow, Provider};
use crate::core::unified_error::{errors, UnifiedResult};
use crate::model_architectures::modernbert_inputs;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::Path};
use tokenizers::{PostProcessor, Tokenizer};

#[derive(Clone, Copy, Default, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairScorerSelection {
    #[serde(default)]
    pub layer: usize,
    #[serde(default)]
    pub dimension: usize,
}

#[derive(Deserialize)]
struct ModelConfig {
    architectures: Vec<String>,
    hidden_size: usize,
    num_hidden_layers: usize,
    max_position_embeddings: usize,
    pad_token_id: i64,
    representation_contract: RepresentationContract,
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

#[derive(Deserialize)]
struct MatryoshkaConfig {
    layer_indices: Vec<usize>,
    dim_indices: Vec<usize>,
    hidden_size: usize,
    num_layers: usize,
    pooling_strategy: String,
    // The encoder contract is authoritative; reject contradictory old metadata.
    has_final_norm: Option<bool>,
    representation_contract: Option<RepresentationContract>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphContract {
    version: usize,
    layer: usize,
    dimension: usize,
    score_type: String,
}

pub struct PairScorer {
    session: Session,
    cache_lease: Option<crate::core::compilation_cache::CompilationCacheLease>,
    tokenizer: Tokenizer,
    selection: PairScorerSelection,
    capacity: usize,
    execution_length: Option<usize>,
    pad_token_id: i64,
}

impl PairScorer {
    pub fn load(
        options: &InstanceOptions,
        mut selection: PairScorerSelection,
    ) -> UnifiedResult<Self> {
        options.validate()?;
        let invalid = |message: &str| errors::config_error("pair_scorer", message);
        if options.overflow != Overflow::Reject {
            return Err(invalid("pair scoring requires reject overflow"));
        }
        let path = Path::new(&options.model_path);
        let read = |name: &str| std::fs::read(path.join(name)).map_err(|e| invalid(&e.to_string()));
        let config: ModelConfig =
            serde_json::from_slice(&read("config.json")?).map_err(|e| invalid(&e.to_string()))?;
        let representation = &config.representation_contract;
        if config.architectures != ["ModernBertModel"]
            || config.max_position_embeddings == 0
            || representation.version != 1
            || representation.pooling != "cls"
            || representation.intermediate_normalization != "final_norm"
            || representation.final_normalization != "final_norm"
            || representation.head_dtype != "float32"
        {
            return Err(invalid("unsupported reranker representation contract"));
        }
        let layout: MatryoshkaConfig = serde_json::from_slice(&read("matryoshka_config.json")?)
            .map_err(|e| invalid(&e.to_string()))?;
        if layout.hidden_size != config.hidden_size
            || layout.num_layers != config.num_hidden_layers
            || layout.pooling_strategy != "cls"
            || layout.has_final_norm == Some(false)
            || layout
                .representation_contract
                .as_ref()
                .is_some_and(|declared| declared != representation)
            || layout.layer_indices.is_empty()
            || layout.dim_indices.is_empty()
            || !layout
                .layer_indices
                .iter()
                .all(|&n| n > 0 && n <= layout.num_layers)
            || !layout
                .dim_indices
                .iter()
                .all(|&n| n >= 2 && n <= layout.hidden_size)
            || layout.layer_indices.iter().collect::<HashSet<_>>().len()
                != layout.layer_indices.len()
            || layout.dim_indices.iter().collect::<HashSet<_>>().len() != layout.dim_indices.len()
        {
            return Err(invalid("invalid trained reranker exit layout"));
        }
        if selection.layer == 0 {
            selection.layer = layout.num_layers;
        }
        if selection.dimension == 0 {
            selection.dimension = layout.hidden_size;
        }
        if !layout.layer_indices.contains(&selection.layer)
            || !layout.dim_indices.contains(&selection.dimension)
        {
            return Err(invalid(
                "no trained reranker head for the selected layer and dimension",
            ));
        }
        let limit = options.effective_limit(config.max_position_embeddings)?;
        let mut tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| invalid(&e.to_string()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| invalid(&e.to_string()))?;
        tokenizer.with_padding(None);
        if limit
            < tokenizer
                .get_post_processor()
                .map_or(0, |p| p.added_tokens(true))
        {
            return Err(invalid(
                "pair budget is smaller than its special-token template",
            ));
        }
        let mut candidates = Vec::new();
        if selection.layer == layout.num_layers && selection.dimension == layout.hidden_size {
            candidates.push(path.join("onnx/model.onnx"));
        }
        candidates.extend([
            path.join(format!(
                "onnx/model_layer_{}_dim_{}.onnx",
                selection.layer, selection.dimension
            )),
            path.join(format!(
                "onnx/layer-{}/dim-{}/model.onnx",
                selection.layer, selection.dimension
            )),
        ]);
        let graph = options.select_graph(candidates)?;
        let execution_limit = options.execution_limit(config.max_position_embeddings)?;
        let prepared = modernbert_inputs::prepare_session(options, &graph, execution_limit)?;
        let session = prepared.session;
        let metadata = session.metadata().map_err(|e| invalid(&e.to_string()))?;
        let declared = metadata
            .custom("semantic_router.pair_scorer")
            .map_err(|e| invalid(&e.to_string()))?
            .ok_or_else(|| invalid("graph has no pair scorer execution contract"))?;
        let contract: GraphContract =
            serde_json::from_str(&declared).map_err(|e| invalid(&e.to_string()))?;
        if contract.version != 1
            || contract.layer != selection.layer
            || contract.dimension != selection.dimension
            || contract.score_type != "relevance_logit"
        {
            return Err(invalid(
                "loaded graph does not implement the selected pair scorer contract",
            ));
        }
        drop(metadata);
        modernbert_inputs::validate(&session.inputs)?;
        if session.outputs.len() != 1 || session.outputs[0].name != "logits" {
            return Err(invalid("reranker requires one logits output"));
        }
        Ok(Self {
            session,
            cache_lease: prepared.cache_lease,
            tokenizer,
            selection,
            capacity: config.max_position_embeddings,
            execution_length: (options.provider == Provider::Migraphx).then_some(execution_limit),
            pad_token_id: config.pad_token_id,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn selection(&self) -> PairScorerSelection {
        self.selection
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn finish_profiling(&mut self) -> UnifiedResult<String> {
        self.session
            .end_profiling()
            .map_err(|e| errors::ort_error(&e.to_string()))
    }

    pub fn score_tokens(&mut self, ids: &[u32]) -> UnifiedResult<f32> {
        let error = |e: ort::Error| errors::inference_error("pair_scores", &e.to_string());
        let length = self.execution_length.unwrap_or(ids.len());
        if ids.is_empty() || length < ids.len() {
            return Err(errors::config_error(
                "pair_scorer",
                "invalid pair execution length",
            ));
        }
        let mut input = vec![self.pad_token_id; length];
        let mut mask = vec![0_i64; length];
        for (i, &id) in ids.iter().enumerate() {
            input[i] = id.into();
            mask[i] = 1;
        }
        let outputs = modernbert_inputs::run_cached(
            &mut self.session,
            &mut self.cache_lease,
            input,
            mask,
            1,
            length,
        )?;
        // The trained head computes FP32 logits even with a lower precision encoder.
        let (shape, values) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(error)?;
        if shape.as_ref() != [1, 1] || values.len() != 1 || !values[0].is_finite() {
            return Err(errors::inference_error(
                "pair_scores",
                "expected one finite FP32 relevance logit",
            ));
        }
        Ok(values[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_graph_layout_preserves_scores_and_requires_the_selected_head_contract() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("instance/testdata/pair_scorer");
        let directory = tempfile::tempdir().unwrap();
        for name in ["config.json", "matryoshka_config.json", "tokenizer.json"] {
            std::fs::copy(source.join(name), directory.path().join(name)).unwrap();
        }
        std::fs::create_dir(directory.path().join("onnx")).unwrap();
        let graph = directory.path().join("onnx/model_layer_2_dim_4.onnx");
        std::fs::copy(source.join("model.onnx"), &graph).unwrap();
        let options = InstanceOptions {
            model_path: directory.path().display().to_string(),
            ..Default::default()
        };
        let mut flat = PairScorer::load(&options, PairScorerSelection::default()).unwrap();
        let mut original = PairScorer::load(
            &InstanceOptions {
                model_path: source.display().to_string(),
                model_file: Some("model.onnx".into()),
                ..Default::default()
            },
            PairScorerSelection::default(),
        )
        .unwrap();
        assert_eq!(
            flat.score_tokens(&[2, 3, 4]).unwrap(),
            original.score_tokens(&[2, 3, 4]).unwrap()
        );
        std::fs::copy(
            &graph,
            directory.path().join("onnx/model_layer_1_dim_2.onnx"),
        )
        .unwrap();
        assert!(PairScorer::load(
            &options,
            PairScorerSelection {
                layer: 1,
                dimension: 2
            }
        )
        .is_err());
        std::fs::rename(&graph, directory.path().join("onnx/model.onnx")).unwrap();
        let mut primary = PairScorer::load(&options, PairScorerSelection::default()).unwrap();
        assert_eq!(
            primary.score_tokens(&[2, 3, 4]).unwrap(),
            flat.score_tokens(&[2, 3, 4]).unwrap()
        );
        // A primary graph is never a fallback for an unrepresented sub-exit.
        assert!(PairScorer::load(
            &options,
            PairScorerSelection {
                layer: 1,
                dimension: 4
            }
        )
        .is_err());
    }
}
