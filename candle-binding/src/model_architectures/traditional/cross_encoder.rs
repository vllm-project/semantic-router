//! BERT Cross-Encoder Reranker
//!
//! A cross-encoder jointly encodes a `(query, document)` pair and produces a
//! single relevance score. Unlike a bi-encoder (which embeds query and document
//! separately and compares with cosine similarity), the cross-encoder lets the
//! query attend to the document through full self-attention, giving higher
//! ranking precision. This is the standard second-stage reranker used in RAG.
//!
//! Architecturally this is a `BertForSequenceClassification` model
//! (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`): BERT encoder + pooler
//! (`bert.pooler.dense` + tanh) + a linear classification head. For a single
//! output label the head emits one logit that is monotonic in relevance; for
//! two labels we take the positive class. We map the result through a sigmoid /
//! softmax so the score lands in `[0, 1]`, matching the Cohere/vLLM rerank API.

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;
use tokenizers::{Tokenizer, TruncationParams, TruncationStrategy};

/// Maximum number of (query, document) pairs scored in a single batched forward.
/// Larger document sets are processed in chunks of this size to bound memory.
const MAX_RERANK_BATCH: usize = 32;

/// A BERT-based cross-encoder reranker.
///
/// SUPPORTED CHECKPOINTS (narrow by design): only BERT-family
/// `BertForSequenceClassification` rerankers with `bert.*` weight names, a
/// `bert.pooler.dense` layer, a `classifier` head, and a WordPiece `tokenizer.json`
/// (e.g. the `cross-encoder/ms-marco-*` family). Non-BERT rerankers such as
/// XLM-RoBERTa (`BAAI/bge-reranker-*`) or ModernBERT rerankers are NOT supported
/// yet; they need a separate architecture-dispatch loading path.
pub struct BertCrossEncoder {
    bert: BertModel,
    pooler: Linear,
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
    num_labels: usize,
    pad_token_id: u32,
}

impl BertCrossEncoder {
    /// Load a cross-encoder reranker from a local directory or HuggingFace Hub id.
    pub fn new(model_id: &str, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        println!("Initializing BERT cross-encoder reranker: {}", model_id);

        let (config_filename, tokenizer_filename, weights_filename, use_pth) =
            Self::resolve_model_files(model_id)?;

        let config_str = std::fs::read_to_string(&config_filename)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let num_labels = Self::detect_num_labels(&config_str);
        let pad_token_id = Self::detect_pad_token_id(&config_str);
        let max_length = Self::detect_max_position_embeddings(&config_str);

        let mut tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(E::msg)?;
        // We pad manually per batch (with a zero attention mask), so disable the
        // tokenizer's own padding to keep encodings minimal and predictable.
        let _ = tokenizer.with_padding(None);
        // Truncate (query, document) pairs to the model's position limit so long
        // RAG inputs can't exceed `max_position_embeddings` and fail at inference.
        // LongestFirst trims whichever side of the pair is longer, preserving the
        // query when the document is the long one (the common RAG case).
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                ..Default::default()
            }))
            .map_err(E::msg)?;

        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&weights_filename),
                    DType::F32,
                    &device,
                )?
            }
        };

        let bert = BertModel::load(vb.pp("bert"), &config)?;

        // BERT pooler: dense(hidden, hidden) followed by tanh (applied at forward time).
        // candle's Linear computes `x @ w.t()`, so HF `[out, in]` weights are passed
        // as-is (NO manual transpose).
        let pooler = {
            let w = vb.get(
                (config.hidden_size, config.hidden_size),
                "bert.pooler.dense.weight",
            )?;
            let b = vb.get(config.hidden_size, "bert.pooler.dense.bias")?;
            Linear::new(w, Some(b))
        };

        // Classification head: (num_labels, hidden).
        let classifier = {
            let w = vb.get((num_labels, config.hidden_size), "classifier.weight")?;
            let b = vb.get(num_labels, "classifier.bias")?;
            Linear::new(w, Some(b))
        };

        println!(
            "  Cross-encoder loaded (hidden_size={}, num_labels={})",
            config.hidden_size, num_labels
        );

        Ok(Self {
            bert,
            pooler,
            classifier,
            tokenizer,
            device,
            num_labels,
            pad_token_id,
        })
    }

    /// Convert a classifier logit row into a [0, 1] relevance score.
    fn score_from_logits(&self, logits: &[f32]) -> f32 {
        if self.num_labels <= 1 {
            sigmoid(logits[0])
        } else {
            // Positive (relevant) class is the last label; softmax for a [0,1] score.
            let max = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps[exps.len() - 1] / sum
        }
    }

    /// Score a batch of (query, document) pairs in a single padded forward pass.
    /// Right-pads each pair to the batch's max length with `pad_token_id` and a
    /// zero attention mask, so padded positions don't affect the real tokens'
    /// representations, making batched scores identical to per-pair scoring.
    fn score_batch(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        let batch = documents.len();
        if batch == 0 {
            return Ok(Vec::new());
        }

        let mut encodings = Vec::with_capacity(batch);
        let mut max_len = 0usize;
        for doc in documents {
            let enc = self.tokenizer.encode((query, *doc), true).map_err(E::msg)?;
            max_len = max_len.max(enc.get_ids().len());
            encodings.push(enc);
        }

        let mut ids = Vec::with_capacity(batch * max_len);
        let mut type_ids = Vec::with_capacity(batch * max_len);
        let mut mask = Vec::with_capacity(batch * max_len);
        for enc in &encodings {
            let e_ids = enc.get_ids();
            let e_types = enc.get_type_ids();
            let e_mask = enc.get_attention_mask();
            ids.extend_from_slice(e_ids);
            type_ids.extend_from_slice(e_types);
            mask.extend_from_slice(e_mask);
            for _ in e_ids.len()..max_len {
                ids.push(self.pad_token_id);
                type_ids.push(0);
                mask.push(0);
            }
        }

        let input_ids = Tensor::from_vec(ids, (batch, max_len), &self.device)?;
        let token_type_ids = Tensor::from_vec(type_ids, (batch, max_len), &self.device)?;
        let attention_mask = Tensor::from_vec(mask, (batch, max_len), &self.device)?;

        // BERT forward -> (batch, max_len, hidden)
        let sequence_output =
            self.bert
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let cls = sequence_output.i((.., 0))?; // (batch, hidden)
        let pooled = self.pooler.forward(&cls)?.tanh()?; // (batch, hidden)
        let logits = self.classifier.forward(&pooled)?; // (batch, num_labels)
        let logits_rows = logits.to_vec2::<f32>()?;

        Ok(logits_rows
            .iter()
            .map(|row| self.score_from_logits(row))
            .collect())
    }

    /// Score a single (query, document) pair, returning a relevance score in [0, 1].
    pub fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        let encoding = self
            .tokenizer
            .encode((query, document), true)
            .map_err(E::msg)?;

        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let type_ids: Vec<u32> = encoding.get_type_ids().to_vec();
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let seq_len = ids.len();

        let input_ids = Tensor::from_vec(ids, (1, seq_len), &self.device)?;
        let token_type_ids = Tensor::from_vec(type_ids, (1, seq_len), &self.device)?;
        let attention_mask = Tensor::from_vec(mask, (1, seq_len), &self.device)?;

        // BERT forward -> (1, seq_len, hidden)
        let sequence_output =
            self.bert
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Pool the [CLS] token, then dense + tanh (the BERT pooler).
        let cls = sequence_output.i((.., 0))?; // (1, hidden)
        let pooled = self.pooler.forward(&cls)?.tanh()?; // (1, hidden)

        // Classification head -> (1, num_labels)
        let logits = self.classifier.forward(&pooled)?;
        let logits_vec = logits.squeeze(0)?.to_vec1::<f32>()?;

        Ok(self.score_from_logits(&logits_vec))
    }

    /// Rerank `documents` against `query`. Returns `(original_index, score)`
    /// pairs sorted by score descending. `top_n <= 0` returns all.
    ///
    /// Documents are scored in batched forward passes of up to
    /// `MAX_RERANK_BATCH` pairs to amortize overhead on large candidate sets.
    pub fn rerank(&self, query: &str, documents: &[&str], top_n: i32) -> Result<Vec<(usize, f32)>> {
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(documents.len());
        for (chunk_idx, chunk) in documents.chunks(MAX_RERANK_BATCH).enumerate() {
            let base = chunk_idx * MAX_RERANK_BATCH;
            let scores = self.score_batch(query, chunk)?;
            for (offset, score) in scores.into_iter().enumerate() {
                scored.push((base + offset, score));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if top_n > 0 && (top_n as usize) < scored.len() {
            scored.truncate(top_n as usize);
        }
        Ok(scored)
    }

    /// Detect the maximum sequence length from config.json
    /// (`max_position_embeddings`), defaulting to 512 (standard BERT).
    fn detect_max_position_embeddings(config_str: &str) -> usize {
        serde_json::from_str::<serde_json::Value>(config_str)
            .ok()
            .and_then(|j| j.get("max_position_embeddings").and_then(|v| v.as_u64()))
            .map(|v| v as usize)
            .filter(|&v| v > 0)
            .unwrap_or(512)
    }

    /// Detect the padding token id from config.json (`pad_token_id`),
    /// defaulting to 0 (BERT's `[PAD]`).
    fn detect_pad_token_id(config_str: &str) -> u32 {
        serde_json::from_str::<serde_json::Value>(config_str)
            .ok()
            .and_then(|j| j.get("pad_token_id").and_then(|v| v.as_u64()))
            .map(|v| v as u32)
            .unwrap_or(0)
    }

    /// Detect the number of output labels from config.json (`id2label` or
    /// `num_labels`), defaulting to 1 (single-logit reranker).
    fn detect_num_labels(config_str: &str) -> usize {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(config_str) {
            if let Some(id2label) = json.get("id2label").and_then(|v| v.as_object()) {
                if !id2label.is_empty() {
                    return id2label.len();
                }
            }
            if let Some(n) = json.get("num_labels").and_then(|v| v.as_u64()) {
                if n > 0 {
                    return n as usize;
                }
            }
        }
        1
    }

    /// Resolve config/tokenizer/weights files from a local dir or the HF Hub.
    fn resolve_model_files(model_id: &str) -> Result<(String, String, String, bool)> {
        if Path::new(model_id).exists() {
            let config_path = Path::new(model_id).join("config.json");
            let tokenizer_path = Path::new(model_id).join("tokenizer.json");
            let (weights_path, use_pth) = if Path::new(model_id).join("model.safetensors").exists()
            {
                (Path::new(model_id).join("model.safetensors"), false)
            } else if Path::new(model_id).join("pytorch_model.bin").exists() {
                (Path::new(model_id).join("pytorch_model.bin"), true)
            } else {
                return Err(E::msg(format!("No model weights found in {}", model_id)));
            };
            Ok((
                config_path.to_string_lossy().to_string(),
                tokenizer_path.to_string_lossy().to_string(),
                weights_path.to_string_lossy().to_string(),
                use_pth,
            ))
        } else {
            use hf_hub::{api::sync::Api, Repo, RepoType};
            let repo =
                Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let (weights, use_pth) = match api.get("model.safetensors") {
                Ok(w) => (w, false),
                Err(_) => (api.get("pytorch_model.bin")?, true),
            };
            Ok((
                config.to_string_lossy().to_string(),
                tokenizer.to_string_lossy().to_string(),
                weights.to_string_lossy().to_string(),
                use_pth,
            ))
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl std::fmt::Debug for BertCrossEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BertCrossEncoder")
            .field("device", &self.device)
            .field("num_labels", &self.num_labels)
            .finish()
    }
}

/// Global cross-encoder instance, initialized once via the FFI layer.
pub static CROSS_ENCODER: std::sync::OnceLock<std::sync::Arc<BertCrossEncoder>> =
    std::sync::OnceLock::new();
