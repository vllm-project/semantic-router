//! Owned task instances. A handle owns a reference, never a process-global role.

use crate::{
    core::{
        instance_options::{InstanceOptions, Overflow, SessionEvidence},
        unified_error::{errors, UnifiedResult},
    },
    MmBertEmbeddingModel, MmBertSequenceClassifier, MmBertTokenClassifier,
    MultiModalEmbeddingModel,
};
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
};
use tokenizers::Tokenizer;

enum Model {
    Sequence(MmBertSequenceClassifier),
    Token(MmBertTokenClassifier),
    Embedding(MmBertEmbeddingModel),
    MultiModal(MultiModalEmbeddingModel),
}

struct Instance {
    model: Mutex<Model>,
    tokenizer: Tokenizer,
    options: InstanceOptions,
    task: &'static str,
    model_limit: usize,
    task_limit: usize,
    effective_limit: usize,
    labels: Vec<String>,
    dimension: usize,
    completed: AtomicU64,
}

static INSTANCES: OnceLock<RwLock<HashMap<u64, Arc<Instance>>>> = OnceLock::new();
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn instances() -> &'static RwLock<HashMap<u64, Arc<Instance>>> {
    INSTANCES.get_or_init(Default::default)
}

fn get(handle: u64) -> UnifiedResult<Arc<Instance>> {
    instances()
        .read()
        .get(&handle)
        .cloned()
        .ok_or_else(|| errors::config_error("handle", "instance is closed or unknown"))
}

fn insert(instance: Arc<Instance>) -> u64 {
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    assert_ne!(handle, 0, "instance handle space exhausted");
    instances().write().insert(handle, instance);
    handle
}

/// Clone ownership of the same physical sessions; model, graph and head stay fixed.
pub fn clone_handle(handle: u64) -> UnifiedResult<u64> {
    Ok(insert(get(handle)?))
}

/// Remove ownership immediately. A native call already holding an Arc finishes
/// before the final model is destroyed, even if its caller has been cancelled.
pub fn close(handle: u64) {
    let removed = instances().write().remove(&handle);
    // Native teardown may be slow; never hold the process-wide registry lock
    // while destructing a session owned by the final reference.
    drop(removed);
}

fn fresh_options(mut options: InstanceOptions) -> InstanceOptions {
    options.evidence = Arc::default();
    options
}

pub fn load_sequence(options: InstanceOptions) -> UnifiedResult<u64> {
    let options = fresh_options(options);
    let model = MmBertSequenceClassifier::load_with_options(&options)?;
    prepare(Model::Sequence(model), options)
}
pub fn load_token(options: InstanceOptions) -> UnifiedResult<u64> {
    let options = fresh_options(options);
    let model = MmBertTokenClassifier::load_with_options(&options)?;
    prepare(Model::Token(model), options)
}
pub fn load_embedding(options: InstanceOptions) -> UnifiedResult<u64> {
    let options = fresh_options(options);
    let model = MmBertEmbeddingModel::load_with_options(&options)?;
    prepare(Model::Embedding(model), options)
}
pub fn load_multimodal(options: InstanceOptions) -> UnifiedResult<u64> {
    let options = fresh_options(options);
    let model = MultiModalEmbeddingModel::load_with_options(&options)?;
    prepare(Model::MultiModal(model), options)
}

fn prepare(model: Model, options: InstanceOptions) -> UnifiedResult<u64> {
    let (tokenizer, task, model_limit, task_limit, labels, dimension) = match &model {
        Model::Sequence(m) => (
            m.tokenizer(),
            "sequence_classification",
            m.config().max_position_embeddings,
            512.min(m.config().max_position_embeddings),
            (0..m.config().num_labels)
                .map(|n| m.config().get_label(n as i32))
                .collect(),
            0,
        ),
        Model::Token(m) => (
            m.tokenizer(),
            "token_classification",
            m.config().max_position_embeddings,
            512.min(m.config().max_position_embeddings),
            (0..m.config().num_labels)
                .map(|n| m.config().get_label(n as i32))
                .collect(),
            0,
        ),
        Model::Embedding(m) => (
            m.tokenizer(),
            "embedding",
            m.config().max_position_embeddings,
            m.config().max_position_embeddings,
            vec![],
            m.config().hidden_size,
        ),
        Model::MultiModal(m) => (
            m.tokenizer(),
            "multimodal_embedding",
            m.config().max_seq_len,
            m.config().max_seq_len,
            vec![],
            m.config().embedding_dim,
        ),
    };
    let mut tokenizer = tokenizer.clone();
    tokenizer
        .with_truncation(None)
        .map_err(|e| errors::tokenization_error(&e.to_string()))?;
    tokenizer.with_padding(None);
    let effective_limit = options.effective_limit(task_limit)?;
    Ok(insert(Arc::new(Instance {
        model: Mutex::new(model),
        tokenizer,
        options,
        task,
        model_limit,
        task_limit,
        effective_limit,
        labels,
        dimension,
        completed: AtomicU64::new(0),
    })))
}

#[derive(Debug, Serialize)]
pub struct InputUsage {
    pub original_tokens: usize,
    pub processed_tokens: usize,
    pub truncated: bool,
}

impl Instance {
    fn input(&self, text: &str) -> UnifiedResult<InputUsage> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        let original_tokens = encoding.len();
        if original_tokens > self.effective_limit && self.options.overflow == Overflow::Reject {
            return Err(errors::validation(
                "input_tokens",
                &format!("at most {}", self.effective_limit),
                &original_tokens.to_string(),
            ));
        }
        Ok(InputUsage {
            original_tokens,
            processed_tokens: original_tokens.min(self.effective_limit),
            truncated: original_tokens > self.effective_limit,
        })
    }
    fn dimension(&self, target: Option<usize>) -> UnifiedResult<()> {
        if let Some(target) = target {
            if target == 0 || target > self.dimension {
                return Err(errors::validation(
                    "target_dimension",
                    &format!("1..={}", self.dimension),
                    &target.to_string(),
                ));
            }
        }
        Ok(())
    }
    fn wrong_task(&self, task: &str) -> crate::UnifiedError {
        errors::validation("task", task, self.task)
    }
    fn completed(&self) {
        self.completed.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Serialize)]
pub struct InstanceInfo {
    pub task: &'static str,
    pub model_limit: usize,
    pub task_limit: usize,
    pub effective_limit: usize,
    pub overflow: Overflow,
    pub labels: Vec<String>,
    pub dimension: usize,
    pub sessions: Vec<SessionEvidence>,
    /// Counts successfully completed real native inference calls, not loads.
    pub completed_inferences: u64,
}

pub fn info(handle: u64) -> UnifiedResult<InstanceInfo> {
    let instance = get(handle)?;
    let sessions = instance.options.evidence.lock().clone();
    Ok(InstanceInfo {
        task: instance.task,
        model_limit: instance.model_limit,
        task_limit: instance.task_limit,
        effective_limit: instance.effective_limit,
        overflow: instance.options.overflow,
        labels: instance.labels.clone(),
        dimension: instance.dimension,
        sessions,
        completed_inferences: instance.completed.load(Ordering::Relaxed),
    })
}

#[derive(Debug, Serialize)]
pub struct Distribution {
    pub label: String,
    pub class_id: i32,
    pub confidence: f32,
    pub labels: Vec<String>,
    pub probabilities: Vec<f32>,
    pub input: InputUsage,
}

pub fn classify(handle: u64, text: &str) -> UnifiedResult<Distribution> {
    let instance = get(handle)?;
    let input = instance.input(text)?;
    let mut model = instance.model.lock();
    let Model::Sequence(model) = &mut *model else {
        return Err(instance.wrong_task("sequence_classification"));
    };
    let result = model.classify(text)?;
    if result.probabilities.len() != instance.labels.len()
        || result
            .probabilities
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err(errors::inference_error(
            "distribution",
            "model returned invalid label probabilities",
        ));
    }
    instance.completed();
    Ok(Distribution {
        label: result.label,
        class_id: result.class_id,
        confidence: result.confidence,
        labels: instance.labels.clone(),
        probabilities: result.probabilities,
        input,
    })
}

#[derive(Debug, Serialize)]
pub struct Span {
    pub text: String,
    pub entity_type: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}
#[derive(Debug, Serialize)]
pub struct TokenSpans {
    pub spans: Vec<Span>,
    pub offset_unit: &'static str,
    pub input: InputUsage,
}

pub fn detect_tokens(handle: u64, text: &str) -> UnifiedResult<TokenSpans> {
    let instance = get(handle)?;
    let input = instance.input(text)?;
    let mut model = instance.model.lock();
    let Model::Token(model) = &mut *model else {
        return Err(instance.wrong_task("token_classification"));
    };
    let result = model.detect_entities(text)?;
    let mut spans = Vec::with_capacity(result.entities.len());
    for entity in result.entities {
        if text.get(entity.start..entity.end) != Some(entity.text.as_str())
            || !entity.confidence.is_finite()
        {
            return Err(errors::inference_error(
                "token_spans",
                "model returned invalid UTF-8 byte offsets or confidence",
            ));
        }
        spans.push(Span {
            text: entity.text,
            entity_type: entity.entity_type,
            start: entity.start,
            end: entity.end,
            confidence: entity.confidence,
        });
    }
    instance.completed();
    Ok(TokenSpans {
        spans,
        offset_unit: "utf8_bytes",
        input,
    })
}

#[derive(Debug, Serialize)]
pub struct Embedding {
    pub values: Vec<f32>,
    pub normalized: bool,
    pub modality: &'static str,
    pub input: Option<InputUsage>,
    pub truncated_frames: bool,
}

fn embedding_result(
    instance: &Instance,
    values: Vec<f32>,
    modality: &'static str,
    input: Option<InputUsage>,
    truncated_frames: bool,
) -> UnifiedResult<Embedding> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err(errors::inference_error(
            "embedding",
            "model returned an empty or nonfinite embedding",
        ));
    }
    instance.completed();
    Ok(Embedding {
        values,
        normalized: true,
        modality,
        input,
        truncated_frames,
    })
}

pub fn encode_text(
    handle: u64,
    text: &str,
    layer: Option<usize>,
    dimension: Option<usize>,
) -> UnifiedResult<Embedding> {
    let instance = get(handle)?;
    instance.dimension(dimension)?;
    let input = instance.input(text)?;
    let mut model = instance.model.lock();
    let values = match &mut *model {
        Model::Embedding(model) => {
            if layer.is_some_and(|layer| {
                layer != model.num_layers() && !model.available_exit_layers().contains(&layer)
            }) {
                return Err(errors::config_error(
                    "target_layer",
                    "requested layer has no loaded ONNX session",
                ));
            }
            model.encode_single(text, layer, dimension)?.to_vec()
        }
        Model::MultiModal(model) if layer.is_none() => model.encode_text(text, dimension)?.to_vec(),
        _ => {
            return Err(instance.wrong_task("embedding or multimodal_embedding without layer exit"))
        }
    };
    embedding_result(&instance, values, "text", Some(input), false)
}

pub fn encode_image(
    handle: u64,
    pixels: &[f32],
    height: usize,
    width: usize,
    dimension: Option<usize>,
) -> UnifiedResult<Embedding> {
    let instance = get(handle)?;
    instance.dimension(dimension)?;
    let model = instance.model.lock();
    let Model::MultiModal(model) = &*model else {
        return Err(instance.wrong_task("multimodal_embedding"));
    };
    if height != model.config().image_size || width != model.config().image_size {
        return Err(errors::config_error(
            "image_size",
            "pixels must use the model's configured image size",
        ));
    }
    if pixels
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err(errors::config_error(
            "pixels",
            "CHW pixels must be finite values in [0,1]",
        ));
    }
    embedding_result(
        &instance,
        model
            .encode_image(pixels, height, width, dimension)?
            .to_vec(),
        "image",
        None,
        false,
    )
}

pub fn encode_image_bytes(
    handle: u64,
    bytes: &[u8],
    dimension: Option<usize>,
) -> UnifiedResult<Embedding> {
    let instance = get(handle)?;
    let size = {
        let model = instance.model.lock();
        let Model::MultiModal(model) = &*model else {
            return Err(instance.wrong_task("multimodal_embedding"));
        };
        model.config().image_size
    };
    let size_u32 =
        u32::try_from(size).map_err(|_| errors::config_error("image_size", "exceeds u32"))?;
    let pixels = crate::ffi::multimodal::decode_resize_to_chw_f32(bytes, size_u32, size_u32)
        .map_err(|e| errors::inference_error("image_decode", &e))?;
    // Keep our Arc alive across decoding and the nested native call. Close may
    // remove the handle concurrently; then the nested lookup safely rejects it.
    encode_image(handle, &pixels, size, size, dimension)
}

pub fn encode_audio(
    handle: u64,
    mel: &[f32],
    n_mels: usize,
    frames: usize,
    dimension: Option<usize>,
) -> UnifiedResult<Embedding> {
    let instance = get(handle)?;
    instance.dimension(dimension)?;
    let model = instance.model.lock();
    let Model::MultiModal(model) = &*model else {
        return Err(instance.wrong_task("multimodal_embedding"));
    };
    if n_mels != model.config().n_mels
        || frames == 0
        || n_mels.checked_mul(frames) != Some(mel.len())
        || mel.iter().any(|v| !v.is_finite())
    {
        return Err(errors::config_error(
            "mel_spectrogram",
            "invalid shape, mel count or nonfinite values",
        ));
    }
    if frames > 3000 && instance.options.overflow == Overflow::Reject {
        return Err(errors::validation(
            "audio_frames",
            "at most 3000",
            &frames.to_string(),
        ));
    }
    embedding_result(
        &instance,
        model.encode_audio(mel, n_mels, frames, dimension)?.to_vec(),
        "audio",
        None,
        frames > 3000,
    )
}

pub fn finish_profiling(handle: u64) -> UnifiedResult<Vec<String>> {
    let instance = get(handle)?;
    if instance.options.profile_prefix.is_none() {
        return Ok(vec![]);
    }
    let mut model = instance.model.lock();
    match &mut *model {
        Model::Sequence(m) => m.finish_profiling(),
        Model::Token(m) => m.finish_profiling(),
        Model::Embedding(m) => m.finish_profiling(),
        Model::MultiModal(m) => m.finish_profiling(),
    }
}

#[derive(Debug, Serialize)]
pub struct TextWindow {
    pub start: usize,
    pub end: usize,
}

/// Split text using this owned model's untruncated tokenizer. A window's budget
/// includes the model's actual post-processing special tokens. This does not
/// run inference or increment completed_inferences.
pub fn text_windows(handle: u64, text: &str, max_tokens: usize) -> UnifiedResult<Vec<TextWindow>> {
    let instance = get(handle)?;
    if !matches!(instance.task, "embedding" | "multimodal_embedding") {
        return Err(instance.wrong_task("embedding"));
    }
    let limit = if max_tokens == 0 {
        instance.effective_limit
    } else {
        max_tokens.min(instance.effective_limit)
    };
    tokenizer_windows(&instance.tokenizer, text, limit)
}

fn tokenizer_windows(
    tokenizer: &Tokenizer,
    text: &str,
    limit: usize,
) -> UnifiedResult<Vec<TextWindow>> {
    let count = |text: &str| {
        tokenizer
            .encode(text, true)
            .map(|encoding| encoding.len())
            .map_err(|e| errors::tokenization_error(&e.to_string()))
    };
    let specials = count("")?;
    if limit <= specials {
        return Err(errors::validation(
            "input_tokens",
            "window capacity greater than reserved special tokens",
            &limit.to_string(),
        ));
    }
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|e| errors::tokenization_error(&e.to_string()))?;
    let offsets: Vec<_> = encoded
        .get_offsets()
        .iter()
        .copied()
        .filter(|(start, end)| end > start)
        .collect();
    let mut windows = Vec::new();
    let mut first = 0;
    while first < offsets.len() {
        let start = offsets[first].0;
        let mut last = (first + limit - specials).min(offsets.len());
        let end = loop {
            if last == first {
                return Err(errors::validation(
                    "input_tokens",
                    "enough capacity for one UTF-8 token span",
                    &limit.to_string(),
                ));
            }
            let end = offsets[last - 1].1;
            if let Some(window) = text.get(start..end) {
                // A substring can tokenize differently at its new boundary.
                // Verify the actual prepared tokenizer count before exposing it.
                if count(window)? <= limit {
                    break end;
                }
            }
            last -= 1;
        };
        windows.push(TextWindow { start, end });
        first = last;
        // Byte-level pieces can describe the same Unicode scalar. The complete
        // scalar has already been included and counted in the preceding window.
        while first < offsets.len() && offsets[first].0 < end {
            first += 1;
        }
    }
    Ok(windows)
}

#[cfg(test)]
mod window_tests {
    use super::tokenizer_windows;
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
        processors::template::TemplateProcessing, Tokenizer,
    };

    #[test]
    fn windows_reserve_special_tokens_and_keep_utf8_boundaries() {
        let vocab = ["[UNK]", "[CLS]", "[SEP]", "hello", "world", "秘密"]
            .into_iter()
            .enumerate()
            .map(|(i, value)| (value.to_owned(), i as u32))
            .collect();
        let mut tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        );
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")
                .unwrap()
                .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                .build()
                .unwrap(),
        ));
        let text = "hello 秘密 world hello 秘密";
        let windows = tokenizer_windows(&tokenizer, text, 4).unwrap();
        assert_eq!(windows.len(), 3);
        let pieces: Vec<_> = windows
            .iter()
            .map(|window| &text[window.start..window.end])
            .collect();
        assert_eq!(pieces, ["hello 秘密", "world hello", "秘密"]);
        for piece in pieces {
            assert!(tokenizer.encode(piece, true).unwrap().len() <= 4);
        }
        assert!(tokenizer_windows(&tokenizer, text, 2).is_err());
        assert!(tokenizer_windows(&tokenizer, "", 4).unwrap().is_empty());
    }
}
