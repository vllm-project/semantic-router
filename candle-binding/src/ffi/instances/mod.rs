//! Owned native task instances. Handles name bindings, never global role slots.
//!
//! Every call acquires an Arc before leaving the registry lock. Removing a
//! handle cannot unload a model underneath an in-flight native operation.

mod generative;
mod tasks;
#[cfg(test)]
mod tests;
mod transport;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{anyhow, bail, ensure, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use tokenizers::{PostProcessor, Tokenizer};

use crate::classifiers::lora::token_lora::LoRATokenClassifier;
use crate::core::similarity::BertSimilarity;
use crate::model_architectures::generative::{
    Qwen3GuardConfig, Qwen3GuardModel, Qwen3MultiLoRAClassifier,
};
use crate::model_architectures::lora::bert_lora::{
    HighPerformanceBertClassifier, HighPerformanceBertTokenClassifier,
};
use crate::model_architectures::model_factory::ModelFactory;
use crate::model_architectures::traditional::bert::{
    TraditionalBertClassifier, TraditionalBertTokenClassifier,
};
use crate::model_architectures::traditional::deberta_v3::DebertaV3Classifier;
use crate::model_architectures::traditional::modernbert::{
    ModernBertBackbone, TraditionalModernBertClassifier, TraditionalModernBertTokenClassifier,
};

#[derive(Clone, Deserialize)]
struct Options {
    model_path: String,
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    device: String,
    #[serde(default)]
    precision: String,
    #[serde(default)]
    max_input_tokens: usize,
    #[serde(default)]
    overflow: String,
    #[serde(default)]
    adapters: Vec<AdapterSpec>,
    #[serde(default)]
    generation_max_tokens: usize,
}

#[derive(Clone, Deserialize)]
struct AdapterSpec {
    name: String,
    path: String,
}

#[derive(Clone, Serialize)]
struct Info {
    resource_id: u64,
    task: String,
    model_type: String,
    device: String,
    precision: String,
    architectural_max_tokens: usize,
    max_input_tokens: usize,
    overflow: String,
    labels: Vec<String>,
    modalities: Vec<String>,
}

// Internal storage only: each FFI inference entry point has a distinct request
// and response type, and rejects a handle of the wrong task.
enum Model {
    Backbone(Box<ModernBertBackbone>),
    Sequence(Box<TraditionalModernBertClassifier>),
    Token(Box<TraditionalModernBertTokenClassifier>),
    Bert(Box<TraditionalBertClassifier>),
    BertToken(Box<TraditionalBertTokenClassifier>),
    BertEmbedding(Box<BertSimilarity>),
    Deberta(Box<DebertaV3Classifier>),
    Embedding(Box<ModelFactory>),
    MergedBert(Box<HighPerformanceBertClassifier>),
    MergedBertToken(Box<HighPerformanceBertTokenClassifier>),
    LoRAToken(Box<LoRATokenClassifier>),
    Guard(Box<Mutex<Qwen3GuardModel>>),
    Generative(Box<Mutex<Qwen3MultiLoRAClassifier>>),
}

struct Instance {
    model: Model,
    info: Info,
    tokenizer: Option<Tokenizer>,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static INSTANCES: LazyLock<Mutex<HashMap<u64, Arc<Instance>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn insert(instance: Arc<Instance>) -> Result<u64> {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    ensure!(id != 0, "native handle capacity exhausted");
    INSTANCES
        .lock()
        .map_err(|_| anyhow!("native handle registry poisoned"))?
        .insert(id, instance);
    Ok(id)
}

fn get(handle: u64) -> Result<Arc<Instance>> {
    INSTANCES
        .lock()
        .map_err(|_| anyhow!("native handle registry poisoned"))?
        .get(&handle)
        .cloned()
        .ok_or_else(|| anyhow!("closed: native instance is closed or invalid"))
}

fn close(handle: u64) -> Result<()> {
    // Drop outside the registry mutex; a device destructor may synchronize work.
    let removed = INSTANCES
        .lock()
        .map_err(|_| anyhow!("native handle registry poisoned"))?
        .remove(&handle);
    drop(removed);
    Ok(())
}

fn resolve_device(name: &str) -> Result<Device> {
    match name {
        "" | "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda:0" => Ok(Device::new_cuda(0)?),
        #[cfg(feature = "metal")]
        "metal:0" => Ok(Device::new_metal(0)?),
        _ => bail!("capability: requested Candle device is unavailable: {name}"),
    }
}

fn device_name(device: &Device) -> &'static str {
    if device.is_cpu() {
        "cpu"
    } else if device.is_cuda() {
        "cuda:0"
    } else {
        "metal:0"
    }
}

// The same postprocessor runs for budget accounting and the actual model input.
// Reject undersized budgets before tokenizers can subtract special tokens.
fn task_tokenizer(path: &str, max_input_tokens: usize) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(format!("{path}/tokenizer.json"))
        .map_err(|e| anyhow!(e.to_string()))?;
    let special_tokens = tokenizer
        .get_post_processor()
        .map_or(0, |processor| processor.added_tokens(false));
    ensure!(
        max_input_tokens >= special_tokens,
        "capability: input budget {max_input_tokens} is smaller than the tokenizer's {special_tokens} special tokens"
    );
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow!(e.to_string()))?;
    tokenizer.with_padding(None);
    Ok(tokenizer)
}

fn load(mut options: Options, task: &str) -> Result<Arc<Instance>> {
    let device = resolve_device(&options.device)?;
    let generative = matches!(task, "guard" | "generative");
    let precision = if generative && !device.is_cpu() {
        "bfloat16"
    } else {
        "float32"
    };
    let requested_precision = match options.precision.as_str() {
        "fp32" => "float32",
        "bf16" => "bfloat16",
        other => other,
    };
    ensure!(requested_precision.is_empty() || requested_precision == precision, "capability: requested precision does not match the selected model/device execution ({precision})");
    ensure!(
        task == "generative" || options.adapters.is_empty(),
        "configuration: adapters require the generative task loader"
    );
    let use_cpu = device.is_cpu();
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{}/config.json",
        options.model_path
    ))?)?;
    let architectural_max_tokens = raw["max_position_embeddings"]
        .as_u64()
        .or_else(|| raw["text_max_position_embeddings"].as_u64())
        .unwrap_or(512) as usize;
    let limit = if task == "embedding" || generative {
        architectural_max_tokens
    } else {
        architectural_max_tokens.min(512)
    };
    let max_input_tokens = if options.max_input_tokens == 0 {
        limit
    } else {
        options.max_input_tokens
    };
    ensure!(
        max_input_tokens > 0 && max_input_tokens <= limit,
        "capability: input budget exceeds the task limit ({limit})"
    );
    let tokenizer = if task == "backbone" {
        None
    } else {
        let tokenizer = task_tokenizer(&options.model_path, max_input_tokens)?;
        Some(tokenizer)
    };
    if options.model_type.is_empty() {
        options.model_type = raw
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("modernbert")
            .to_owned();
    }
    let modern = matches!(
        options.model_type.as_str(),
        "modernbert" | "mmbert" | "mmbert32k" | "mmbert-32k"
    );
    let model = match task {
        "backbone" if modern => {
            Model::Backbone(ModernBertBackbone::load(&options.model_path, &device)?.into())
        }
        "sequence" | "nli" if modern => {
            let model =
                TraditionalModernBertClassifier::load_from_directory(&options.model_path, use_cpu)?;
            ensure!(
                device_name(model.device()) == device_name(&device),
                "capability: classifier silently selected a different device"
            );
            Model::Sequence(model.into())
        }
        "sequence" if options.model_type == "bert" => {
            ensure!(
                use_cpu || device.is_cuda(),
                "capability: BERT classifier does not support the selected device"
            );
            let classes = raw["id2label"]
                .as_object()
                .map(|m| m.len())
                .or_else(|| raw["num_labels"].as_u64().map(|n| n as usize))
                .ok_or_else(|| anyhow!("configuration: missing class count"))?;
            let model = TraditionalBertClassifier::new(&options.model_path, classes, use_cpu)?;
            ensure!(
                device_name(model.device()) == device_name(&device),
                "capability: classifier silently selected a different device"
            );
            Model::Bert(model.into())
        }
        "sequence"
            if matches!(
                options.model_type.as_str(),
                "deberta" | "deberta-v2" | "deberta_v3"
            ) =>
        {
            let model = DebertaV3Classifier::new(&options.model_path, use_cpu)?;
            ensure!(
                device_name(model.device()) == device_name(&device),
                "capability: classifier silently selected a different device"
            );
            Model::Deberta(model.into())
        }
        "token" | "hallucination" if modern => {
            let model = TraditionalModernBertTokenClassifier::new(&options.model_path, use_cpu)?;
            ensure!(
                device_name(model.device()) == device_name(&device),
                "capability: token classifier silently selected a different device"
            );
            Model::Token(model.into())
        }
        "token" if options.model_type == "bert" => {
            ensure!(
                use_cpu || device.is_cuda(),
                "capability: BERT token classifier does not support selected device"
            );
            let model = TraditionalBertTokenClassifier::new(&options.model_path, 0, use_cpu)?;
            ensure!(
                device_name(model.device()) == device_name(&device),
                "capability: classifier silently selected a different device"
            );
            Model::BertToken(model.into())
        }
        "sequence" if options.model_type == "bert_lora" => {
            ensure!(
                use_cpu || device.is_cuda(),
                "capability: merged BERT classifier does not support selected device"
            );
            let classes = raw["id2label"]
                .as_object()
                .map(|m| m.len())
                .ok_or_else(|| anyhow!("configuration: missing label mapping"))?;
            Model::MergedBert(
                HighPerformanceBertClassifier::new(&options.model_path, classes, use_cpu)?.into(),
            )
        }
        "token" if options.model_type == "bert_lora" => {
            ensure!(
                use_cpu || device.is_cuda(),
                "capability: merged BERT token classifier does not support selected device"
            );
            let classes = raw["id2label"]
                .as_object()
                .map(|m| m.len())
                .ok_or_else(|| anyhow!("configuration: missing label mapping"))?;
            Model::MergedBertToken(
                HighPerformanceBertTokenClassifier::new(&options.model_path, classes, use_cpu)?
                    .into(),
            )
        }
        "token" if options.model_type == "lora_token" => {
            ensure!(
                use_cpu || device.is_cuda(),
                "capability: LoRA token classifier does not support selected device"
            );
            let mut model = LoRATokenClassifier::new(&options.model_path, use_cpu)?;
            model.include_all_predictions();
            Model::LoRAToken(model.into())
        }
        "guard" => {
            let mut config = Qwen3GuardConfig::default();
            if options.generation_max_tokens > 0 {
                config.max_tokens = options.generation_max_tokens;
            }
            Model::Guard(
                Mutex::new(Qwen3GuardModel::new(
                    &options.model_path,
                    &device,
                    Some(config),
                )?)
                .into(),
            )
        }
        "generative" => {
            let mut model = Qwen3MultiLoRAClassifier::new(&options.model_path, &device)?;
            let mut names = std::collections::HashSet::new();
            for adapter in &options.adapters {
                ensure!(
                    !adapter.name.is_empty() && names.insert(&adapter.name),
                    "configuration: duplicate or empty adapter name"
                );
                model.load_adapter(&adapter.name, &adapter.path)?;
            }
            Model::Generative(Mutex::new(model).into())
        }
        "embedding" if options.model_type == "bert" => {
            ensure!(
                use_cpu || device.is_cuda(),
                "capability: BERT embedding does not support selected device"
            );
            let model = BertSimilarity::new(&options.model_path, use_cpu)?;
            ensure!(
                device_name(model.device()) == device_name(&device),
                "capability: embedding silently selected a different device"
            );
            Model::BertEmbedding(model.into())
        }
        "embedding" => {
            let mut factory = ModelFactory::new(device.clone());
            match options.model_type.as_str() {
                "qwen3" => factory.register_qwen3_embedding_model(&options.model_path)?,
                "gemma" | "gemma3" => {
                    factory.register_gemma_embedding_model(&options.model_path)?
                }
                "mmbert" | "mmbert_embedding" | "modernbert" => {
                    factory.register_mmbert_embedding_model(&options.model_path)?
                }
                "multimodal" => factory.register_multimodal_embedding_model(&options.model_path)?,
                _ => bail!("capability: unsupported embedding model type"),
            }
            Model::Embedding(factory.into())
        }
        _ => bail!("capability: model type does not implement requested task"),
    };
    let overflow = if options.overflow.is_empty() {
        if generative {
            "reject".to_owned()
        } else {
            "truncate".to_owned()
        }
    } else {
        options.overflow
    };
    ensure!(
        matches!(overflow.as_str(), "truncate" | "reject"),
        "configuration: overflow must be truncate or reject"
    );
    ensure!(!generative || overflow == "reject", "capability: generative task inputs use reject overflow to preserve full templates and candidates");
    let labels = if task == "backbone" {
        vec![]
    } else {
        raw["id2label"]
            .as_object()
            .map(|map| {
                (0..map.len())
                    .map(|i| {
                        map.get(&i.to_string())
                            .and_then(|v| v.as_str())
                            .map(str::to_owned)
                            .ok_or_else(|| {
                                anyhow!("configuration: labels must have consecutive numeric IDs")
                            })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default()
    };
    if task == "nli" {
        ensure!(labels.len() == 3, "capability: NLI requires three labels");
    }
    if task == "hallucination" {
        ensure!(
            labels.len() == 2,
            "capability: hallucination requires two labels"
        );
    }
    let modalities = if task == "embedding" {
        if let Model::Embedding(factory) = &model {
            factory
                .get_multimodal_model()
                .map(|m| m.supported_modalities())
                .unwrap_or_else(|| vec!["text"])
        } else {
            vec!["text"]
        }
    } else {
        vec![]
    }
    .into_iter()
    .map(str::to_owned)
    .collect();
    crate::core::device::drain_loader_queue(&device);
    Ok(Arc::new(Instance {
        model,
        tokenizer,
        info: Info {
            resource_id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            task: task.to_owned(),
            model_type: options.model_type,
            device: device_name(&device).to_owned(),
            precision: precision.to_owned(),
            architectural_max_tokens,
            max_input_tokens,
            overflow,
            labels,
            modalities,
        },
    }))
}

fn bind_head(source: &Instance, path: &str, task: &str) -> Result<Arc<Instance>> {
    let tokenizer = task_tokenizer(path, source.info.max_input_tokens)?;
    let model = match (&source.model, task) {
        (Model::Backbone(model), "sequence") => {
            Model::Sequence(model.bind_sequence_head(path)?.into())
        }
        (Model::Backbone(model), "token") => Model::Token(model.bind_token_head(path)?.into()),
        (Model::Sequence(model), "sequence") => {
            Model::Sequence(model.bind_sequence_head(path)?.into())
        }
        (Model::Sequence(model), "token") => Model::Token(model.bind_token_head(path)?.into()),
        (Model::Token(model), "sequence") => {
            Model::Sequence(model.bind_sequence_head(path)?.into())
        }
        (Model::Token(model), "token") => Model::Token(model.bind_token_head(path)?.into()),
        _ => bail!("capability: only ModernBERT backbones support explicit head binding"),
    };
    let mut info = source.info.clone();
    info.task = task.to_owned();
    let mapping = crate::ffi::classify::load_id2label_from_config(&format!("{path}/config.json"))?;
    info.labels = (0..mapping.len())
        .map(|id| {
            mapping
                .get(&id.to_string())
                .cloned()
                .ok_or_else(|| anyhow!("configuration: labels must have consecutive numeric IDs"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(Instance {
        model,
        info,
        tokenizer: Some(tokenizer),
    }))
}
