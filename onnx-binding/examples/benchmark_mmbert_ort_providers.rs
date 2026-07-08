//! mmBERT ONNX Runtime execution-provider validation harness.
//!
//! This is intentionally a low-level ONNX Runtime probe: it feeds known-valid
//! token ids directly to mmBERT layer ONNX files so provider load, compile,
//! inference, parity, and latency can be investigated without tokenizer or
//! pooling code in the middle.
//!
//! Example:
//!   cargo run --release --features migraphx-dynamic \
//!     --example benchmark_mmbert_ort_providers -- \
//!     --model-dir ./mmbert-onnx/onnx \
//!     --providers cpu,migraphx,rocm \
//!     --layers 6,22 \
//!     --seq-lens 1,32,128 \
//!     --batch-sizes 1,4 \
//!     --jsonl /tmp/mmbert-provider-bench.jsonl

use half::f16;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::json;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Provider {
    Cpu,
    Migraphx,
    Rocm,
    Auto,
}

impl Provider {
    fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "migraphx" | "migx" => Ok(Self::Migraphx),
            "rocm" => Ok(Self::Rocm),
            "auto" => Ok(Self::Auto),
            other => Err(format!("unknown provider {other:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Migraphx => "migraphx",
            Self::Rocm => "rocm",
            Self::Auto => "auto",
        }
    }
}

#[derive(Debug)]
struct Config {
    model_dir: PathBuf,
    providers: Vec<Provider>,
    layers: Vec<usize>,
    seq_lens: Vec<usize>,
    batch_sizes: Vec<usize>,
    extra_models: Vec<ModelSpec>,
    warmup: usize,
    iters: usize,
    token_id: i64,
    jsonl: Option<PathBuf>,
    summary_md: Option<PathBuf>,
    fail_on_error: bool,
}

#[derive(Debug, Clone)]
struct ModelSpec {
    name: String,
    path: PathBuf,
    layer: Option<usize>,
}

#[derive(Clone)]
struct OutputData {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
struct LatencyStats {
    avg_ms: f64,
    min_ms: f64,
    max_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

struct SessionInfo {
    session: Session,
    selected_provider: String,
    fallback_reason: Option<String>,
}

struct SummaryRow {
    model: String,
    provider: String,
    selected_provider: String,
    batch_size: usize,
    seq_len: usize,
    first_ms: f64,
    avg_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    throughput_items_per_s: f64,
    parity_max_abs: Option<f64>,
    parity_mean_abs: Option<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse()?;
    let mut jsonl = if let Some(path) = &config.jsonl {
        Some(BufWriter::new(File::create(path)?))
    } else {
        None
    };

    ort::init().commit()?;

    emit_inventory(&config, jsonl.as_mut())?;

    let mut had_error = false;
    let mut summary_rows = Vec::new();

    for &layer in &config.layers {
        let Some(model_path) = resolve_model_path(&config.model_dir, layer) else {
            had_error = true;
            emit_event(
                jsonl.as_mut(),
                json!({
                    "event": "model",
                    "layer": layer,
                    "status": "error",
                    "error": "model file not found",
                    "model_dir": config.model_dir.display().to_string(),
                }),
            )?;
            eprintln!(
                "layer {layer}: model file not found under {}",
                config.model_dir.display()
            );
            continue;
        };

        println!();
        println!("layer {layer}: {}", model_path.display());

        had_error |= run_model(
            &ModelSpec {
                name: format!("mmbert-layer-{layer}"),
                path: model_path,
                layer: Some(layer),
            },
            &config,
            jsonl.as_mut(),
            &mut summary_rows,
        )?;
    }

    for model in &config.extra_models {
        println!();
        println!("model {}: {}", model.name, model.path.display());
        if !model.path.exists() {
            had_error = true;
            emit_event(
                jsonl.as_mut(),
                json!({
                    "event": "model",
                    "model_name": model.name,
                    "status": "error",
                    "error": "model file not found",
                    "model": model.path.display().to_string(),
                }),
            )?;
            eprintln!(
                "model {}: file not found at {}",
                model.name,
                model.path.display()
            );
            continue;
        }
        had_error |= run_model(model, &config, jsonl.as_mut(), &mut summary_rows)?;
    }

    if let Some(path) = &config.summary_md {
        write_summary_markdown(path, &summary_rows)?;
    }

    if had_error && config.fail_on_error {
        Err("one or more provider checks failed".into())
    } else {
        Ok(())
    }
}

impl Config {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut model_dir = env::var("MMBERT_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./mmbert-onnx/onnx"));
        let mut providers = parse_csv("cpu,migraphx,rocm", Provider::parse)?;
        let mut layers = parse_csv("6,11,16,22", parse_usize)?;
        let mut seq_lens = parse_csv("1,32,128,512", parse_usize)?;
        let mut batch_sizes = parse_csv("1,4", parse_usize)?;
        let mut extra_models = Vec::new();
        let mut warmup = 3;
        let mut iters = 20;
        let mut token_id = 1_i64;
        let mut jsonl = None;
        let mut summary_md = None;
        let mut fail_on_error = false;

        let args: Vec<String> = env::args().skip(1).collect();
        let mut idx = 0;
        while idx < args.len() {
            let arg = &args[idx];
            match arg.as_str() {
                "--model-dir" => {
                    idx += 1;
                    model_dir = PathBuf::from(require_value(&args, idx, arg)?);
                }
                "--providers" => {
                    idx += 1;
                    providers = parse_csv(require_value(&args, idx, arg)?, Provider::parse)?;
                }
                "--layers" => {
                    idx += 1;
                    layers = parse_csv(require_value(&args, idx, arg)?, parse_usize)?;
                }
                "--seq-lens" => {
                    idx += 1;
                    seq_lens = parse_csv(require_value(&args, idx, arg)?, parse_usize)?;
                }
                "--batch-sizes" => {
                    idx += 1;
                    batch_sizes = parse_csv(require_value(&args, idx, arg)?, parse_usize)?;
                }
                "--model" => {
                    idx += 1;
                    extra_models.push(parse_model_spec(require_value(&args, idx, arg)?)?);
                }
                "--warmup" => {
                    idx += 1;
                    warmup = require_value(&args, idx, arg)?.parse()?;
                }
                "--iters" => {
                    idx += 1;
                    iters = require_value(&args, idx, arg)?.parse()?;
                }
                "--token-id" => {
                    idx += 1;
                    token_id = require_value(&args, idx, arg)?.parse()?;
                }
                "--jsonl" => {
                    idx += 1;
                    jsonl = Some(PathBuf::from(require_value(&args, idx, arg)?));
                }
                "--summary-md" => {
                    idx += 1;
                    summary_md = Some(PathBuf::from(require_value(&args, idx, arg)?));
                }
                "--fail-on-error" => fail_on_error = true,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other:?}").into()),
            }
            idx += 1;
        }

        if providers.contains(&Provider::Cpu) {
            providers.sort_by_key(|provider| match provider {
                Provider::Cpu => 0,
                _ => 1,
            });
        }

        Ok(Self {
            model_dir,
            providers,
            layers,
            seq_lens,
            batch_sizes,
            extra_models,
            warmup,
            iters,
            token_id,
            jsonl,
            summary_md,
            fail_on_error,
        })
    }
}

fn run_model(
    model: &ModelSpec,
    config: &Config,
    mut jsonl: Option<&mut BufWriter<File>>,
    summary_rows: &mut Vec<SummaryRow>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut had_error = false;
    let mut baselines: HashMap<(usize, usize), OutputData> = HashMap::new();

    for &provider in &config.providers {
        let session_start = Instant::now();
        let session_info = match create_session(provider, &model.path) {
            Ok(session_info) => {
                let create_ms = session_start.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "  provider={:<8} selected={:<8} session=ok create_ms={:.2}",
                    provider.as_str(),
                    session_info.selected_provider,
                    create_ms
                );
                emit_event(
                    jsonl.as_mut().map(|writer| &mut **writer),
                    json!({
                        "event": "session",
                        "provider": provider.as_str(),
                        "selected_provider": session_info.selected_provider,
                        "fallback_reason": session_info.fallback_reason,
                        "model_name": model.name,
                        "layer": model.layer,
                        "status": "ok",
                        "create_ms": create_ms,
                        "model": model.path.display().to_string(),
                    }),
                )?;
                session_info
            }
            Err(error) => {
                had_error = true;
                println!(
                    "  provider={:<8} session=error error={}",
                    provider.as_str(),
                    error
                );
                emit_event(
                    jsonl.as_mut().map(|writer| &mut **writer),
                    json!({
                        "event": "session",
                        "provider": provider.as_str(),
                        "model_name": model.name,
                        "layer": model.layer,
                        "status": "error",
                        "error": error,
                        "model": model.path.display().to_string(),
                    }),
                )?;
                continue;
            }
        };

        let mut session = session_info.session;
        for &batch_size in &config.batch_sizes {
            for &seq_len in &config.seq_lens {
                match benchmark_shape(
                    &mut session,
                    provider,
                    &session_info.selected_provider,
                    model,
                    batch_size,
                    seq_len,
                    config,
                    &baselines,
                    jsonl.as_mut().map(|writer| &mut **writer),
                    summary_rows,
                ) {
                    Ok(Some(output)) if provider == Provider::Cpu => {
                        baselines.insert((batch_size, seq_len), output);
                    }
                    Ok(_) => {}
                    Err(error) => {
                        had_error = true;
                        println!(
                            "    batch={batch_size:<3} seq={seq_len:<5} status=error error={error}"
                        );
                        emit_event(
                            jsonl.as_mut().map(|writer| &mut **writer),
                            json!({
                                "event": "inference",
                                "provider": provider.as_str(),
                                "selected_provider": session_info.selected_provider,
                                "model_name": model.name,
                                "layer": model.layer,
                                "batch_size": batch_size,
                                "seq_len": seq_len,
                                "status": "error",
                                "error": error,
                            }),
                        )?;
                    }
                }
            }
        }
    }

    Ok(had_error)
}

fn create_session(provider: Provider, model_path: &Path) -> Result<SessionInfo, String> {
    match provider {
        Provider::Cpu => Session::builder()
            .and_then(|builder| builder.commit_from_file(model_path))
            .map(|session| SessionInfo {
                session,
                selected_provider: "cpu".to_string(),
                fallback_reason: None,
            })
            .map_err(|error| error.to_string()),
        Provider::Migraphx => {
            #[cfg(feature = "migraphx")]
            {
                let mut builder = Session::builder().map_err(|error| error.to_string())?;
                onnx_semantic_router::core::ort_migraphx::append_migraphx_execution_provider(
                    &mut builder,
                    0,
                )
                .map_err(|error| error.to_string())?;
                builder
                    .commit_from_file(model_path)
                    .map(|session| SessionInfo {
                        session,
                        selected_provider: "migraphx".to_string(),
                        fallback_reason: None,
                    })
                    .map_err(|error| error.to_string())
            }
            #[cfg(not(feature = "migraphx"))]
            {
                let _ = model_path;
                Err("migraphx feature is not enabled".to_string())
            }
        }
        Provider::Rocm => {
            #[cfg(feature = "rocm")]
            {
                use ort::execution_providers::ROCmExecutionProvider;
                Session::builder()
                    .and_then(|builder| {
                        builder.with_execution_providers([ROCmExecutionProvider::default()
                            .build()
                            .error_on_failure()])
                    })
                    .and_then(|builder| builder.commit_from_file(model_path))
                    .map(|session| SessionInfo {
                        session,
                        selected_provider: "rocm".to_string(),
                        fallback_reason: None,
                    })
                    .map_err(|error| error.to_string())
            }
            #[cfg(not(feature = "rocm"))]
            {
                let _ = model_path;
                Err("rocm feature is not enabled".to_string())
            }
        }
        Provider::Auto => {
            #[cfg(any(feature = "migraphx", feature = "rocm"))]
            {
                let ck_fa_lib =
                    onnx_semantic_router::core::ort_migraphx::ck_flash_attention_library_for_model(
                        model_path,
                    );
                if let Ok(amd_session) =
                    onnx_semantic_router::core::ort_migraphx::create_amd_session(
                        model_path,
                        ck_fa_lib.as_deref(),
                    )
                {
                    return Ok(SessionInfo {
                        session: amd_session.session,
                        selected_provider: amd_session.provider.to_string(),
                        fallback_reason: amd_session.fallback_reason,
                    });
                }
            }
            create_session(Provider::Cpu, model_path)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn benchmark_shape(
    session: &mut Session,
    provider: Provider,
    selected_provider: &str,
    model: &ModelSpec,
    batch_size: usize,
    seq_len: usize,
    config: &Config,
    baselines: &HashMap<(usize, usize), OutputData>,
    jsonl: Option<&mut BufWriter<File>>,
    summary_rows: &mut Vec<SummaryRow>,
) -> Result<Option<OutputData>, String> {
    let vram_before_pct = read_rocm_vram_used_pct();
    let (first_ms, output) = timed_run(session, batch_size, seq_len, config.token_id, true)?;

    for _ in 0..config.warmup {
        timed_run(session, batch_size, seq_len, config.token_id, false)?;
    }

    let mut times = Vec::with_capacity(config.iters);
    for _ in 0..config.iters {
        let (elapsed_ms, _) = timed_run(session, batch_size, seq_len, config.token_id, false)?;
        times.push(elapsed_ms);
    }

    let stats = LatencyStats::from_samples(&times);
    let throughput_items_per_s = if stats.avg_ms > 0.0 {
        batch_size as f64 * 1000.0 / stats.avg_ms
    } else {
        0.0
    };
    let vram_after_pct = read_rocm_vram_used_pct();
    let vram_peak_observed_pct = match (vram_before_pct, vram_after_pct) {
        (Some(before), Some(after)) => Some(before.max(after)),
        (Some(before), None) => Some(before),
        (None, Some(after)) => Some(after),
        (None, None) => None,
    };
    let parity = baselines
        .get(&(batch_size, seq_len))
        .and_then(|baseline| compare_outputs(baseline, &output));

    println!(
        "    batch={batch_size:<3} seq={seq_len:<5} first={first_ms:>8.2}ms avg={:>8.3}ms p95={:>8.3}ms p99={:>8.3}ms tput={:>8.1}/s output={} {:?}{}",
        stats.avg_ms,
        stats.p95_ms,
        stats.p99_ms,
        throughput_items_per_s,
        output.name,
        output.shape,
        parity
            .map(|(max_abs, mean_abs)| format!(" parity max_abs={max_abs:.6} mean_abs={mean_abs:.6}"))
            .unwrap_or_default()
    );

    emit_event(
        jsonl,
        json!({
            "event": "inference",
            "provider": provider.as_str(),
            "selected_provider": selected_provider,
            "model_name": model.name,
            "model": model.path.display().to_string(),
            "layer": model.layer,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "status": "ok",
            "first_ms": first_ms,
            "avg_ms": stats.avg_ms,
            "min_ms": stats.min_ms,
            "max_ms": stats.max_ms,
            "p50_ms": stats.p50_ms,
            "p95_ms": stats.p95_ms,
            "p99_ms": stats.p99_ms,
            "throughput_items_per_s": throughput_items_per_s,
            "vram_before_pct": vram_before_pct,
            "vram_after_pct": vram_after_pct,
            "vram_peak_observed_pct": vram_peak_observed_pct,
            "output_name": output.name,
            "output_shape": output.shape,
            "parity_max_abs": parity.map(|value| value.0),
            "parity_mean_abs": parity.map(|value| value.1),
        }),
    )
    .map_err(|error| error.to_string())?;

    summary_rows.push(SummaryRow {
        model: model.name.clone(),
        provider: provider.as_str().to_string(),
        selected_provider: selected_provider.to_string(),
        batch_size,
        seq_len,
        first_ms,
        avg_ms: stats.avg_ms,
        p95_ms: stats.p95_ms,
        p99_ms: stats.p99_ms,
        throughput_items_per_s,
        parity_max_abs: parity.map(|value| value.0),
        parity_mean_abs: parity.map(|value| value.1),
    });

    Ok(Some(output))
}

fn timed_run(
    session: &mut Session,
    batch_size: usize,
    seq_len: usize,
    token_id: i64,
    extract_output: bool,
) -> Result<(f64, OutputData), String> {
    let input_ids = vec![token_id; batch_size * seq_len];
    let attention_mask = vec![1_i64; batch_size * seq_len];
    let input_ids_tensor = Tensor::from_array(([batch_size, seq_len], input_ids))
        .map_err(|error| error.to_string())?;
    let attention_mask_tensor = Tensor::from_array(([batch_size, seq_len], attention_mask))
        .map_err(|error| error.to_string())?;

    let start = Instant::now();
    let outputs = session
        .run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])
        .map_err(|error| error.to_string())?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let output = if extract_output {
        extract_first_output(&outputs)?
    } else {
        OutputData {
            name: String::new(),
            shape: Vec::new(),
            values: Vec::new(),
        }
    };

    Ok((elapsed_ms, output))
}

fn extract_first_output(outputs: &ort::session::SessionOutputs<'_>) -> Result<OutputData, String> {
    let Some((name, value)) = outputs.iter().next() else {
        return Err("session returned no outputs".to_string());
    };

    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return Ok(OutputData {
            name: name.to_string(),
            shape: shape.iter().map(|&dim| dim as usize).collect(),
            values: data.to_vec(),
        });
    }

    if let Ok((shape, data)) = value.try_extract_tensor::<f16>() {
        return Ok(OutputData {
            name: name.to_string(),
            shape: shape.iter().map(|&dim| dim as usize).collect(),
            values: data.iter().map(|value| value.to_f32()).collect(),
        });
    }

    Err(format!("first output {name} is not f32 or f16"))
}

fn compare_outputs(left: &OutputData, right: &OutputData) -> Option<(f64, f64)> {
    if left.shape != right.shape || left.values.len() != right.values.len() {
        return None;
    }

    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;

    for (left, right) in left.values.iter().zip(right.values.iter()) {
        let diff = (f64::from(*left) - f64::from(*right)).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
    }

    Some((max_abs, sum_abs / left.values.len().max(1) as f64))
}

impl LatencyStats {
    fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                avg_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
            };
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let sum: f64 = sorted.iter().sum();

        Self {
            avg_ms: sum / sorted.len() as f64,
            min_ms: *sorted.first().unwrap(),
            max_ms: *sorted.last().unwrap(),
            p50_ms: percentile(&sorted, 0.50),
            p95_ms: percentile(&sorted, 0.95),
            p99_ms: percentile(&sorted, 0.99),
        }
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let idx = ((sorted.len() as f64 * pct).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

fn resolve_model_path(model_dir: &Path, layer: usize) -> Option<PathBuf> {
    let candidates = [
        model_dir.join(format!("layer-{layer}")).join("model.onnx"),
        model_dir
            .join("onnx")
            .join(format!("layer-{layer}"))
            .join("model.onnx"),
    ];
    candidates.into_iter().find(|path| path.exists())
}

fn emit_inventory(
    config: &Config,
    jsonl: Option<&mut BufWriter<File>>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("mmBERT ORT provider benchmark");
    println!("  ort_minor_api={}", ort::MINOR_VERSION);
    println!("  model_dir={}", config.model_dir.display());
    println!(
        "  providers={}",
        config
            .providers
            .iter()
            .map(|provider| provider.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("  layers={:?}", config.layers);
    if !config.extra_models.is_empty() {
        println!(
            "  extra_models={}",
            config
                .extra_models
                .iter()
                .map(|model| format!("{}={}", model.name, model.path.display()))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    println!("  seq_lens={:?}", config.seq_lens);
    println!("  batch_sizes={:?}", config.batch_sizes);
    println!("  warmup={} iters={}", config.warmup, config.iters);
    println!(
        "  features: rocm={} migraphx={} cuda={}",
        cfg!(feature = "rocm"),
        cfg!(feature = "migraphx"),
        cfg!(feature = "cuda")
    );
    println!(
        "  ORT_DYLIB_PATH={}",
        env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "<unset>".to_string())
    );
    println!(
        "  LD_LIBRARY_PATH={}",
        env::var("LD_LIBRARY_PATH").unwrap_or_else(|_| "<unset>".to_string())
    );

    emit_event(
        jsonl,
        json!({
            "event": "inventory",
            "ort_minor_api": ort::MINOR_VERSION,
            "model_dir": config.model_dir.display().to_string(),
            "extra_models": config.extra_models.iter().map(|model| {
                json!({
                    "name": model.name,
                    "path": model.path.display().to_string(),
                })
            }).collect::<Vec<_>>(),
            "providers": config.providers.iter().map(|provider| provider.as_str()).collect::<Vec<_>>(),
            "layers": config.layers,
            "seq_lens": config.seq_lens,
            "batch_sizes": config.batch_sizes,
            "warmup": config.warmup,
            "iters": config.iters,
            "feature_rocm": cfg!(feature = "rocm"),
            "feature_migraphx": cfg!(feature = "migraphx"),
            "feature_cuda": cfg!(feature = "cuda"),
            "ort_dylib_path": env::var("ORT_DYLIB_PATH").ok(),
            "ld_library_path": env::var("LD_LIBRARY_PATH").ok(),
        }),
    )?;

    Ok(())
}

fn emit_event(
    jsonl: Option<&mut BufWriter<File>>,
    event: serde_json::Value,
) -> std::io::Result<()> {
    if let Some(writer) = jsonl {
        serde_json::to_writer(&mut *writer, &event)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    Ok(())
}

fn parse_csv<T>(
    values: &str,
    mut parse_one: impl FnMut(&str) -> Result<T, String>,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    values
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| parse_one(value).map_err(Into::into))
        .collect()
}

fn parse_usize(value: &str) -> Result<usize, String> {
    value
        .trim()
        .parse()
        .map_err(|error| format!("invalid integer {value:?}: {error}"))
}

fn parse_model_spec(value: &str) -> Result<ModelSpec, Box<dyn std::error::Error>> {
    let Some((name, path)) = value.split_once('=') else {
        return Err(format!("--model expects name=path, got {value:?}").into());
    };
    let name = name.trim();
    let path = path.trim();
    if name.is_empty() || path.is_empty() {
        return Err(format!("--model expects non-empty name=path, got {value:?}").into());
    }
    Ok(ModelSpec {
        name: name.to_string(),
        path: PathBuf::from(path),
        layer: None,
    })
}

fn read_rocm_vram_used_pct() -> Option<f64> {
    let output = Command::new("rocm-smi")
        .args(["--showmemuse", "--json"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let mut values = Vec::new();
    collect_vram_used_values(&json, None, &mut values);
    values.into_iter().reduce(f64::max)
}

fn collect_vram_used_values(value: &serde_json::Value, key: Option<&str>, values: &mut Vec<f64>) {
    match value {
        serde_json::Value::Object(map) => {
            for (child_key, child_value) in map {
                collect_vram_used_values(child_value, Some(child_key), values);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_vram_used_values(item, key, values);
            }
        }
        serde_json::Value::Number(number) => {
            if is_vram_used_key(key) {
                if let Some(value) = number.as_f64() {
                    values.push(value);
                }
            }
        }
        serde_json::Value::String(text) => {
            if is_vram_used_key(key) {
                if let Some(value) = parse_first_float(text) {
                    values.push(value);
                }
            }
        }
        _ => {}
    }
}

fn is_vram_used_key(key: Option<&str>) -> bool {
    let Some(key) = key else {
        return false;
    };
    let key = key.to_ascii_lowercase();
    (key.contains("vram") && (key.contains("used") || key.contains("allocated")))
        || key.contains("memory allocated")
}

fn parse_first_float(value: &str) -> Option<f64> {
    let number: String = value
        .chars()
        .skip_while(|ch| !ch.is_ascii_digit() && *ch != '.')
        .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
        .collect();
    number.parse().ok()
}

fn write_summary_markdown(path: &Path, rows: &[SummaryRow]) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# mmBERT ORT Provider Benchmark Summary")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "| Model | Provider | Selected | Batch | Seq | First ms | Avg ms | P95 ms | P99 ms | Throughput items/s | Parity max abs | Parity mean abs |"
    )?;
    writeln!(
        writer,
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for row in rows {
        writeln!(
            writer,
            "| {} | {} | {} | {} | {} | {:.2} | {:.3} | {:.3} | {:.3} | {:.1} | {} | {} |",
            row.model,
            row.provider,
            row.selected_provider,
            row.batch_size,
            row.seq_len,
            row.first_ms,
            row.avg_ms,
            row.p95_ms,
            row.p99_ms,
            row.throughput_items_per_s,
            row.parity_max_abs
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "n/a".to_string()),
            row.parity_mean_abs
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "n/a".to_string()),
        )?;
    }
    writer.flush()
}

fn require_value<'a>(args: &'a [String], idx: usize, flag: &str) -> Result<&'a str, String> {
    args.get(idx)
        .map(|value| value.as_str())
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_help() {
    println!(
        "Usage: benchmark_mmbert_ort_providers [OPTIONS]\n\
\n\
Options:\n\
  --model-dir PATH       Directory containing layer-N/model.onnx files\n\
  --providers LIST       Comma-separated providers: cpu,migraphx,rocm,auto\n\
  --layers LIST          Comma-separated layers, for example 6,22\n\
  --seq-lens LIST        Comma-separated sequence lengths\n\
  --batch-sizes LIST     Comma-separated batch sizes\n\
  --model NAME=PATH      Additional ONNX model file to benchmark; repeatable\n\
  --warmup N             Warmup iterations per shape\n\
  --iters N              Timed iterations per shape\n\
  --token-id ID          Legal token id to feed; default 1\n\
  --jsonl PATH           Write machine-readable JSONL results\n\
  --summary-md PATH      Write a Markdown summary table\n\
  --fail-on-error        Exit non-zero if any provider/model/inference fails\n\
  -h, --help             Show this help"
    );
}
