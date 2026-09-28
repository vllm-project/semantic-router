//! Real sequence/token classifier regression; missing models or fixtures fail.
//!
//! Usage: benchmark_classifier_context MODEL_DIR sequence|token MAX_TOKENS cpu|rocm FIXTURES.json
//! Fixtures are an array of {"id": "...", "text": "...", "expected_tokens": 1024}.
//! Token counts include special tokens. Use the checkpoint's own tokenizer to
//! construct fixtures, and compare probabilities/entities with the same weights.

use onnx_semantic_router::model_architectures::classification::{
    ClassifierExecutionProvider, MmBertSequenceClassifier, MmBertTokenClassifier,
};
use serde_json::json;
use std::{error::Error, fs, time::Instant};
use tokenizers::{Tokenizer, TruncationParams};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        return Err("expected MODEL_DIR sequence|token MAX_TOKENS cpu|rocm FIXTURES.json".into());
    }
    let limit: usize = args[3].parse()?;
    let provider = match args[4].as_str() {
        "cpu" => ClassifierExecutionProvider::Cpu,
        "rocm" => ClassifierExecutionProvider::Rocm,
        _ => return Err("provider must be cpu or rocm".into()),
    };
    let fixtures: Vec<serde_json::Value> = serde_json::from_str(&fs::read_to_string(&args[5])?)?;
    if fixtures.is_empty() {
        return Err("fixtures must contain at least one input".into());
    }
    let mut sequence = if args[2] == "sequence" {
        Some(MmBertSequenceClassifier::load_with_max_sequence_length(
            &args[1], provider, limit,
        )?)
    } else {
        None
    };
    let mut token = if args[2] == "token" {
        Some(MmBertTokenClassifier::load_with_max_sequence_length(
            &args[1], provider, limit,
        )?)
    } else {
        None
    };
    if sequence.is_none() && token.is_none() {
        return Err("task must be sequence or token".into());
    }
    // The loader validates that the budget can hold the tokenizer's special
    // tokens before the benchmark configures its independent count check.
    let mut tokenizer =
        Tokenizer::from_file(format!("{}/tokenizer.json", args[1])).map_err(|e| e.to_string())?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: limit,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
    for fixture in fixtures {
        let text = fixture["text"].as_str().ok_or("missing text")?;
        let count = tokenizer
            .encode(text, true)
            .map_err(|e| e.to_string())?
            .len();
        if fixture["expected_tokens"].as_u64() != Some(count as u64) {
            return Err(format!(
                "fixture {}: expected {} tokens, encoded {count}",
                fixture["id"], fixture["expected_tokens"]
            )
            .into());
        }
        let start = Instant::now();
        let result = if let Some(model) = sequence.as_mut() {
            let result = model.classify(text)?;
            json!({"label": result.label, "probabilities": result.probabilities})
        } else {
            let result = token.as_mut().unwrap().detect_entities(text)?;
            json!({"entities": result.entities.iter().map(|e| json!({
                "text":e.text,"type":e.entity_type,"start":e.start,"end":e.end,"confidence":e.confidence
            })).collect::<Vec<_>>()})
        };
        println!(
            "{}",
            json!({"id":fixture["id"],"tokens":count,"max_tokens":limit,
            "elapsed_ms":start.elapsed().as_secs_f64()*1000.0,"result":result})
        );
    }
    Ok(())
}
