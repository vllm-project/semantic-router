//! Real sequence/token classifier regression; missing models or fixtures fail.
//!
//! Usage: benchmark_classifier_context MODEL_DIR sequence|token MAX_TOKENS cpu FIXTURES.json
//! Fixtures are an array of {"id": "...", "text": "...", "expected_tokens": 1024}.
//! Token counts include special tokens. Use the checkpoint's own tokenizer to
//! construct fixtures, and compare probabilities/entities with the same weights.

use candle_semantic_router::model_architectures::traditional::modernbert::{
    TraditionalModernBertClassifier, TraditionalModernBertTokenClassifier,
};
use serde_json::json;
use std::{error::Error, fs, time::Instant};
use tokenizers::{Tokenizer, TruncationParams};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        return Err("expected MODEL_DIR sequence|token MAX_TOKENS cpu FIXTURES.json".into());
    }
    let limit: usize = args[3].parse()?;
    if args[4] != "cpu" {
        return Err("this benchmark requires the cpu provider".into());
    }
    let sequence = if args[2] == "sequence" {
        Some(
            TraditionalModernBertClassifier::load_from_directory_with_max_sequence_length(
                &args[1], true, limit,
            )?,
        )
    } else {
        None
    };
    let token = if args[2] == "token" {
        Some(
            TraditionalModernBertTokenClassifier::new_with_max_sequence_length(
                &args[1], true, limit,
            )?,
        )
    } else {
        None
    };
    if sequence.is_none() && token.is_none() {
        return Err("task must be sequence or token".into());
    }
    let mut tokenizer =
        Tokenizer::from_file(format!("{}/tokenizer.json", args[1])).map_err(|e| e.to_string())?;
    let config: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(format!("{}/config.json", args[1]))?)?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: limit,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
    let fixtures: Vec<serde_json::Value> = serde_json::from_str(&fs::read_to_string(&args[5])?)?;
    if fixtures.is_empty() {
        return Err("fixtures must contain at least one input".into());
    }
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
        let result = if let Some(model) = sequence.as_ref() {
            let (label_id, confidence, probabilities) =
                model.classify_text_with_probabilities(text)?;
            let label = config["id2label"][label_id.to_string()]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| label_id.to_string());
            json!({"label":label,"label_id":label_id,"confidence":confidence,
                "probabilities":probabilities})
        } else {
            let model = token.as_ref().unwrap();
            let entities = model.classify_tokens(text)?;
            json!({"entities":entities.iter().map(|(text,label_id,confidence,start,end)| {
                // TokenClassifier's public label accessor currently returns
                // None. Resolve the checkpoint's BIO labels directly here.
                let label = config["id2label"][label_id.to_string()].as_str().unwrap_or("UNKNOWN");
                let entity_type = label.strip_prefix("B-")
                    .or_else(|| label.strip_prefix("I-")).unwrap_or(label);
                json!({"text":text,"type":entity_type,"label_id":label_id,
                    "start":start,"end":end,"confidence":confidence})
            }).collect::<Vec<_>>()})
        };
        println!(
            "{}",
            json!({"id":fixture["id"],"tokens":count,"max_tokens":limit,
                "elapsed_ms":start.elapsed().as_secs_f64()*1000.0,"result":result})
        );
    }
    Ok(())
}
