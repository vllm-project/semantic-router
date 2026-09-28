//! Vela Halu grounding has an explicit pair template and immutable token policy.
use super::*;
use tokenizers::{TruncationDirection, TruncationParams, TruncationStrategy};

fn policy(path: &str) -> UnifiedResult<()> {
    let read = |name: &str| -> UnifiedResult<serde_json::Value> {
        let path = std::path::Path::new(path).join(name);
        let bytes =
            std::fs::read(&path).map_err(|e| errors::config_error("vela_halu", &e.to_string()))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| errors::config_error("vela_halu", &e.to_string()))
    };
    let policy = read("operating_point.json")?;
    let config = read("config.json")?;
    if policy["max_input_tokens"] != 8192
        || policy["token_threshold"] != 0.5
        || policy["threshold_comparison"] != "strictly_greater"
        || policy["label2id"]["supported"] != 0
        || policy["label2id"]["hallucinated"] != 1
        || policy["input_pair"]
            != serde_json::json!(["User request: {question}\n\n{context}", "answer"])
        || policy["answer_offsets"] != "Unicode code points"
        || config["id2label"]["0"] != "supported"
        || config["id2label"]["1"] != "hallucinated"
        || config["label2id"]["supported"] != 0
        || config["label2id"]["hallucinated"] != 1
    {
        return Err(errors::config_error("vela_halu", "artifact must declare the supported Vela Halu operating point and supported=0, hallucinated=1 label order"));
    }
    Ok(())
}

pub fn load_grounded(mut options: InstanceOptions) -> UnifiedResult<u64> {
    policy(&options.model_path)?;
    let limit = options.max_input_tokens.unwrap_or(8192);
    if limit == 0 || limit > 8192 {
        return Err(errors::config_error(
            "input_tokens",
            "Vela Halu task budget is at most 8192 tokens",
        ));
    }
    options.max_input_tokens = Some(limit);
    let options = fresh_options(options);
    let model = MmBertTokenClassifier::load_with_options(&options)?;
    if model.config().num_labels != 2 {
        return Err(errors::config_error(
            "vela_halu",
            "grounding requires a binary token head",
        ));
    }
    prepare(Model::GroundedToken(model), options)
}

pub fn grounded(
    handle: u64,
    context: &str,
    question: &str,
    answer: &str,
) -> UnifiedResult<TokenSpans> {
    let instance = get(handle)?;
    let mut model = instance.model.lock();
    let Model::GroundedToken(model) = &mut *model else {
        return Err(instance.wrong_task("grounded_text"));
    };
    let prompt = format!("User request: {question}\n\n{context}");
    let mut tokenizer = instance.tokenizer.clone();
    let original = tokenizer
        .encode((prompt.as_str(), answer), true)
        .map_err(|e| errors::tokenization_error(&e.to_string()))?;
    let original_tokens = original.len();
    let encoding = if original_tokens > instance.effective_limit {
        if matches!(instance.options.overflow, Overflow::Reject) {
            return Err(errors::validation(
                "input_tokens",
                &format!("at most {}", instance.effective_limit),
                &original_tokens.to_string(),
            ));
        }
        let special = tokenizer
            .get_post_processor()
            .map_or(0, |p| tokenizers::PostProcessor::added_tokens(p, true));
        if instance.effective_limit < special {
            return Err(errors::validation(
                "input_tokens",
                "complete pair template",
                "budget too small",
            ));
        }
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: instance.effective_limit,
                strategy: TruncationStrategy::OnlyFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|e| errors::validation("input_tokens", "complete answer", &e.to_string()))?;
        tokenizer
            .encode((prompt.as_str(), answer), true)
            .map_err(|e| errors::validation("input_tokens", "complete answer", &e.to_string()))?
    } else {
        original
    };
    let probabilities = model.classify_encoded_tokens(&encoding)?;
    instance.completed.fetch_add(1, Ordering::Relaxed);
    let mut spans: Vec<Span> = Vec::new();
    let mut current: Option<Span> = None;
    for ((sequence, &(start, end)), scores) in encoding
        .get_sequence_ids()
        .iter()
        .zip(encoding.get_offsets())
        .zip(&probabilities)
    {
        if *sequence != Some(1) || end <= start {
            continue;
        }
        if scores.len() != 2
            || !scores.iter().all(|score| score.is_finite())
            || answer.get(start..end).is_none()
        {
            return Err(errors::inference_error(
                "token_spans",
                "invalid grounding scores or answer byte offsets",
            ));
        }
        if scores[1] > 0.5 {
            if let Some(span) = current.as_mut() {
                span.end = span.end.max(end);
                span.confidence = span.confidence.max(scores[1]);
            } else {
                current = Some(Span {
                    text: String::new(),
                    entity_type: "HALLUCINATED".into(),
                    start,
                    end,
                    confidence: scores[1],
                });
            }
        } else if let Some(span) = current.take() {
            spans.push(span);
        }
    }
    if let Some(span) = current {
        spans.push(span);
    }
    for span in &mut spans {
        span.text = answer[span.start..span.end].to_owned();
    }
    Ok(TokenSpans {
        spans,
        offset_unit: "utf8_bytes",
        input: InputUsage {
            original_tokens,
            processed_tokens: encoding.len(),
            truncated: original_tokens > encoding.len(),
        },
    })
}
