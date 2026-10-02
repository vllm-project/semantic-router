//! The Vela Halu task's published pair input and token decision policy.
use super::tasks::{HallucinationOutput, InputMetadata, Span};
use super::*;
use tokenizers::{Encoding, TruncationDirection, TruncationParams, TruncationStrategy};

pub(super) fn validate_policy(path: &str, config: &serde_json::Value) -> Result<()> {
    let policy: serde_json::Value = serde_json::from_slice(&std::fs::read(
        std::path::Path::new(path).join("operating_point.json"),
    )?)?;
    ensure!(
        policy["max_input_tokens"] == 8192
            && policy["token_threshold"] == 0.5
            && policy["threshold_comparison"] == "strictly_greater"
            && policy["label2id"]["supported"] == 0
            && policy["label2id"]["hallucinated"] == 1
            && policy["input_pair"]
                == serde_json::json!(["User request: {question}\n\n{context}", "answer"])
            && policy["answer_offsets"] == "Unicode code points",
        "capability: unsupported Vela Halu operating point"
    );
    ensure!(
        config["id2label"]["0"] == "supported"
            && config["id2label"]["1"] == "hallucinated"
            && config["label2id"]["supported"] == 0
            && config["label2id"]["hallucinated"] == 1,
        "capability: Vela Halu labels must be supported=0, hallucinated=1"
    );
    Ok(())
}

fn encode_pair(
    tokenizer: &Tokenizer,
    prompt: &str,
    answer: &str,
    limit: usize,
    truncate: bool,
) -> Result<(Encoding, InputMetadata)> {
    let mut tokenizer = tokenizer.clone();
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow!(e.to_string()))?;
    tokenizer.with_padding(None);
    let original = tokenizer
        .encode((prompt, answer), true)
        .map_err(|e| anyhow!(e.to_string()))?;
    let original_tokens = original.len();
    let encoding = if original_tokens > limit {
        ensure!(
            truncate,
            "input_limit: grounded pair has {original_tokens} tokens, limit is {limit}"
        );
        let special = tokenizer
            .get_post_processor()
            .map_or(0, |p| p.added_tokens(true));
        ensure!(
            limit >= special,
            "input_limit: pair template exceeds token budget"
        );
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: limit,
                strategy: TruncationStrategy::OnlyFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|e| anyhow!("input_limit: {e}"))?;
        tokenizer
            .encode((prompt, answer), true)
            .map_err(|e| anyhow!("input_limit: complete answer does not fit grounded pair: {e}"))?
    } else {
        original
    };
    let input = InputMetadata {
        input_tokens: original_tokens,
        processed_tokens: encoding.len(),
        truncated: encoding.len() < original_tokens,
    };
    Ok((encoding, input))
}

fn answer_spans(
    encoding: &Encoding,
    answer: &str,
    probabilities: &[Vec<f32>],
) -> Result<Vec<Span>> {
    ensure!(
        probabilities.len() == encoding.len(),
        "result_invalid: grounding token count mismatch"
    );
    let mut spans = Vec::new();
    let mut current: Option<Span> = None;
    for ((sequence, &(start, end)), scores) in encoding
        .get_sequence_ids()
        .iter()
        .zip(encoding.get_offsets())
        .zip(probabilities)
    {
        ensure!(
            scores.len() == 2 && scores.iter().all(|s| s.is_finite()),
            "result_invalid: invalid grounding probabilities"
        );
        if *sequence != Some(1) || end <= start {
            continue;
        }
        // Rust tokenizers emits UTF-8 byte offsets. Python's fast tokenizer
        // exposes code-point offsets; both refer to the exact same answer text.
        ensure!(
            answer.get(start..end).is_some(),
            "result_invalid: invalid answer UTF-8 offsets"
        );
        if scores[1] > 0.5 {
            if let Some(span) = current.as_mut() {
                span.end = span.end.max(end);
                span.confidence = span.confidence.max(scores[1]);
            } else {
                current = Some(Span {
                    text: String::new(),
                    start,
                    end,
                    confidence: scores[1],
                    label: "HALLUCINATED".into(),
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
    Ok(spans)
}

pub(super) fn detect(
    instance: &Instance,
    model: &TraditionalModernBertTokenClassifier,
    context: &str,
    question: &str,
    answer: &str,
) -> Result<HallucinationOutput> {
    let prompt = format!("User request: {question}\n\n{context}");
    let tokenizer = instance
        .tokenizer
        .as_ref()
        .ok_or_else(|| anyhow!("capability: grounding tokenizer missing"))?;
    let (encoding, input) = encode_pair(
        tokenizer,
        &prompt,
        answer,
        instance.info.max_input_tokens,
        instance.info.overflow == "truncate",
    )?;
    let probabilities = model.classify_encoded_tokens(&encoding)?;
    let spans = answer_spans(&encoding, answer, &probabilities)?;
    let confidence = spans.iter().map(|s| s.confidence).fold(0.0, f32::max);
    Ok(HallucinationOutput {
        has_hallucination: !spans.is_empty(),
        confidence,
        spans,
        input,
        offset_unit: "utf8_bytes",
    })
}
