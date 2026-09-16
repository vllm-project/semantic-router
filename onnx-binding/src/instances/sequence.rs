//! Complete categorical and independent label results from owned sessions.
use super::*;

pub(super) fn validate_artifact(path: &str, independent: bool) -> UnifiedResult<()> {
    let file = std::path::Path::new(path).join("config.json");
    let raw: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&file).map_err(|e| errors::model_load(path, &e.to_string()))?,
    )
    .map_err(|e| errors::config_error("config", &e.to_string()))?;
    let actual = match raw.get("problem_type") {
        None | Some(serde_json::Value::Null) => "single_label_classification",
        Some(serde_json::Value::String(value)) => value.as_str(),
        Some(_) => {
            return Err(errors::config_error(
                "problem_type",
                "must be a string or null",
            ))
        }
    };
    let expected = if independent {
        "multi_label_classification"
    } else {
        "single_label_classification"
    };
    if actual != expected {
        return Err(errors::config_error(
            "problem_type",
            &format!("task requires {expected}, got {actual}"),
        ));
    }
    Ok(())
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

#[derive(Debug, Serialize)]
pub struct LabelScores {
    pub scores: Vec<f32>,
    pub labels: Vec<String>,
    pub input: InputUsage,
}

#[derive(Debug, Serialize)]
pub struct ClassificationWindow {
    pub start: usize,
    pub end: usize,
    pub probabilities: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct LabelScoreWindow {
    pub start: usize,
    pub end: usize,
    pub scores: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct WindowOutput<T> {
    pub windows: Vec<T>,
    pub labels: Vec<String>,
    pub content_tokens: usize,
    pub input: InputUsage,
}

fn validate(scores: &[f32], labels: &[String], categorical: bool) -> UnifiedResult<()> {
    if scores.is_empty()
        || scores.len() != labels.len()
        || scores
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || (categorical && (scores.iter().sum::<f32>() - 1.0).abs() > 1e-4)
    {
        return Err(errors::inference_error(
            "distribution",
            "invalid complete label scores",
        ));
    }
    Ok(())
}

fn input(instance: &Instance, text: &str) -> UnifiedResult<(Vec<u32>, InputUsage)> {
    let usage = instance.input(text)?;
    let mut tokenizer = instance.tokenizer.clone();
    if usage.truncated {
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: instance.effective_limit,
                ..Default::default()
            }))
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
    }
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| errors::tokenization_error(&e.to_string()))?;
    Ok((
        encoding.get_ids().to_vec(),
        InputUsage {
            processed_tokens: encoding.len(),
            ..usage
        },
    ))
}

pub fn classify(handle: u64, text: &str) -> UnifiedResult<Distribution> {
    let instance = get(handle)?;
    let mut model = instance.model.lock();
    let Model::Sequence(model) = &mut *model else {
        return Err(instance.wrong_task("sequence_classification"));
    };
    let (ids, input) = input(&instance, text)?;
    let result = model.classify_tokens_with_activation(&ids, false)?;
    validate(&result.probabilities, &instance.labels, true)?;
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

pub fn score(handle: u64, text: &str) -> UnifiedResult<LabelScores> {
    let instance = get(handle)?;
    let mut model = instance.model.lock();
    let Model::LabelScores(model) = &mut *model else {
        return Err(instance.wrong_task("label_scores"));
    };
    let (ids, input) = input(&instance, text)?;
    let result = model.classify_tokens_with_activation(&ids, true)?;
    validate(&result.probabilities, &instance.labels, false)?;
    instance.completed();
    Ok(LabelScores {
        scores: result.probabilities,
        labels: instance.labels.clone(),
        input,
    })
}

fn windows<T>(
    instance: &Instance,
    model: &mut MmBertSequenceClassifier,
    text: &str,
    size: usize,
    overlap: usize,
    categorical: bool,
    wrap: impl Fn(usize, usize, Vec<f32>) -> T,
) -> UnifiedResult<WindowOutput<T>> {
    let input = instance.input(text)?;
    if input.truncated {
        return Err(errors::validation(
            "input_tokens",
            &format!("at most {}", instance.effective_limit),
            &input.original_tokens.to_string(),
        ));
    }
    let plan = crate::core::sequence_windows::encode_windows(
        &instance.tokenizer,
        text,
        instance.effective_limit,
        size,
        overlap,
    )
    .map_err(|e| errors::config_error("window", &e))?;
    let content_tokens = plan.last().expect("nonempty plan validated").end;
    let mut windows = Vec::with_capacity(plan.len());
    for window in plan {
        let result = model.classify_tokens_with_activation(&window.ids, !categorical)?;
        validate(&result.probabilities, &instance.labels, categorical)?;
        // Count each completed native forward, including successful prefixes
        // when a later window fails. No partial result is returned as success.
        instance.completed();
        windows.push(wrap(window.start, window.end, result.probabilities));
    }
    Ok(WindowOutput {
        windows,
        labels: instance.labels.clone(),
        content_tokens,
        input,
    })
}

pub fn classify_windows(
    handle: u64,
    text: &str,
    size: usize,
    overlap: usize,
) -> UnifiedResult<WindowOutput<ClassificationWindow>> {
    let instance = get(handle)?;
    let mut model = instance.model.lock();
    let Model::Sequence(model) = &mut *model else {
        return Err(instance.wrong_task("sequence_classification"));
    };
    windows(
        &instance,
        model,
        text,
        size,
        overlap,
        true,
        |start, end, probabilities| ClassificationWindow {
            start,
            end,
            probabilities,
        },
    )
}

pub fn score_windows(
    handle: u64,
    text: &str,
    size: usize,
    overlap: usize,
) -> UnifiedResult<WindowOutput<LabelScoreWindow>> {
    let instance = get(handle)?;
    let mut model = instance.model.lock();
    let Model::LabelScores(model) = &mut *model else {
        return Err(instance.wrong_task("label_scores"));
    };
    windows(
        &instance,
        model,
        text,
        size,
        overlap,
        false,
        |start, end, scores| LabelScoreWindow { start, end, scores },
    )
}
