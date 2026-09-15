//! Typed categorical and independent-label inference over the same owned model.
//! Window outputs retain all labels; task policy belongs to the caller.
use super::tasks::InputMetadata;
use super::*;

#[derive(Serialize)]
pub(super) struct Distribution {
    class: usize,
    confidence: f32,
    probabilities: Vec<f32>,
    labels: Vec<String>,
    input: InputMetadata,
}

#[derive(Serialize)]
pub(super) struct LabelScores {
    scores: Vec<f32>,
    labels: Vec<String>,
    input: InputMetadata,
}

#[derive(Serialize)]
pub(super) struct ClassificationWindow {
    start: usize,
    end: usize,
    probabilities: Vec<f32>,
}

#[derive(Serialize)]
pub(super) struct LabelScoreWindow {
    start: usize,
    end: usize,
    scores: Vec<f32>,
}

#[derive(Serialize)]
pub(super) struct WindowOutput<T> {
    windows: Vec<T>,
    labels: Vec<String>,
    content_tokens: usize,
    input: InputMetadata,
}

pub(super) fn validate_problem_type(raw: &serde_json::Value, task: &str) -> Result<()> {
    let expected = match task {
        "sequence" | "nli" => "single_label_classification",
        "label_scores" => "multi_label_classification",
        _ => return Ok(()),
    };
    let actual = match raw.get("problem_type") {
        None | Some(serde_json::Value::Null) => "single_label_classification",
        Some(serde_json::Value::String(value)) => value.as_str(),
        Some(_) => bail!("configuration: problem_type must be a string or null"),
    };
    ensure!(
        actual == expected,
        "configuration: {task} requires problem_type {expected}, got {actual}"
    );
    Ok(())
}

fn validate_scores(scores: &[f32], labels: &[String], categorical: bool) -> Result<()> {
    ensure!(
        !scores.is_empty()
            && scores.len() == labels.len()
            && scores
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "result_invalid: invalid complete label scores"
    );
    ensure!(
        !categorical || (scores.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
        "result_invalid: probability distribution does not sum to one"
    );
    Ok(())
}

impl Instance {
    fn sequence_model(&self) -> Result<&TraditionalModernBertClassifier> {
        match &self.model {
            Model::Sequence(model) => Ok(model),
            _ => bail!("capability: exact-token sequence execution requires a ModernBERT adapter"),
        }
    }

    // Encode at the boundary and pass those exact IDs to the model. Explicit
    // truncation uses the tokenizer's special-token template, never decoded text.
    fn sequence_input(&self, text: &str) -> Result<(Vec<u32>, InputMetadata)> {
        let mut encoding = self
            .tokenizer()?
            .encode(text, true)
            .map_err(|e| anyhow!(e.to_string()))?;
        let original = encoding.len();
        if original > self.info.max_input_tokens {
            ensure!(
                self.info.overflow == "truncate",
                "input_limit: input has {original} tokens, task budget is {}",
                self.info.max_input_tokens
            );
            let mut tokenizer = self.tokenizer()?.clone();
            tokenizer
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length: self.info.max_input_tokens,
                    ..Default::default()
                }))
                .map_err(|e| anyhow!(e.to_string()))?;
            encoding = tokenizer
                .encode(text, true)
                .map_err(|e| anyhow!(e.to_string()))?;
        }
        Ok((
            encoding.get_ids().to_vec(),
            InputMetadata {
                input_tokens: original,
                processed_tokens: encoding.len(),
                truncated: encoding.len() < original,
            },
        ))
    }

    fn distribution(&self, probabilities: Vec<f32>, input: InputMetadata) -> Result<Distribution> {
        validate_scores(&probabilities, &self.info.labels, true)?;
        let (class, confidence) = probabilities
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .expect("nonempty probabilities validated");
        Ok(Distribution {
            class,
            confidence,
            probabilities,
            labels: self.info.labels.clone(),
            input,
        })
    }

    pub(super) fn sequence_on_text(
        &self,
        text: &str,
        input: InputMetadata,
    ) -> Result<Distribution> {
        let (_, _, probabilities) = match &self.model {
            Model::Sequence(model) => model.classify_text_with_probabilities(text)?,
            Model::Bert(model) => model.classify_text_with_probabilities(text)?,
            Model::MergedBert(model) => model.classify_text_with_probabilities(text)?,
            Model::Deberta(model) => model.classify_text_with_probabilities(text)?,
            _ => bail!("capability: handle is not a sequence classifier"),
        };
        self.distribution(probabilities, input)
    }

    pub(super) fn sequence(&self, text: &str) -> Result<Distribution> {
        ensure!(
            self.info.task == "sequence",
            "capability: wrong task handle"
        );
        if matches!(self.model, Model::Sequence(_)) {
            let (ids, input) = self.sequence_input(text)?;
            let (_, _, probabilities) = self
                .sequence_model()?
                .classify_tokens_with_activation(&ids, false)?;
            self.distribution(probabilities, input)
        } else {
            let (text, input) = self.prepare_text(text)?;
            self.sequence_on_text(text, input)
        }
    }

    pub(super) fn score(&self, text: &str) -> Result<LabelScores> {
        ensure!(
            self.info.task == "label_scores",
            "capability: wrong task handle"
        );
        let (ids, input) = self.sequence_input(text)?;
        let (_, _, scores) = self
            .sequence_model()?
            .classify_tokens_with_activation(&ids, true)?;
        validate_scores(&scores, &self.info.labels, false)?;
        Ok(LabelScores {
            scores,
            labels: self.info.labels.clone(),
            input,
        })
    }

    fn windows<T>(
        &self,
        text: &str,
        size: usize,
        overlap: usize,
        categorical: bool,
        wrap: impl Fn(usize, usize, Vec<f32>) -> T,
    ) -> Result<WindowOutput<T>> {
        let model = self.sequence_model()?;
        let input_tokens = self.count_tokens(text)?;
        ensure!(
            input_tokens <= self.info.max_input_tokens,
            "input_limit: input has {input_tokens} tokens, task budget is {}",
            self.info.max_input_tokens
        );
        let planned = crate::core::sequence_windows::encode_windows(
            self.tokenizer()?,
            text,
            self.info.max_input_tokens,
            size,
            overlap,
        )
        .map_err(|error| anyhow!("configuration: {error}"))?;
        let content_tokens = planned.last().expect("nonempty plan validated").end;
        let mut windows = Vec::with_capacity(planned.len());
        for window in planned {
            let (_, _, scores) =
                model.classify_tokens_with_activation(&window.ids, !categorical)?;
            validate_scores(&scores, &self.info.labels, categorical)?;
            windows.push(wrap(window.start, window.end, scores));
        }
        Ok(WindowOutput {
            windows,
            labels: self.info.labels.clone(),
            content_tokens,
            input: InputMetadata {
                input_tokens,
                processed_tokens: input_tokens,
                truncated: false,
            },
        })
    }

    pub(super) fn classify_windows(
        &self,
        text: &str,
        size: usize,
        overlap: usize,
    ) -> Result<WindowOutput<ClassificationWindow>> {
        ensure!(
            self.info.task == "sequence",
            "capability: wrong task handle"
        );
        self.windows(text, size, overlap, true, |start, end, probabilities| {
            ClassificationWindow {
                start,
                end,
                probabilities,
            }
        })
    }

    pub(super) fn score_windows(
        &self,
        text: &str,
        size: usize,
        overlap: usize,
    ) -> Result<WindowOutput<LabelScoreWindow>> {
        ensure!(
            self.info.task == "label_scores",
            "capability: wrong task handle"
        );
        self.windows(text, size, overlap, false, |start, end, scores| {
            LabelScoreWindow { start, end, scores }
        })
    }
}
