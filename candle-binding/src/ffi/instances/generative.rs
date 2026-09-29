//! Owned existing generative adapters. Mutable KV state is serialized per
//! physical instance; no request shares it concurrently with another binding.
use super::tasks::InputMetadata;
use super::*;

#[derive(Serialize)]
pub(super) struct GuardOutput {
    raw_output: String,
    input: InputMetadata,
}

#[derive(Serialize)]
pub(super) struct GenerativeOutput {
    class: usize,
    confidence: f32,
    probabilities: Vec<f32>,
    labels: Vec<String>,
    input: InputMetadata,
    score_semantics: &'static str,
    // Existing Qwen3MultiLoRAClassifier loads adapter metadata but its forward
    // currently uses the base LM only. Never report those deltas as applied.
    adapter_weights_applied: bool,
}

impl Instance {
    pub(super) fn guard(&self, text: &str, mode: &str) -> Result<GuardOutput> {
        ensure!(self.info.task == "guard", "capability: wrong task handle");
        ensure!(
            matches!(mode, "input" | "output"),
            "configuration: guard mode must be input or output"
        );
        let Model::Guard(model) = &self.model else {
            bail!("capability: not a guard model")
        };
        let mut model = model
            .lock()
            .map_err(|_| anyhow!("execution: guard state poisoned"))?;
        let tokens = model
            .input_token_count(text, mode)
            .map_err(|e| anyhow!("configuration: {e}"))?;
        let total = tokens
            .checked_add(model.output_token_budget())
            .ok_or_else(|| anyhow!("input_limit: token budget overflow"))?;
        ensure!(tokens <= self.info.max_input_tokens && total <= self.info.architectural_max_tokens, "input_limit: guard template, input and output reserve exceed the deployment/model budget");
        let output = model.generate_guard(text, mode)?;
        Ok(GuardOutput {
            raw_output: output.raw_output,
            input: InputMetadata {
                input_tokens: tokens,
                processed_tokens: tokens,
                truncated: false,
            },
        })
    }

    pub(super) fn generative(
        &self,
        text: &str,
        adapter: Option<&str>,
        categories: Vec<String>,
        multi_token: bool,
    ) -> Result<GenerativeOutput> {
        ensure!(
            self.info.task == "generative",
            "capability: wrong task handle"
        );
        let Model::Generative(model) = &self.model else {
            bail!("capability: not a generative classifier")
        };
        let mut model = model
            .lock()
            .map_err(|_| anyhow!("execution: generative model state poisoned"))?;
        let labels = if let Some(adapter) = adapter {
            model
                .get_adapter_categories(adapter)
                .ok_or_else(|| anyhow!("configuration: adapter is not bound to this instance"))?
        } else {
            categories
        };
        ensure!(
            !labels.is_empty() && labels.iter().all(|label| !label.is_empty()),
            "configuration: categories must be nonempty"
        );
        let mut unique = std::collections::HashSet::new();
        ensure!(
            labels.iter().all(|label| unique.insert(label)),
            "configuration: duplicate categories"
        );
        let tokens = model
            .input_token_count(text, adapter, &labels)
            .map_err(|e| anyhow!("configuration: {e}"))?;
        let lengths = model
            .category_token_lengths(&labels)
            .map_err(|e| anyhow!("configuration: {e}"))?;
        ensure!(
            lengths.iter().all(|length| *length > 0),
            "configuration: empty tokenized category"
        );
        let reserve = if multi_token {
            *lengths.iter().max().unwrap_or(&0)
        } else {
            0
        };
        ensure!(
            tokens <= self.info.max_input_tokens
                && tokens
                    .checked_add(reserve)
                    .is_some_and(|n| n <= self.info.architectural_max_tokens),
            "input_limit: classification template, candidates and input exceed the task budget"
        );
        let result = if let Some(adapter) = adapter {
            ensure!(
                !multi_token,
                "capability: adapter classification uses existing first-token scoring"
            );
            model.classify_with_adapter(text, adapter)?
        } else if multi_token {
            model.classify_zero_shot_multi_tokens(text, labels)?
        } else {
            model.classify_zero_shot(text, labels)?
        };
        let class = result
            .all_categories
            .iter()
            .position(|label| label == &result.category)
            .ok_or_else(|| {
                anyhow!("result_invalid: generated category is outside the candidate set")
            })?;
        ensure!(
            result.probabilities.len() == result.all_categories.len()
                && result
                    .probabilities
                    .iter()
                    .all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0),
            "result_invalid: invalid conditional label probabilities"
        );
        ensure!(
            (result.probabilities.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
            "result_invalid: conditional probabilities do not sum to one"
        );
        Ok(GenerativeOutput {
            class,
            confidence: result.confidence,
            probabilities: result.probabilities,
            labels: result.all_categories,
            input: InputMetadata {
                input_tokens: tokens,
                processed_tokens: tokens,
                truncated: false,
            },
            score_semantics: if multi_token {
                "label_sequence_softmax"
            } else {
                "first_token_label_softmax"
            },
            adapter_weights_applied: false,
        })
    }
}
