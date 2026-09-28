use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextPair {
    query: String,
    document: String,
}

#[derive(Serialize)]
pub struct PairScores {
    scores: Vec<f32>,
    inputs: Vec<InputUsage>,
}

pub fn load_pair_scorer(
    options: InstanceOptions,
    selection: PairScorerSelection,
) -> UnifiedResult<u64> {
    let options = fresh_options(options);
    let model = PairScorer::load(&options, selection)?;
    prepare(Model::PairScorer(Box::new(model)), options)
}

pub fn score_pairs(handle: u64, pairs: Vec<TextPair>) -> UnifiedResult<PairScores> {
    let instance = get(handle)?;
    if instance.task != "pair_scores" {
        return Err(instance.wrong_task("pair_scores"));
    }
    if pairs.is_empty() {
        return Err(errors::config_error(
            "pairs",
            "at least one pair is required",
        ));
    }
    let encodings = pairs
        .iter()
        .map(|pair| {
            if pair.query.trim().is_empty() || pair.document.trim().is_empty() {
                return Err(errors::config_error(
                    "pairs",
                    "query and document must be nonempty",
                ));
            }
            let encoded = instance
                .tokenizer
                .encode((pair.query.as_str(), pair.document.as_str()), true)
                .map_err(|error| errors::tokenization_error(&error.to_string()))?;
            if encoded.len() > instance.effective_limit {
                return Err(errors::validation(
                    "input_tokens",
                    &format!("at most {}", instance.effective_limit),
                    &encoded.len().to_string(),
                ));
            }
            Ok(encoded)
        })
        .collect::<UnifiedResult<Vec<_>>>()?;
    let mut model = instance.model.lock();
    let Model::PairScorer(model) = &mut *model else {
        return Err(instance.wrong_task("pair_scores"));
    };
    let mut scores = Vec::with_capacity(encodings.len());
    let mut inputs = Vec::with_capacity(encodings.len());
    for encoded in encodings {
        scores.push(model.score_tokens(encoded.get_ids())?);
        instance.completed.fetch_add(1, Ordering::Relaxed);
        inputs.push(InputUsage {
            original_tokens: encoded.len(),
            processed_tokens: encoded.len(),
            truncated: false,
        });
    }
    Ok(PairScores { scores, inputs })
}
