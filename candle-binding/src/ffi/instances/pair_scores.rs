//! Pair scoring has its own input and output contract: complete native pairs
//! and raw relevance scores, without classification or embedding transforms.
use super::tasks::InputMetadata;
use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TextPair {
    query: String,
    document: String,
}

#[derive(Serialize)]
pub(super) struct PairScores {
    scores: Vec<f32>,
    inputs: Vec<InputMetadata>,
}

impl Instance {
    pub(super) fn score_pairs(&self, pairs: Vec<TextPair>) -> Result<PairScores> {
        let Model::Reranker(model) = &self.model else {
            bail!("capability: handle is not a pair scorer")
        };
        ensure!(
            !pairs.is_empty(),
            "configuration: pair scoring requires at least one pair"
        );
        // Validate the entire request before starting expensive native work.
        let encodings = pairs
            .iter()
            .map(|pair| {
                ensure!(
                    !pair.query.trim().is_empty() && !pair.document.trim().is_empty(),
                    "configuration: query and document must be nonempty"
                );
                let encoded = self
                    .tokenizer()?
                    .encode((pair.query.as_str(), pair.document.as_str()), true)
                    .map_err(|error| anyhow!(error.to_string()))?;
                ensure!(
                    encoded.len() <= self.info.max_input_tokens,
                    "input_limit: pair has {} tokens, task budget is {}",
                    encoded.len(),
                    self.info.max_input_tokens
                );
                Ok(encoded)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scores = Vec::with_capacity(encodings.len());
        let mut inputs = Vec::with_capacity(encodings.len());
        for encoded in encodings {
            scores.push(model.score_tokens(encoded.get_ids())?);
            inputs.push(InputMetadata {
                input_tokens: encoded.len(),
                processed_tokens: encoded.len(),
                truncated: false,
            });
        }
        Ok(PairScores { scores, inputs })
    }
}
