//! Length-control contract for embedding tokenizers.
//!
//! Embedding callers must observe the model's real context window. Tokenizer
//! JSON files may carry truncation or fixed-padding settings, so initialization
//! removes those controls exactly once. Request-time encoding then asserts the
//! invariant before producing the token IDs used by model tensors.

use std::fmt;

use tokenizers::{Encoding, Tokenizer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingTokenError {
    InputTooLong {
        model: String,
        count: usize,
        maximum: usize,
    },
    EmptyEncoding {
        model: String,
    },
    Configuration {
        model: String,
        detail: String,
    },
    Tokenization {
        model: String,
        detail: String,
    },
}

impl EmbeddingTokenError {
    pub(crate) fn is_input_too_long(&self) -> bool {
        matches!(self, Self::InputTooLong { .. })
    }
}

impl fmt::Display for EmbeddingTokenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLong {
                model,
                count,
                maximum,
            } => write!(
                formatter,
                "{model} input has {count} tokens; maximum context is {maximum}"
            ),
            Self::EmptyEncoding { model } => {
                write!(formatter, "{model} tokenizer produced no tokens")
            }
            Self::Configuration { model, detail } => {
                write!(formatter, "{model} tokenizer configuration error: {detail}")
            }
            Self::Tokenization { model, detail } => {
                write!(formatter, "{model} tokenization failed: {detail}")
            }
        }
    }
}

impl std::error::Error for EmbeddingTokenError {}

pub(crate) fn prepare_embedding_tokenizer(
    mut tokenizer: Tokenizer,
    model: &str,
) -> Result<Tokenizer, EmbeddingTokenError> {
    tokenizer
        .with_truncation(None)
        .map_err(|error| EmbeddingTokenError::Configuration {
            model: model.to_string(),
            detail: format!("failed to disable truncation: {error:?}"),
        })?;
    tokenizer.with_padding(None);
    assert_embedding_tokenizer_prepared(&tokenizer, model)?;
    Ok(tokenizer)
}

pub(crate) fn assert_embedding_tokenizer_prepared(
    tokenizer: &Tokenizer,
    model: &str,
) -> Result<(), EmbeddingTokenError> {
    if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
        return Err(EmbeddingTokenError::Configuration {
            model: model.to_string(),
            detail: "truncation or padding controls changed after initialization".to_string(),
        });
    }
    Ok(())
}

pub(crate) fn encode_embedding_checked(
    tokenizer: &Tokenizer,
    text: &str,
    maximum: usize,
    model: &str,
) -> Result<Encoding, EmbeddingTokenError> {
    assert_embedding_tokenizer_prepared(tokenizer, model)?;
    let encoding =
        tokenizer
            .encode(text, true)
            .map_err(|error| EmbeddingTokenError::Tokenization {
                model: model.to_string(),
                detail: format!("{error:?}"),
            })?;
    validate_embedding_token_count(encoding.get_ids().len(), maximum, model)?;
    Ok(encoding)
}

pub(crate) fn encode_embedding_batch_checked(
    tokenizer: &Tokenizer,
    texts: &[&str],
    maximum: usize,
    model: &str,
) -> Result<Vec<Encoding>, EmbeddingTokenError> {
    assert_embedding_tokenizer_prepared(tokenizer, model)?;
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|error| EmbeddingTokenError::Tokenization {
            model: model.to_string(),
            detail: format!("batch: {error:?}"),
        })?;
    for (index, encoding) in encodings.iter().enumerate() {
        validate_embedding_token_count(encoding.get_ids().len(), maximum, model).map_err(
            |error| match error {
                EmbeddingTokenError::InputTooLong {
                    model,
                    count,
                    maximum,
                } => EmbeddingTokenError::InputTooLong {
                    model: format!("texts[{index}] {model}"),
                    count,
                    maximum,
                },
                other => other,
            },
        )?;
    }
    Ok(encodings)
}

pub(crate) fn validate_embedding_token_count(
    count: usize,
    maximum: usize,
    model: &str,
) -> Result<(), EmbeddingTokenError> {
    if count == 0 {
        return Err(EmbeddingTokenError::EmptyEncoding {
            model: model.to_string(),
        });
    }
    if count > maximum {
        return Err(EmbeddingTokenError::InputTooLong {
            model: model.to_string(),
            count,
            maximum,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{
        models::wordlevel::WordLevel, PaddingParams, PaddingStrategy, TruncationParams,
    };

    fn test_tokenizer() -> Tokenizer {
        let model = WordLevel::builder()
            .vocab(
                [("[UNK]".to_string(), 0), ("request".to_string(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .expect("build word-level model");
        Tokenizer::new(model)
    }

    #[test]
    fn preparation_removes_length_controls_once_and_request_encoding_keeps_them_off() {
        let mut tokenizer = test_tokenizer();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 1,
                ..Default::default()
            }))
            .expect("configure truncation");
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            ..Default::default()
        }));

        let prepared = prepare_embedding_tokenizer(tokenizer, "test").expect("prepare tokenizer");
        for _ in 0..32 {
            let encoding =
                encode_embedding_checked(&prepared, "request", 8, "test").expect("encode");
            assert_eq!(encoding.get_ids(), &[1]);
        }

        assert!(prepared.get_truncation().is_none());
        assert!(prepared.get_padding().is_none());
    }

    #[test]
    fn request_encoding_fails_closed_if_controls_drift() {
        let mut tokenizer = test_tokenizer();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 1,
                ..Default::default()
            }))
            .expect("configure truncation");

        let error = encode_embedding_checked(&tokenizer, "request", 8, "test").unwrap_err();

        assert!(matches!(error, EmbeddingTokenError::Configuration { .. }));
    }

    #[test]
    fn token_count_boundaries_preserve_typed_input_too_long_error() {
        assert!(validate_embedding_token_count(512, 512, "multimodal").is_ok());
        let error = validate_embedding_token_count(513, 512, "multimodal").unwrap_err();
        assert!(error.is_input_too_long());
        assert!(matches!(
            error,
            EmbeddingTokenError::InputTooLong {
                count: 513,
                maximum: 512,
                ..
            }
        ));
    }
}
