//! Length-control contract shared by ONNX embedding tokenizers.
//!
//! Tokenizer JSON files may contain truncation or fixed-padding settings. Those
//! settings must not silently change request meaning or hide a context overflow,
//! so model initialization removes them and request-time encoding verifies the
//! invariant before validating the true token count.

use crate::core::unified_error::{errors, UnifiedError, UnifiedResult};
use tokenizers::{Encoding, Tokenizer};

pub(crate) fn prepare_embedding_tokenizer(
    mut tokenizer: Tokenizer,
    model: &str,
) -> UnifiedResult<Tokenizer> {
    tokenizer
        .with_truncation(None)
        .map_err(|error| errors::config_error(model, &format!("disable truncation: {error:?}")))?;
    tokenizer.with_padding(None);
    assert_embedding_tokenizer_prepared(&tokenizer, model)?;
    Ok(tokenizer)
}

fn assert_embedding_tokenizer_prepared(tokenizer: &Tokenizer, model: &str) -> UnifiedResult<()> {
    if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
        return Err(errors::config_error(
            model,
            "embedding tokenizer truncation or padding changed after initialization",
        ));
    }
    Ok(())
}

pub(crate) fn encode_embedding_checked(
    tokenizer: &Tokenizer,
    text: &str,
    maximum: usize,
    model: &str,
) -> UnifiedResult<Encoding> {
    assert_embedding_tokenizer_prepared(tokenizer, model)?;
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|error| errors::tokenization_error(&format!("{model}: {error:?}")))?;
    validate_embedding_token_count(encoding.len(), maximum, model)?;
    Ok(encoding)
}

pub(crate) fn encode_embedding_batch_checked(
    tokenizer: &Tokenizer,
    texts: &[&str],
    maximum: usize,
    model: &str,
) -> UnifiedResult<Vec<Encoding>> {
    assert_embedding_tokenizer_prepared(tokenizer, model)?;
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|error| errors::tokenization_error(&format!("{model} batch: {error:?}")))?;
    for (index, encoding) in encodings.iter().enumerate() {
        validate_embedding_token_count(encoding.len(), maximum, model).map_err(
            |error| match error {
                UnifiedError::InputTooLong {
                    model,
                    count,
                    maximum,
                } => UnifiedError::InputTooLong {
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
) -> UnifiedResult<()> {
    if count == 0 {
        return Err(errors::tokenization_error(&format!(
            "{model} tokenizer produced no tokens"
        )));
    }
    if count > maximum {
        return Err(errors::input_too_long(model, count, maximum));
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
    fn preparation_removes_serialized_length_controls() {
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
        assert!(prepared.get_truncation().is_none());
        assert!(prepared.get_padding().is_none());
        assert!(encode_embedding_checked(&prepared, "request", 1, "test").is_ok());
    }

    #[test]
    fn request_encoding_fails_if_length_controls_drift() {
        let mut tokenizer = test_tokenizer();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 1,
                ..Default::default()
            }))
            .expect("configure truncation");

        let error = encode_embedding_checked(&tokenizer, "request", 8, "test")
            .expect_err("unprepared tokenizer must fail closed");
        assert!(!error.is_input_too_long());
        assert!(matches!(error, UnifiedError::Config { .. }));
    }

    #[test]
    fn token_count_boundary_preserves_typed_overflow() {
        assert!(validate_embedding_token_count(512, 512, "multimodal").is_ok());

        let error = validate_embedding_token_count(513, 512, "multimodal")
            .expect_err("limit + 1 must fail");
        assert!(error.is_input_too_long());
        assert!(matches!(
            error,
            UnifiedError::InputTooLong {
                count: 513,
                maximum: 512,
                ..
            }
        ));
    }

    #[test]
    fn batch_error_identifies_the_oversized_item() {
        let tokenizer = prepare_embedding_tokenizer(test_tokenizer(), "test").unwrap();
        let error = encode_embedding_batch_checked(&tokenizer, &["request", "request"], 0, "test")
            .expect_err("zero context rejects the first encoded item");

        assert!(matches!(
            error,
            UnifiedError::InputTooLong { ref model, .. } if model == "texts[0] test"
        ));
    }
}
