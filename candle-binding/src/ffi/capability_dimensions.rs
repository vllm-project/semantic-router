//! Observed dimensions sourced from the embedding model implementation.
//!
//! No dimension facts live in the FFI inventory. The native model owns them.

#[derive(Debug, PartialEq, Eq)]
pub(super) struct EmbeddingDimensions {
    pub native: u32,
    pub supported: Box<[u32]>,
}

impl EmbeddingDimensions {
    pub fn from_model(native: usize, declared: &[usize]) -> Result<Self, ()> {
        let native = positive_dimension(native)?;
        let mut supported = vec![native];
        for &dimension in declared {
            let dimension = positive_dimension(dimension)?;
            if !supported.contains(&dimension) {
                supported.push(dimension);
            }
        }
        if supported.len() > 1024 {
            return Err(());
        }
        Ok(Self {
            native,
            supported: supported.into_boxed_slice(),
        })
    }
}

fn positive_dimension(value: usize) -> Result<u32, ()> {
    let value = u32::try_from(value).map_err(|_| ())?;
    if value == 0 {
        return Err(());
    }
    Ok(value)
}

use super::capabilities::{
    MODEL_TYPE_GEMMA, MODEL_TYPE_MMBERT, MODEL_TYPE_MULTIMODAL, MODEL_TYPE_QWEN3,
};
use super::embedding::{get_batched_qwen3_dimensions, get_multimodal_refs, GLOBAL_MODEL_FACTORY};
use crate::model_architectures::embedding::gemma_embedding::SUPPORTED_EMBEDDING_DIMENSIONS;
use crate::model_architectures::traits::LongContextEmbeddingCapable;

pub(super) fn for_model(model_type: u32) -> Result<Option<EmbeddingDimensions>, ()> {
    let Some((native, declared)) = loaded_metadata(model_type) else {
        return Ok(None);
    };
    EmbeddingDimensions::from_model(native, &declared).map(Some)
}

fn loaded_metadata(model_type: u32) -> Option<(usize, Vec<usize>)> {
    if model_type == MODEL_TYPE_MULTIMODAL {
        let (model, _) = get_multimodal_refs()?;
        return Some((
            model.get_embedding_dimension(),
            model.get_matryoshka_dimensions(),
        ));
    }
    if model_type == MODEL_TYPE_QWEN3 {
        return GLOBAL_MODEL_FACTORY
            .get()
            .and_then(|factory| factory.get_qwen3_model())
            .map(|model| {
                (
                    model.get_embedding_dimension(),
                    model.get_matryoshka_dimensions(),
                )
            })
            .or_else(get_batched_qwen3_dimensions);
    }
    let factory = GLOBAL_MODEL_FACTORY.get()?;
    match model_type {
        MODEL_TYPE_GEMMA => {
            let model = factory.get_gemma_model()?;
            Some((
                model.config().hidden_size,
                SUPPORTED_EMBEDDING_DIMENSIONS.to_vec(),
            ))
        }
        MODEL_TYPE_MMBERT => {
            let model = factory.get_mmbert_model()?;
            Some((
                model.get_embedding_dimension(),
                model.get_matryoshka_dimensions(),
            ))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_declared_dimensions_and_native_width() {
        let dimensions = EmbeddingDimensions::from_model(960, &[320, 640, 320]).unwrap();
        assert_eq!(dimensions.native, 960);
        assert_eq!(&*dimensions.supported, &[960, 320, 640]);
        let native_only = EmbeddingDimensions::from_model(960, &[]).unwrap();
        assert_eq!(&*native_only.supported, &[960]);
    }

    #[test]
    fn rejects_invalid_model_metadata() {
        assert!(EmbeddingDimensions::from_model(0, &[320]).is_err());
        assert!(EmbeddingDimensions::from_model(960, &[0]).is_err());
        let oversized: Vec<usize> = (1..=1025).collect();
        assert!(EmbeddingDimensions::from_model(1, &oversized).is_err());
        if usize::BITS > 32 {
            assert!(EmbeddingDimensions::from_model(usize::MAX, &[]).is_err());
            assert!(EmbeddingDimensions::from_model(960, &[usize::MAX]).is_err());
        }
    }
}
