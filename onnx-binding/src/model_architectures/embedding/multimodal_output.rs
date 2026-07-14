//! Output contract for the exported multimodal ONNX encoders.
//!
//! Each encoder artifact is responsible for producing its final pooled vector
//! in the shared embedding space. In particular, the image encoder's native
//! hidden size is projected to `embedding_dim` inside the exported artifact.
//! Sequence outputs are therefore not valid embeddings at this boundary: the
//! correct pooling operation is modality-specific and must not be guessed by
//! the runtime.

use crate::core::unified_error::{errors, UnifiedError, UnifiedResult};
use ndarray::Array1;
use ort::session::SessionOutputs;
use ort::tensor::Shape;

const OUTPUT_SHAPE_FIELD: &str = "multimodal_embedding_output_shape";

/// Extract the final pooled embedding from the runtime outputs. Known pooled
/// names take precedence. A non-standard name is accepted only for a single,
/// unambiguous output; it still has to satisfy the same shape contract.
pub(super) fn extract_embedding_from_outputs(
    outputs: &SessionOutputs,
    expected_embedding_dim: usize,
) -> UnifiedResult<Array1<f32>> {
    let names = ["embedding", "sentence_embedding", "pooler_output"];
    for name in &names {
        let Some(output_value) = outputs.get(*name) else {
            continue;
        };
        let (shape, data) = output_value.try_extract_tensor::<f32>().map_err(|e| {
            errors::inference_error(
                "extract_multimodal_embedding",
                &format!("output '{name}' is not an f32 tensor: {e}"),
            )
        })?;

        let dims = concrete_dimensions(shape)?;
        return extract_pooled_embedding(&dims, data, expected_embedding_dim);
    }

    let mut output_iter = outputs.iter();
    let Some((output_name, output_value)) = output_iter.next() else {
        return Err(errors::inference_error(
            "extract_multimodal_embedding",
            "model returned no outputs",
        ));
    };
    if output_iter.next().is_some() {
        return Err(errors::inference_error(
            "extract_multimodal_embedding",
            "model returned multiple outputs without a recognized pooled embedding name",
        ));
    }

    let (shape, data) = output_value.try_extract_tensor::<f32>().map_err(|e| {
        errors::inference_error(
            "extract_multimodal_embedding",
            &format!("output '{output_name}' is not an f32 tensor: {e}"),
        )
    })?;
    let dims = concrete_dimensions(shape)?;
    extract_pooled_embedding(&dims, data, expected_embedding_dim)
}

fn concrete_dimensions(shape: &Shape) -> UnifiedResult<Vec<usize>> {
    shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| UnifiedError::Validation {
                field: OUTPUT_SHAPE_FIELD.to_string(),
                expected: "concrete non-negative dimensions".to_string(),
                actual: format!("{shape:?}"),
            })
        })
        .collect()
}

/// Extract one final pooled embedding while enforcing the exported-model
/// contract. A single-vector export may omit the batch axis (`[hidden]`), but a
/// present batch axis must be exactly one (`[1, hidden]`).
pub(super) fn extract_pooled_embedding(
    dims: &[usize],
    data: &[f32],
    expected_embedding_dim: usize,
) -> UnifiedResult<Array1<f32>> {
    let valid_shape = match dims {
        [hidden] => expected_embedding_dim > 0 && *hidden == expected_embedding_dim,
        [batch, hidden] => {
            expected_embedding_dim > 0 && *batch == 1 && *hidden == expected_embedding_dim
        }
        _ => false,
    };

    if !valid_shape {
        return Err(UnifiedError::Validation {
            field: OUTPUT_SHAPE_FIELD.to_string(),
            expected: format!(
                "a final pooled embedding shaped [{expected_embedding_dim}] or [1, {expected_embedding_dim}]"
            ),
            actual: format!("{dims:?}"),
        });
    }

    if data.len() != expected_embedding_dim {
        return Err(UnifiedError::Validation {
            field: OUTPUT_SHAPE_FIELD.to_string(),
            expected: format!("exactly {expected_embedding_dim} tensor values"),
            actual: format!("{} tensor values", data.len()),
        });
    }

    Ok(Array1::from_vec(data.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 384;

    #[test]
    fn accepts_only_single_pooled_vectors_at_the_configured_dimension() {
        let data = vec![0.25; DIM];

        let one_dimensional = extract_pooled_embedding(&[DIM], &data, DIM).unwrap();
        let batched = extract_pooled_embedding(&[1, DIM], &data, DIM).unwrap();

        assert_eq!(one_dimensional.len(), DIM);
        assert_eq!(batched.len(), DIM);
    }

    #[test]
    fn rejects_sequence_unpooled_and_multi_batch_outputs() {
        let data = vec![0.25; DIM];
        for invalid in [
            vec![2, DIM],
            vec![1, 8, DIM],
            vec![8, DIM],
            vec![1, 8, DIM, 1],
        ] {
            let error = extract_pooled_embedding(&invalid, &data, DIM)
                .expect_err("non-pooled output must fail closed");
            assert!(matches!(
                error,
                UnifiedError::Validation { ref field, .. }
                    if field == OUTPUT_SHAPE_FIELD
            ));
        }
    }

    #[test]
    fn rejects_empty_or_wrong_projection_dimensions() {
        let data = vec![0.25; DIM];
        for invalid in [vec![], vec![0], vec![1, 0], vec![768], vec![1, 768]] {
            assert!(extract_pooled_embedding(&invalid, &data, DIM).is_err());
        }
        assert!(extract_pooled_embedding(&[DIM], &data, 0).is_err());
    }

    #[test]
    fn rejects_shape_and_storage_cardinality_mismatch() {
        let short = vec![0.25; DIM - 1];
        let long = vec![0.25; DIM + 1];

        assert!(extract_pooled_embedding(&[DIM], &short, DIM).is_err());
        assert!(extract_pooled_embedding(&[1, DIM], &long, DIM).is_err());
    }

    #[test]
    fn rejects_symbolic_runtime_dimensions() {
        let error = concrete_dimensions(&Shape::new([1, -1]))
            .expect_err("runtime outputs must have concrete dimensions");
        assert!(matches!(
            error,
            UnifiedError::Validation { ref field, .. }
                if field == OUTPUT_SHAPE_FIELD
        ));
    }
}
