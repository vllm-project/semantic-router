//! Concrete input shapes bound to a prepared execution, independent of paths.
use crate::core::unified_error::{errors, UnifiedResult};
use ort::{session::Input, tensor::TensorElementType, value::ValueType};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Resolve named symbolic dimensions from the graph itself, never task-specific
/// symbol guesses. ORT applies these before partitioning/constant folding.
pub fn dimension_overrides(
    declared: &[super::onnx_artifacts::GraphInput],
    actual: &[ExecutionInput],
) -> anyhow::Result<BTreeMap<String, i64>> {
    validate_contract(actual)?;
    anyhow::ensure!(
        declared.len() == actual.len(),
        "input count differs from graph"
    );
    let mut overrides = BTreeMap::new();
    for input in declared {
        let resolved = actual
            .iter()
            .find(|item| item.name == input.name)
            .ok_or_else(|| anyhow::anyhow!("input name differs from graph"))?;
        anyhow::ensure!(
            input.dimensions.len() == resolved.shape.len()
                && input.dimension_symbols.len() == resolved.shape.len(),
            "input rank differs from graph"
        );
        for ((size, symbol), value) in input
            .dimensions
            .iter()
            .zip(&input.dimension_symbols)
            .zip(&resolved.shape)
        {
            if let Some(size) = size {
                anyhow::ensure!(size == value, "fixed input dimension differs from contract");
            } else {
                let symbol = symbol.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("fixed execution requires named dynamic dimensions")
                })?;
                if let Some(previous) = overrides.insert(symbol.clone(), *value) {
                    anyhow::ensure!(previous == *value, "conflicting symbolic input dimensions");
                }
            }
        }
    }
    Ok(overrides)
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionInput {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
}

/// Describe inputs from the session ORT actually loaded, independently of any
/// fixed execution contract. In particular, an empty execution contract does
/// not prove that a graph is dynamic: its declared dimensions are authoritative.
pub fn session_input_schema(inputs: &[Input]) -> Vec<ExecutionInput> {
    inputs
        .iter()
        .map(|input| {
            let (dtype, shape) = match &input.input_type {
                ValueType::Tensor { ty, shape, .. } => {
                    (format!("{ty:?}").to_ascii_lowercase(), shape.to_vec())
                }
                // Preserve unsupported kinds as explicit evidence, not as a
                // tensor with an inferred or fabricated dynamic shape.
                other => (format!("{other:?}"), Vec::new()),
            };
            ExecutionInput {
                name: input.name.clone(),
                dtype,
                shape,
            }
        })
        .collect()
}

pub fn validate_contract(inputs: &[ExecutionInput]) -> UnifiedResult<()> {
    let mut names = BTreeSet::new();
    if inputs.is_empty() {
        return Err(errors::config_error(
            "execution_inputs",
            "a concrete input contract is required",
        ));
    }
    for input in inputs {
        if input.name.is_empty()
            || !names.insert(&input.name)
            || !matches!(
                input.dtype.as_str(),
                "int64" | "int32" | "float32" | "float16" | "bool"
            )
            || input.shape.len() > 32
            || input.shape.iter().any(|&dimension| dimension <= 0)
        {
            return Err(errors::config_error(
                "execution_inputs",
                "invalid input name, dtype or concrete shape",
            ));
        }
    }
    Ok(())
}

pub fn validate_session(declared: &[Input], actual: &[ExecutionInput]) -> UnifiedResult<()> {
    validate_contract(actual)?;
    if declared.len() != actual.len() {
        return Err(errors::config_error(
            "execution_inputs",
            "input count differs from actual session",
        ));
    }
    for declaration in declared {
        let input = actual
            .iter()
            .find(|input| input.name == declaration.name)
            .ok_or_else(|| {
                errors::config_error("execution_inputs", "input name differs from actual session")
            })?;
        let ValueType::Tensor { ty, shape, .. } = &declaration.input_type else {
            return Err(errors::config_error(
                "execution_inputs",
                "cached execution requires tensor inputs",
            ));
        };
        let dtype = match ty {
            TensorElementType::Int64 => "int64",
            TensorElementType::Int32 => "int32",
            TensorElementType::Float32 => "float32",
            TensorElementType::Float16 => "float16",
            TensorElementType::Bool => "bool",
            _ => "unsupported",
        };
        if dtype != input.dtype
            || shape.len() != input.shape.len()
            || shape
                .iter()
                .zip(&input.shape)
                .any(|(&wanted, &value)| wanted != -1 && wanted != value)
        {
            return Err(errors::config_error(
                "execution_inputs",
                "dtype or shape differs from actual session",
            ));
        }
    }
    Ok(())
}
