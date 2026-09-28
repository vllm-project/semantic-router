//! Standard ModernBERT input adaptation, shared by every task and physical exit.
//! Absolute positions follow the actual execution length, including padding;
//! they never depend on token values, attended-token counts or window offsets.

use crate::core::unified_error::{errors, UnifiedResult};
use ort::{
    session::{Input, Session, SessionOutputs},
    tensor::TensorElementType,
    value::{Tensor, ValueType},
};
use std::collections::HashSet;

fn invalid(message: &str) -> crate::core::unified_error::UnifiedError {
    errors::config_error("modernbert_inputs", message)
}

fn dimensions(input: &Input) -> UnifiedResult<&[i64]> {
    match &input.input_type {
        ValueType::Tensor { ty, shape, .. }
            if *ty == TensorElementType::Int64
                && shape.len() == 2
                && shape.iter().all(|&n| n == -1 || n > 0) =>
        {
            Ok(shape.as_ref())
        }
        _ => Err(invalid("expected rank-two int64 tensor inputs")),
    }
}

/// Validate the actual session schema before exposing a loaded model.
pub(crate) fn validate(inputs: &[Input]) -> UnifiedResult<bool> {
    let mut names = HashSet::new();
    for input in inputs {
        if !matches!(
            input.name.as_str(),
            "input_ids" | "attention_mask" | "position_ids"
        ) || !names.insert(input.name.as_str())
        {
            return Err(invalid("unknown or duplicate required graph input"));
        }
        let shape = dimensions(input)?;
        if input.name == "position_ids" && shape[0] != 1 {
            return Err(invalid("position_ids must declare one broadcast row"));
        }
    }
    if !names.contains("input_ids") || !names.contains("attention_mask") {
        return Err(invalid("input_ids and attention_mask are required"));
    }
    let mut fixed = [None; 2];
    for input in inputs {
        let shape = dimensions(input)?;
        for axis in 0..2 {
            if axis == 0 && input.name == "position_ids" {
                continue;
            }
            if shape[axis] > 0 {
                if fixed[axis].is_some_and(|size| size != shape[axis]) {
                    return Err(invalid(
                        "contradictory declared token/mask/position dimensions",
                    ));
                }
                fixed[axis] = Some(shape[axis]);
            }
        }
    }
    Ok(names.contains("position_ids"))
}

/// Fixed dimensions constrain the complete feed, even when another input uses
/// a symbolic dimension. Position IDs share sequence but broadcast across batch.
pub(crate) fn fixed_dimensions(inputs: &[Input]) -> UnifiedResult<[Option<usize>; 2]> {
    validate(inputs)?;
    let mut fixed = [None; 2];
    for input in inputs {
        for (axis, &size) in dimensions(input)?.iter().enumerate() {
            if size > 0 && !(axis == 0 && input.name == "position_ids") {
                fixed[axis] = Some(
                    usize::try_from(size).map_err(|_| invalid("graph dimension exceeds usize"))?,
                );
            }
        }
    }
    Ok(fixed)
}

/// Resolve the actual graph's two/three token inputs before cached preparation.
pub(crate) fn resolved_inputs(
    path: &std::path::Path,
    batch: usize,
    sequence: usize,
) -> UnifiedResult<Vec<crate::core::execution_contract::ExecutionInput>> {
    use crate::core::{execution_contract::ExecutionInput, onnx_artifacts::input_schema};
    if batch == 0 || sequence == 0 || batch > i64::MAX as usize || sequence > i64::MAX as usize {
        return Err(invalid("invalid nonempty token execution shape"));
    }
    let declarations = input_schema(path).map_err(|error| invalid(&error.to_string()))?;
    let mut names = HashSet::new();
    let mut result = Vec::new();
    for input in declarations {
        if !matches!(
            input.name.as_str(),
            "input_ids" | "attention_mask" | "position_ids"
        ) || !names.insert(input.name.clone())
            || input.element_type != 7
            || input.dimensions.len() != 2
        {
            return Err(invalid("unknown, duplicate or non-int64 token input"));
        }
        if input.name == "position_ids" && input.dimensions[0] != Some(1) {
            return Err(invalid("position_ids must declare one broadcast row"));
        }
        let shape = vec![
            if input.name == "position_ids" {
                1
            } else {
                batch as i64
            },
            sequence as i64,
        ];
        if input
            .dimensions
            .iter()
            .zip(&shape)
            .any(|(declared, actual)| declared.is_some_and(|n| n != *actual))
        {
            return Err(invalid(
                "declared graph dimensions differ from execution shape",
            ));
        }
        result.push(ExecutionInput {
            name: input.name,
            dtype: "int64".into(),
            shape,
        });
    }
    if !names.contains("input_ids") || !names.contains("attention_mask") {
        return Err(invalid("input_ids and attention_mask are required"));
    }
    result.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(result)
}

/// Resolve explicit GPU execution budgets before graph partitioning. CPU
/// retains dynamic short inputs, and legacy ROCm callers remain dynamic.
pub(crate) fn prepare_session(
    options: &crate::core::instance_options::InstanceOptions,
    path: &std::path::Path,
    execution_limit: usize,
) -> UnifiedResult<crate::core::instance_options::PreparedSession> {
    use crate::core::instance_options::Provider;
    if options.provider == Provider::Rocm && options.execution_max_input_tokens.is_some() {
        let inputs = resolved_inputs(path, 1, execution_limit)?;
        options.create_session_with_fixed_contract(path, &inputs)
    } else if options.provider == Provider::Migraphx {
        let inputs = resolved_inputs(path, 1, execution_limit)?;
        options.create_session_with_contract(path, &inputs)
    } else {
        options.prepare_session(path)
    }
}

pub(crate) fn run(
    session: &mut Session,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    batch: usize,
    sequence: usize,
) -> UnifiedResult<SessionOutputs<'_>> {
    let positions = validate(&session.inputs)?;
    if batch == 0
        || sequence == 0
        || sequence > i64::MAX as usize
        || batch.checked_mul(sequence) != Some(input_ids.len())
        || input_ids.len() != attention_mask.len()
    {
        return Err(invalid("invalid nonempty token/mask execution shape"));
    }
    for input in &session.inputs {
        let wanted = dimensions(input)?;
        let actual = [
            if input.name == "position_ids" {
                1
            } else {
                batch
            },
            sequence,
        ];
        if wanted
            .iter()
            .zip(actual)
            .any(|(&n, value)| n > 0 && n as usize != value)
        {
            return Err(invalid(
                "graph input dimensions differ from execution shape",
            ));
        }
    }
    let error = |e: ort::Error| errors::inference_error("modernbert_inputs", &e.to_string());
    let mut inputs = ort::inputs![
        "input_ids" => Tensor::from_array(([batch, sequence], input_ids)).map_err(error)?,
        "attention_mask" => Tensor::from_array(([batch, sequence], attention_mask)).map_err(error)?,
    ];
    if positions {
        inputs.push((
            "position_ids".into(),
            Tensor::from_array(([1, sequence], (0..sequence as i64).collect::<Vec<_>>()))
                .map_err(error)?
                .into(),
        ));
    }
    session.run(inputs).map_err(error)
}

/// Retain the compile lease until the first actual execution has completed.
/// Uncached CPU sessions retain their ordinary dynamic batch/sequence behavior.
pub(crate) fn run_cached<'a>(
    session: &'a mut Session,
    lease: &mut Option<crate::core::compilation_cache::CompilationCacheLease>,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    batch: usize,
    sequence: usize,
) -> UnifiedResult<SessionOutputs<'a>> {
    use crate::core::{compilation_cache::with_inference, execution_contract::ExecutionInput};
    validate(&session.inputs)?;
    let mut actual = session
        .inputs
        .iter()
        .map(|input| ExecutionInput {
            name: input.name.clone(),
            dtype: "int64".into(),
            shape: vec![
                if input.name == "position_ids" {
                    1
                } else {
                    batch as i64
                },
                sequence as i64,
            ],
        })
        .collect::<Vec<_>>();
    actual.sort_by(|a, b| a.name.cmp(&b.name));
    with_inference(lease, &actual, || {
        run(session, input_ids, attention_mask, batch, sequence)
    })
}

#[cfg(test)]
mod tests;
