use crate::core::{ConfigErrorType, UnifiedError, UnifiedResult};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub(super) fn load_guard_var_builder<'a>(
    base_dir: &Path,
    dtype: DType,
    device: &'a Device,
) -> UnifiedResult<VarBuilder<'a>> {
    let single_weights_path = base_dir.join("model.safetensors");
    let index_path = base_dir.join("model.safetensors.index.json");

    if single_weights_path.exists() {
        return load_safetensors(&[single_weights_path], dtype, device);
    }
    if index_path.exists() {
        return load_sharded_safetensors(base_dir, &index_path, dtype, device);
    }

    Err(UnifiedError::Configuration {
        operation: "find weights".to_string(),
        source: ConfigErrorType::ParseError(
            "No model.safetensors or model.safetensors.index.json found".to_string(),
        ),
        context: None,
    })
}

fn load_sharded_safetensors<'a>(
    base_dir: &Path,
    index_path: &Path,
    dtype: DType,
    device: &'a Device,
) -> UnifiedResult<VarBuilder<'a>> {
    let index_data = std::fs::read(index_path)?;
    let index: serde_json::Value =
        serde_json::from_slice(&index_data).map_err(|e| UnifiedError::Configuration {
            operation: "parse index".to_string(),
            source: ConfigErrorType::ParseError(e.to_string()),
            context: None,
        })?;

    let weight_map =
        index["weight_map"]
            .as_object()
            .ok_or_else(|| UnifiedError::Configuration {
                operation: "parse weight_map".to_string(),
                source: ConfigErrorType::ParseError("Missing weight_map".to_string()),
                context: None,
            })?;

    let mut weight_files = HashSet::new();
    for file in weight_map.values() {
        if let Some(f) = file.as_str() {
            weight_files.insert(base_dir.join(f));
        }
    }
    let weight_files: Vec<PathBuf> = weight_files.into_iter().collect();
    load_safetensors(&weight_files, dtype, device)
}

fn load_safetensors<'a>(
    paths: &[PathBuf],
    dtype: DType,
    device: &'a Device,
) -> UnifiedResult<VarBuilder<'a>> {
    unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }.map_err(|e| {
        UnifiedError::Processing {
            operation: "load weights".to_string(),
            source: e.to_string(),
            input_context: None,
        }
    })
}
