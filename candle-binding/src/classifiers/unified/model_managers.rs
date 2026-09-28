//! Model-manager adapters used by the dual-path classifier.

use super::UnifiedTaskResult;
use crate::core::{ModelErrorType, UnifiedError};
use crate::model_architectures::config::{LoRAConfig, TraditionalConfig};
use crate::model_architectures::traits::TaskType;
use crate::model_architectures::unified_interface::CoreModel;
use crate::model_error;
use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

type ClassificationModel =
    dyn CoreModel<Output = (usize, f32), Config = String, Error = candle_core::Error>;
type ClassificationModelRegistry = HashMap<String, Box<ClassificationModel>>;

/// LoRA classification output with performance metrics.
#[derive(Debug, Clone)]
pub struct LoRAClassificationOutput {
    pub task_results: HashMap<TaskType, UnifiedTaskResult>,
    pub processing_time_ms: f32,
    pub performance_improvement: f32,
    pub parallel_efficiency: f32,
}

/// Traditional model manager for the unified classifier.
#[derive(Debug)]
pub struct TraditionalModelManager {
    pub models: ClassificationModelRegistry,
    pub device: Device,
}

impl TraditionalModelManager {
    pub fn new(_config: TraditionalConfig) -> Result<Self, candle_core::Error> {
        Ok(Self {
            models: HashMap::new(),
            device: Device::Cpu,
        })
    }

    pub fn load_modernbert_for_task(&self, task: TaskType) -> Result<(), candle_core::Error> {
        let _model_key = format!("modernbert_{:?}", task);
        let (_model_path, _config_path) = match task {
            TaskType::Intent => (
                "models/intent_classifier",
                "models/intent_classifier/config.json",
            ),
            TaskType::PII => ("models/pii_classifier", "models/pii_classifier/config.json"),
            TaskType::Security => (
                "models/jailbreak_classifier",
                "models/jailbreak_classifier/config.json",
            ),
            TaskType::Classification => (
                "models/category_classifier",
                "models/category_classifier/config.json",
            ),
            TaskType::TokenClassification => (
                "models/token_classifier",
                "models/token_classifier/config.json",
            ),
        };

        Ok(())
    }
}

/// LoRA model manager for the unified classifier.
#[derive(Debug)]
pub struct LoRAModelManager {
    pub models: ClassificationModelRegistry,
    pub device: Device,
}

impl LoRAModelManager {
    pub fn new_with_model_paths(
        intent_model_path: &str,
        pii_model_path: &str,
        security_model_path: &str,
        use_cpu: bool,
    ) -> Result<Self, candle_core::Error> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };
        let manager = Self {
            models: HashMap::new(),
            device,
        };
        manager.load_lora_models(
            intent_model_path,
            pii_model_path,
            security_model_path,
            use_cpu,
        )?;
        Ok(manager)
    }

    pub fn new(_config: LoRAConfig) -> Result<Self, candle_core::Error> {
        Ok(Self {
            models: HashMap::new(),
            device: Device::Cpu,
        })
    }

    pub fn load_lora_models(
        &self,
        intent_model_path: &str,
        pii_model_path: &str,
        security_model_path: &str,
        use_cpu: bool,
    ) -> Result<(), candle_core::Error> {
        use crate::classifiers::lora::parallel_engine::ParallelLoRAEngine;

        let _engine = ParallelLoRAEngine::new(
            self.device.clone(),
            intent_model_path,
            pii_model_path,
            security_model_path,
            use_cpu,
        )
        .map_err(|error| {
            let unified_error = model_error!(
                ModelErrorType::LoRA,
                "parallel engine creation",
                format!("Failed to create ParallelLoRAEngine: {}", error),
                "unified classifier"
            );
            candle_core::Error::from(unified_error)
        })?;

        Ok(())
    }

    pub fn auto_classify(
        &self,
        _input_tensor: &Tensor,
        _tasks: Vec<TaskType>,
    ) -> Result<LoRAClassificationOutput, candle_core::Error> {
        let unified_error = model_error!(
            ModelErrorType::LoRA,
            "auto classification",
            "LoRA auto_classify requires ParallelLoRAEngine to be stored in struct and used for tensor inference",
            "unified classifier"
        );
        Err(candle_core::Error::from(unified_error))
    }
}
