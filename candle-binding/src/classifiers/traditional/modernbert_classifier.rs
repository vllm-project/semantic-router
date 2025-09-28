//! ModernBERT specialized classifier
//!
//! Provides specialized classification functionality for ModernBERT models
//! in the traditional path of the dual-path architecture.

use crate::core::{ModelErrorType, UnifiedError};
use crate::model_error;
use candle_core::{Device, Module, Result, Tensor};
use std::collections::HashMap;

/// Simplified Traditional ModernBERT classifier for compatibility
#[derive(Debug, Clone)]
pub struct TraditionalModernBertClassifier {
    device: Device,
    // Simplified placeholder structure
}

impl TraditionalModernBertClassifier {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    pub fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        // Simplified placeholder implementation
        Tensor::zeros(&[1, 768], candle_core::DType::F32, &self.device)
    }

    pub fn get_embeddings(&self, _text: &str) -> Result<Tensor> {
        // Simplified placeholder implementation for embeddings
        Tensor::zeros(&[1, 768], candle_core::DType::F32, &self.device)
    }
}

/// ModernBERT specialized classifier for traditional path
pub struct ModernBertClassifier {
    model: TraditionalModernBertClassifier,
    classification_heads: HashMap<String, ClassificationHead>,
    device: Device,
    config: ModernBertClassifierConfig,
}

impl ModernBertClassifier {
    /// Create new ModernBERT classifier
    pub fn new(
        model: TraditionalModernBertClassifier,
        config: ModernBertClassifierConfig,
        device: Device,
    ) -> Result<Self> {
        let mut classification_heads = HashMap::new();

        // Create classification heads for different tasks
        for (task_name, num_classes) in &config.task_configs {
            let head = ClassificationHead::new(*num_classes, config.hidden_size, &device)?;
            classification_heads.insert(task_name.clone(), head);
        }

        Ok(Self {
            model,
            classification_heads,
            device,
            config,
        })
    }

    /// Classify text for specific task
    pub fn classify_task(&self, text: &str, task: &str) -> Result<ClassificationResult> {
        // Get embeddings from ModernBERT
        let embeddings = self.model.get_embeddings(text)?;

        // Get task-specific classification head
        let head = self.classification_heads.get(task).ok_or_else(|| {
            let unified_err = model_error!(
                ModelErrorType::ModernBERT,
                "task lookup",
                format!("Unknown task: {}", task),
                task
            );
            candle_core::Error::from(unified_err)
        })?;

        // Perform classification
        let logits = head.forward(&embeddings)?;
        let probabilities = self.softmax(&logits)?;

        // Find best class
        let (class_id, confidence) = self.argmax_with_confidence(&probabilities)?;
        let class_name = self
            .config
            .get_class_name(task, class_id)
            .unwrap_or_else(|| format!("class_{}", class_id));

        Ok(ClassificationResult {
            task: task.to_string(),
            class_name,
            class_id,
            confidence,
            probabilities: self.tensor_to_vec(&probabilities)?,
        })
    }

    /// Classify text for multiple tasks
    pub fn classify_multi_task(
        &self,
        text: &str,
        tasks: &[&str],
    ) -> Result<Vec<ClassificationResult>> {
        let mut results = Vec::new();

        for &task in tasks {
            let result = self.classify_task(text, task)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Batch classification for single task
    pub fn classify_batch(&self, texts: &[&str], task: &str) -> Result<Vec<ClassificationResult>> {
        let mut results = Vec::new();

        for &text in texts {
            let result = self.classify_task(text, task)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Batch classification for multiple tasks
    pub fn classify_batch_multi_task(
        &self,
        texts: &[&str],
        tasks: &[&str],
    ) -> Result<HashMap<String, Vec<ClassificationResult>>> {
        let mut task_results = HashMap::new();

        for &task in tasks {
            let results = self.classify_batch(texts, task)?;
            task_results.insert(task.to_string(), results);
        }

        Ok(task_results)
    }

    /// Get model confidence for text classification
    pub fn get_confidence(&self, text: &str, task: &str) -> Result<f32> {
        let result = self.classify_task(text, task)?;
        Ok(result.confidence)
    }

    /// Extract embeddings without classification
    pub fn extract_embeddings(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.model.get_embeddings(text)?;
        self.tensor_to_vec(&embeddings)
    }

    /// Get supported tasks
    pub fn get_supported_tasks(&self) -> Vec<String> {
        self.classification_heads.keys().cloned().collect()
    }

    /// Add new classification task
    pub fn add_task(&mut self, task_name: &str, num_classes: usize) -> Result<()> {
        if self.classification_heads.contains_key(task_name) {
            let unified_err = model_error!(
                ModelErrorType::ModernBERT,
                "task registration",
                format!("Task already exists: {}", task_name),
                task_name
            );
            return Err(candle_core::Error::from(unified_err));
        }

        let head = ClassificationHead::new(num_classes, self.config.hidden_size, &self.device)?;
        self.classification_heads
            .insert(task_name.to_string(), head);

        Ok(())
    }

    /// Remove classification task
    pub fn remove_task(&mut self, task_name: &str) -> Result<()> {
        if self.classification_heads.remove(task_name).is_none() {
            let unified_err = model_error!(
                ModelErrorType::ModernBERT,
                "task removal",
                format!("Task not found: {}", task_name),
                task_name
            );
            return Err(candle_core::Error::from(unified_err));
        }
        Ok(())
    }

    // Helper methods
    fn softmax(&self, tensor: &Tensor) -> Result<Tensor> {
        candle_nn::ops::softmax(tensor, candle_core::D::Minus1)
    }

    fn argmax_with_confidence(&self, probabilities: &Tensor) -> Result<(usize, f32)> {
        let probs_vec = self.tensor_to_vec(probabilities)?;
        let (max_idx, &max_val) = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Ok((max_idx, max_val))
    }

    fn tensor_to_vec(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        tensor.flatten_all()?.to_vec1::<f32>()
    }
}

/// Classification head for specific tasks
#[derive(Debug)]
pub struct ClassificationHead {
    linear: candle_nn::Linear,
    dropout: candle_nn::Dropout,
    num_classes: usize,
}

impl ClassificationHead {
    pub fn new(num_classes: usize, input_size: usize, device: &Device) -> Result<Self> {
        let vs = candle_nn::VarBuilder::zeros(candle_core::DType::F32, device);
        let linear = candle_nn::linear(input_size, num_classes, vs.pp("classifier"))?;
        let dropout = candle_nn::Dropout::new(0.1);

        Ok(Self {
            linear,
            dropout,
            num_classes,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.dropout.forward(input, false)?;
        self.linear.forward(&hidden)
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Configuration for ModernBERT classifier
#[derive(Debug, Clone)]
pub struct ModernBertClassifierConfig {
    pub hidden_size: usize,
    pub task_configs: HashMap<String, usize>, // task_name -> num_classes
    pub class_names: HashMap<String, Vec<String>>, // task_name -> class_names
    pub dropout_rate: f32,
    pub temperature: f32,
}

impl Default for ModernBertClassifierConfig {
    fn default() -> Self {
        let mut task_configs = HashMap::new();
        task_configs.insert("intent".to_string(), 10);
        task_configs.insert("sentiment".to_string(), 3);

        let mut class_names = HashMap::new();
        class_names.insert(
            "sentiment".to_string(),
            vec![
                "negative".to_string(),
                "neutral".to_string(),
                "positive".to_string(),
            ],
        );

        Self {
            hidden_size: 768,
            task_configs,
            class_names,
            dropout_rate: 0.1,
            temperature: 1.0,
        }
    }
}

impl ModernBertClassifierConfig {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            ..Default::default()
        }
    }

    pub fn add_task(
        &mut self,
        task_name: &str,
        num_classes: usize,
        class_names: Option<Vec<String>>,
    ) {
        self.task_configs.insert(task_name.to_string(), num_classes);
        if let Some(names) = class_names {
            self.class_names.insert(task_name.to_string(), names);
        }
    }

    pub fn get_class_name(&self, task: &str, class_id: usize) -> Option<String> {
        self.class_names
            .get(task)
            .and_then(|names| names.get(class_id))
            .cloned()
    }
}

/// Classification result for ModernBERT classifier
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub task: String,
    pub class_name: String,
    pub class_id: usize,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
}

impl ClassificationResult {
    /// Check if classification is high confidence
    pub fn is_high_confidence(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Get top-k predictions
    pub fn get_top_k(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed_probs: Vec<(usize, f32)> = self
            .probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();

        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_probs.into_iter().take(k).collect()
    }

    /// Get entropy of the prediction distribution
    pub fn get_entropy(&self) -> f32 {
        -self
            .probabilities
            .iter()
            .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
            .sum::<f32>()
    }
}

/// Batch classification result
#[derive(Debug, Clone)]
pub struct BatchClassificationResult {
    pub task: String,
    pub results: Vec<ClassificationResult>,
    pub average_confidence: f32,
    pub high_confidence_count: usize,
    pub processing_time_ms: u64,
}

impl BatchClassificationResult {
    pub fn new(task: String, results: Vec<ClassificationResult>) -> Self {
        let total_confidence: f32 = results.iter().map(|r| r.confidence).sum();
        let average_confidence = total_confidence / results.len() as f32;
        let high_confidence_count = results.iter().filter(|r| r.is_high_confidence(0.9)).count();

        Self {
            task,
            results,
            average_confidence,
            high_confidence_count,
            processing_time_ms: 0, // Will be set externally
        }
    }

    pub fn get_accuracy_stats(&self) -> AccuracyStats {
        let confidence_scores: Vec<f32> = self.results.iter().map(|r| r.confidence).collect();
        let min_confidence = confidence_scores
            .iter()
            .fold(f32::INFINITY, |a, &b| a.min(b));
        let max_confidence = confidence_scores
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        AccuracyStats {
            average_confidence: self.average_confidence,
            min_confidence,
            max_confidence,
            high_confidence_ratio: self.high_confidence_count as f32 / self.results.len() as f32,
            total_samples: self.results.len(),
        }
    }
}

/// Accuracy statistics for batch results
#[derive(Debug, Clone)]
pub struct AccuracyStats {
    pub average_confidence: f32,
    pub min_confidence: f32,
    pub max_confidence: f32,
    pub high_confidence_ratio: f32,
    pub total_samples: usize,
}
