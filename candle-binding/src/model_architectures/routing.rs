//! Intelligent Routing System for Dual-Path Architecture
//!
//! This module implements smart routing logic that automatically selects
//! the optimal path (Traditional vs LoRA) based on requirements and performance.

use crate::model_architectures::config::{PathSelectionStrategy, ProcessingPriority};
use crate::model_architectures::traits::{ModelType, TaskType};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Intelligent router for dual-path selection
#[derive(Debug)]
pub struct DualPathRouter {
    /// Path selection strategy
    strategy: PathSelectionStrategy,
    /// Performance history for learning
    performance_history: PerformanceHistory,
    /// Current performance metrics
    current_metrics: HashMap<ModelType, PathMetrics>,
}

/// Performance history for intelligent learning
#[derive(Debug)]
struct PerformanceHistory {
    /// Historical performance data
    history: Vec<PerformanceRecord>,
    /// Maximum history size
    max_size: usize,
}

/// Individual performance record
#[derive(Debug, Clone)]
struct PerformanceRecord {
    /// Model type used
    model_type: ModelType,
    /// Tasks performed
    tasks: Vec<TaskType>,
    /// Batch size
    batch_size: usize,
    /// Execution time
    execution_time: Duration,
    /// Confidence achieved
    confidence: f32,
    /// Timestamp
    timestamp: Instant,
}

/// Path performance metrics
#[derive(Debug, Clone)]
pub struct PathMetrics {
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Average confidence
    pub avg_confidence: f32,
    /// Success rate
    pub success_rate: f32,
    /// Total executions
    pub total_executions: u64,
}

/// Processing requirements for path selection
#[derive(Debug, Clone)]
pub struct ProcessingRequirements {
    /// Required confidence threshold
    pub confidence_threshold: f32,
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Batch size
    pub batch_size: usize,
    /// Required tasks
    pub tasks: Vec<TaskType>,
    /// Processing priority
    pub priority: ProcessingPriority,
}

/// Path selection result
#[derive(Debug, Clone)]
pub struct PathSelection {
    /// Selected model type
    pub selected_path: ModelType,
    /// Selection confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Reasoning for selection
    pub reasoning: String,
    /// Expected performance
    pub expected_performance: PathMetrics,
}

impl DualPathRouter {
    /// Create new router with strategy
    pub fn new(strategy: PathSelectionStrategy) -> Self {
        Self {
            strategy,
            performance_history: PerformanceHistory::new(1000),
            current_metrics: HashMap::new(),
        }
    }

    /// Select optimal path based on requirements
    pub fn select_path(&self, requirements: &ProcessingRequirements) -> PathSelection {
        match self.strategy {
            PathSelectionStrategy::AlwaysLoRA => PathSelection {
                selected_path: ModelType::LoRA,
                confidence: 1.0,
                reasoning: "Strategy: Always use LoRA path".to_string(),
                expected_performance: self.get_expected_performance(ModelType::LoRA),
            },
            PathSelectionStrategy::AlwaysTraditional => PathSelection {
                selected_path: ModelType::Traditional,
                confidence: 1.0,
                reasoning: "Strategy: Always use Traditional path".to_string(),
                expected_performance: self.get_expected_performance(ModelType::Traditional),
            },
            PathSelectionStrategy::Automatic => self.automatic_selection(requirements),
            PathSelectionStrategy::PerformanceBased => {
                self.performance_based_selection(requirements)
            }
        }
    }

    /// Automatic path selection based on requirements
    fn automatic_selection(&self, requirements: &ProcessingRequirements) -> PathSelection {
        // High confidence requirement -> LoRA path
        if requirements.confidence_threshold >= 0.99 {
            return PathSelection {
                selected_path: ModelType::LoRA,
                confidence: 0.95,
                reasoning: "High confidence requirement (≥0.99) -> LoRA path".to_string(),
                expected_performance: self.get_expected_performance(ModelType::LoRA),
            };
        }

        // Multiple tasks -> LoRA parallel processing
        if requirements.tasks.len() > 1 {
            return PathSelection {
                selected_path: ModelType::LoRA,
                confidence: 0.90,
                reasoning: "Multiple tasks -> LoRA parallel processing".to_string(),
                expected_performance: self.get_expected_performance(ModelType::LoRA),
            };
        }

        // Low latency requirement -> LoRA path
        if requirements.max_latency < Duration::from_millis(2000) {
            return PathSelection {
                selected_path: ModelType::LoRA,
                confidence: 0.85,
                reasoning: "Low latency requirement -> LoRA path".to_string(),
                expected_performance: self.get_expected_performance(ModelType::LoRA),
            };
        }

        // Accuracy priority -> Traditional path
        if requirements.priority == ProcessingPriority::Accuracy {
            return PathSelection {
                selected_path: ModelType::Traditional,
                confidence: 0.80,
                reasoning: "Accuracy priority -> Traditional path".to_string(),
                expected_performance: self.get_expected_performance(ModelType::Traditional),
            };
        }

        // Default: LoRA for better performance
        PathSelection {
            selected_path: ModelType::LoRA,
            confidence: 0.75,
            reasoning: "Default: LoRA for better performance".to_string(),
            expected_performance: self.get_expected_performance(ModelType::LoRA),
        }
    }

    /// Performance-based selection using historical data
    fn performance_based_selection(&self, requirements: &ProcessingRequirements) -> PathSelection {
        let lora_score = self.calculate_path_score(ModelType::LoRA, requirements);
        let traditional_score = self.calculate_path_score(ModelType::Traditional, requirements);

        if lora_score > traditional_score {
            PathSelection {
                selected_path: ModelType::LoRA,
                confidence: (lora_score / (lora_score + traditional_score)).min(1.0),
                reasoning: format!(
                    "Performance-based: LoRA score {:.2} > Traditional score {:.2}",
                    lora_score, traditional_score
                ),
                expected_performance: self.get_expected_performance(ModelType::LoRA),
            }
        } else {
            PathSelection {
                selected_path: ModelType::Traditional,
                confidence: (traditional_score / (lora_score + traditional_score)).min(1.0),
                reasoning: format!(
                    "Performance-based: Traditional score {:.2} > LoRA score {:.2}",
                    traditional_score, lora_score
                ),
                expected_performance: self.get_expected_performance(ModelType::Traditional),
            }
        }
    }

    /// Calculate path score based on requirements and history
    fn calculate_path_score(
        &self,
        model_type: ModelType,
        requirements: &ProcessingRequirements,
    ) -> f32 {
        let base_score = match model_type {
            ModelType::LoRA => 0.8,        // LoRA baseline: high performance
            ModelType::Traditional => 0.7, // Traditional baseline: high reliability
        };

        let mut score = base_score;

        // Adjust based on historical performance
        if let Some(metrics) = self.current_metrics.get(&model_type) {
            // Confidence factor
            if metrics.avg_confidence >= requirements.confidence_threshold {
                score += 0.2;
            } else {
                score -= 0.3;
            }

            // Latency factor
            if metrics.avg_execution_time <= requirements.max_latency {
                score += 0.1;
            } else {
                score -= 0.2;
            }

            // Success rate factor
            score += (metrics.success_rate - 0.5) * 0.4;
        }

        // Task-specific adjustments
        match model_type {
            ModelType::LoRA => {
                // LoRA excels at multiple tasks
                if requirements.tasks.len() > 1 {
                    score += 0.3;
                }
                // LoRA excels at high confidence requirements
                if requirements.confidence_threshold >= 0.99 {
                    score += 0.2;
                }
            }
            ModelType::Traditional => {
                // Traditional excels at single tasks
                if requirements.tasks.len() == 1 {
                    score += 0.1;
                }
                // Traditional excels at accuracy priority
                if requirements.priority == ProcessingPriority::Accuracy {
                    score += 0.2;
                }
            }
        }

        score.max(0.0).min(1.0)
    }

    /// Get expected performance for model type
    fn get_expected_performance(&self, model_type: ModelType) -> PathMetrics {
        self.current_metrics
            .get(&model_type)
            .cloned()
            .unwrap_or_else(|| {
                match model_type {
                    ModelType::LoRA => PathMetrics {
                        avg_execution_time: Duration::from_millis(1345), // 70.5% faster
                        avg_confidence: 0.99,
                        success_rate: 0.98,
                        total_executions: 0,
                    },
                    ModelType::Traditional => PathMetrics {
                        avg_execution_time: Duration::from_millis(4567), // Stable baseline
                        avg_confidence: 0.95,
                        success_rate: 0.99,
                        total_executions: 0,
                    },
                }
            })
    }

    ///   Set preferred path for dynamic switching
    pub fn set_preferred_path(&mut self, preferred_path: ModelType) {
        match preferred_path {
            ModelType::LoRA => {
                self.strategy = PathSelectionStrategy::AlwaysLoRA;
            }
            ModelType::Traditional => {
                self.strategy = PathSelectionStrategy::AlwaysTraditional;
            }
        }
    }

    /// Record performance for adaptive learning
    pub fn record_performance(
        &mut self,
        model_type: ModelType,
        tasks: Vec<TaskType>,
        batch_size: usize,
        execution_time: Duration,
        confidence: f32,
    ) {
        let record = PerformanceRecord {
            model_type,
            tasks,
            batch_size,
            execution_time,
            confidence,
            timestamp: Instant::now(),
        };

        self.performance_history.add_record(record);
        self.update_current_metrics(model_type, execution_time, confidence);
    }

    /// Update current performance metrics
    fn update_current_metrics(
        &mut self,
        model_type: ModelType,
        execution_time: Duration,
        confidence: f32,
    ) {
        let metrics = self
            .current_metrics
            .entry(model_type)
            .or_insert(PathMetrics {
                avg_execution_time: Duration::from_millis(0),
                avg_confidence: 0.0,
                success_rate: 1.0,
                total_executions: 0,
            });

        let old_count = metrics.total_executions;
        let new_count = old_count + 1;

        // Update average execution time
        let old_avg_ms = metrics.avg_execution_time.as_millis() as f32;
        let new_avg_ms =
            (old_avg_ms * old_count as f32 + execution_time.as_millis() as f32) / new_count as f32;
        metrics.avg_execution_time = Duration::from_millis(new_avg_ms as u64);

        // Update average confidence
        metrics.avg_confidence =
            (metrics.avg_confidence * old_count as f32 + confidence) / new_count as f32;

        // Update success rate (assuming confidence > 0.8 is success)
        let success_count = if confidence > 0.8 {
            old_count + 1
        } else {
            old_count
        };
        metrics.success_rate = success_count as f32 / new_count as f32;

        metrics.total_executions = new_count;
    }

    /// Get performance comparison between paths
    pub fn get_performance_comparison(&self) -> HashMap<ModelType, PathMetrics> {
        self.current_metrics.clone()
    }

    ///  Reset performance history
    pub fn reset_performance_history(&mut self) {
        self.performance_history = PerformanceHistory::new(1000);
        self.current_metrics.clear();
    }

    ///  Enhanced path selection with super intelligence
    pub fn select_path_intelligent(&self, requirements: &ProcessingRequirements) -> PathSelection {
        // Multi-factor analysis for super intelligent routing
        let mut lora_score = 0.0f32;
        let mut traditional_score = 0.0f32;

        // Factor 1: Multi-task parallel benefit
        if requirements.tasks.len() > 1 {
            lora_score += 0.4; // LoRA excels at Intent||PII||Security parallel processing
        } else {
            traditional_score += 0.2; // Traditional is stable for single tasks
        }

        // Factor 2: Batch size efficiency
        if requirements.batch_size >= 4 {
            lora_score += 0.3; // LoRA parallel processing benefits
        } else if requirements.batch_size == 1 {
            traditional_score += 0.3; // Traditional efficient for single items
        }

        // Factor 3: Confidence requirements
        if requirements.confidence_threshold >= 0.99 {
            lora_score += 0.3; // LoRA provides ultra-high confidence
        } else if requirements.confidence_threshold <= 0.9 {
            traditional_score += 0.2; // Traditional sufficient for lower requirements
        }

        // Factor 4: Latency requirements
        if requirements.max_latency <= Duration::from_millis(2000) {
            lora_score += 0.4; // LoRA is 70.5% faster
        } else {
            traditional_score += 0.1; // Traditional acceptable for relaxed timing
        }

        // Factor 5: Historical performance
        if let Some(lora_metrics) = self.current_metrics.get(&ModelType::LoRA) {
            if let Some(traditional_metrics) = self.current_metrics.get(&ModelType::Traditional) {
                if lora_metrics.avg_execution_time < traditional_metrics.avg_execution_time {
                    lora_score += 0.2;
                } else {
                    traditional_score += 0.2;
                }
            }
        }

        // Make intelligent decision
        let (selected_path, confidence, reasoning) = if lora_score > traditional_score {
            (
                ModelType::LoRA,
                (lora_score / (lora_score + traditional_score)).min(1.0),
                format!("LoRA selected: multi-task={}, batch_size={}, confidence_req={:.2}, latency_req={}ms",
                    requirements.tasks.len() > 1,
                    requirements.batch_size,
                    requirements.confidence_threshold,
                    requirements.max_latency.as_millis())
            )
        } else {
            (
                ModelType::Traditional,
                (traditional_score / (lora_score + traditional_score)).min(1.0),
                format!(
                    "Traditional selected: single_task={}, simple_batch={}, standard_confidence",
                    requirements.tasks.len() == 1,
                    requirements.batch_size <= 3
                ),
            )
        };

        // Create expected performance based on historical data
        let expected_performance = self
            .current_metrics
            .get(&selected_path)
            .cloned()
            .unwrap_or_else(|| PathMetrics {
                avg_execution_time: if selected_path == ModelType::LoRA {
                    Duration::from_millis(1345) // LoRA baseline: 1.345s
                } else {
                    Duration::from_millis(4567) // Traditional baseline: 4.567s
                },
                avg_confidence: if selected_path == ModelType::LoRA {
                    0.99
                } else {
                    0.95
                },
                success_rate: if selected_path == ModelType::LoRA {
                    0.98
                } else {
                    0.95
                },
                total_executions: 0,
            });

        PathSelection {
            selected_path,
            confidence,
            reasoning,
            expected_performance,
        }
    }

    /// Get current path statistics
    pub fn get_statistics(&self) -> RouterStatistics {
        let total_records = self.performance_history.history.len();
        let lora_count = self
            .performance_history
            .history
            .iter()
            .filter(|r| r.model_type == ModelType::LoRA)
            .count();
        let traditional_count = total_records - lora_count;

        RouterStatistics {
            total_selections: total_records as u64,
            lora_selections: lora_count as u64,
            traditional_selections: traditional_count as u64,
            lora_metrics: self.current_metrics.get(&ModelType::LoRA).cloned(),
            traditional_metrics: self.current_metrics.get(&ModelType::Traditional).cloned(),
        }
    }
}

impl PerformanceHistory {
    /// Create new performance history
    fn new(max_size: usize) -> Self {
        Self {
            history: Vec::new(),
            max_size,
        }
    }

    /// Add performance record
    fn add_record(&mut self, record: PerformanceRecord) {
        self.history.push(record);

        // Keep history size under limit
        if self.history.len() > self.max_size {
            self.history.remove(0);
        }
    }

    /// Get recent performance for model type
    fn get_recent_performance(
        &self,
        model_type: ModelType,
        limit: usize,
    ) -> Vec<&PerformanceRecord> {
        self.history
            .iter()
            .rev()
            .filter(|record| record.model_type == model_type)
            .take(limit)
            .collect()
    }

    /// Calculate average performance for model type
    fn calculate_average_performance(&self, model_type: ModelType) -> Option<PathMetrics> {
        let records: Vec<_> = self
            .history
            .iter()
            .filter(|record| record.model_type == model_type)
            .collect();

        if records.is_empty() {
            return None;
        }

        let total_time: u128 = records.iter().map(|r| r.execution_time.as_millis()).sum();
        let total_confidence: f32 = records.iter().map(|r| r.confidence).sum();
        let success_count = records.iter().filter(|r| r.confidence > 0.8).count();

        Some(PathMetrics {
            avg_execution_time: Duration::from_millis((total_time / records.len() as u128) as u64),
            avg_confidence: total_confidence / records.len() as f32,
            success_rate: success_count as f32 / records.len() as f32,
            total_executions: records.len() as u64,
        })
    }
}

/// Router statistics
#[derive(Debug, Clone)]
pub struct RouterStatistics {
    /// Total path selections made
    pub total_selections: u64,
    /// LoRA path selections
    pub lora_selections: u64,
    /// Traditional path selections
    pub traditional_selections: u64,
    /// LoRA path metrics
    pub lora_metrics: Option<PathMetrics>,
    /// Traditional path metrics
    pub traditional_metrics: Option<PathMetrics>,
}

impl Default for ProcessingRequirements {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.95,
            max_latency: Duration::from_millis(5000),
            batch_size: 4,
            tasks: vec![TaskType::Intent],
            priority: ProcessingPriority::Balanced,
        }
    }
}

impl Default for PathMetrics {
    fn default() -> Self {
        Self {
            avg_execution_time: Duration::from_millis(3000),
            avg_confidence: 0.95,
            success_rate: 0.95,
            total_executions: 0,
        }
    }
}
