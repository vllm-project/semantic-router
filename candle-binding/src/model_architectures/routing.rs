//! Intelligent Routing System for Dual-Path Architecture
//!
//! This module implements smart routing logic that automatically selects
//! the optimal path (Traditional vs LoRA) based on requirements and performance.

use crate::core::config_loader::{GlobalConfigLoader, RouterConfig};
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
    /// Router configuration (loaded from config.yaml)
    router_config: RouterConfig,
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
            router_config: GlobalConfigLoader::load_router_config_safe(),
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
        if requirements.confidence_threshold >= self.router_config.high_confidence_threshold {
            return PathSelection {
                selected_path: ModelType::LoRA,
                confidence: 0.95,
                reasoning: format!(
                    "High confidence requirement (≥{}) -> LoRA path",
                    self.router_config.high_confidence_threshold
                ),
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
        if requirements.max_latency
            < Duration::from_millis(self.router_config.low_latency_threshold_ms)
        {
            return PathSelection {
                selected_path: ModelType::LoRA,
                confidence: 0.85,
                reasoning: format!(
                    "Low latency requirement (<{}ms) -> LoRA path",
                    self.router_config.low_latency_threshold_ms
                ),
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
        // Calculate base score for model type
        let base_score = match model_type {
            ModelType::LoRA => self.router_config.lora_baseline_score,
            ModelType::Traditional => self.router_config.traditional_baseline_score,
            ModelType::Qwen3Embedding | ModelType::GemmaEmbedding | ModelType::MmBertEmbedding => {
                self.router_config.embedding_baseline_score
            }
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
                if requirements.confidence_threshold >= self.router_config.high_confidence_threshold
                {
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
            ModelType::Qwen3Embedding => {
                // Qwen3 excels at long context (up to 32K)
                // Adjust score based on sequence length (estimated from batch size * avg tokens)
                let estimated_seq_len = requirements.batch_size * 128; // Conservative estimate
                if estimated_seq_len > 2048 {
                    score += 0.3; // Strong advantage for very long context (only Qwen3 supports)
                } else if estimated_seq_len > 512 {
                    score += 0.15; // Moderate advantage for long context
                }
                // Qwen3 provides high quality embeddings
                if requirements.priority == ProcessingPriority::Accuracy {
                    score += 0.2;
                }
            }
            ModelType::GemmaEmbedding => {
                //Gemma excels at short-to-medium context (up to 8K) with speed
                let estimated_seq_len = requirements.batch_size * 128;
                if estimated_seq_len <= 2048 {
                    score += 0.15; // Advantage for short-to-medium context
                }
                // Gemma is faster (good for latency-sensitive applications)
                if requirements.priority == ProcessingPriority::Latency {
                    score += 0.25;
                }
            }
            ModelType::MmBertEmbedding => {
                // mmBERT excels at multilingual and long context (up to 32K)
                let estimated_seq_len = requirements.batch_size * 128;
                if estimated_seq_len > 2048 {
                    score += 0.25; // Strong advantage for long context
                } else if estimated_seq_len > 512 {
                    score += 0.15; // Moderate advantage
                }
                // mmBERT is faster than BGE-M3 (1.6-3.1× due to FA2)
                if requirements.priority == ProcessingPriority::Latency {
                    score += 0.2;
                }
                // 2D Matryoshka provides flexibility
                score += 0.1;
            }
        }

        score.max(0.0).min(1.0)
    }

    /// Get expected performance for model type
    fn get_expected_performance(&self, model_type: ModelType) -> PathMetrics {
        self.current_metrics
            .get(&model_type)
            .cloned()
            .unwrap_or_else(|| match model_type {
                ModelType::LoRA => PathMetrics {
                    avg_execution_time: Duration::from_millis(
                        self.router_config.lora_default_execution_time_ms,
                    ),
                    avg_confidence: self.router_config.lora_default_confidence,
                    success_rate: self.router_config.lora_default_success_rate,
                    total_executions: 0,
                },
                ModelType::Traditional => PathMetrics {
                    avg_execution_time: Duration::from_millis(
                        self.router_config.traditional_default_execution_time_ms,
                    ),
                    avg_confidence: self.router_config.traditional_default_confidence,
                    success_rate: self.router_config.traditional_default_success_rate,
                    total_executions: 0,
                },
                ModelType::Qwen3Embedding => PathMetrics {
                    avg_execution_time: Duration::from_millis(30), // ~30ms for short sequences
                    avg_confidence: 0.8,
                    success_rate: 0.95,
                    total_executions: 0,
                },
                ModelType::GemmaEmbedding => PathMetrics {
                    avg_execution_time: Duration::from_millis(20), // ~20ms for short sequences
                    avg_confidence: 0.75,
                    success_rate: 0.95,
                    total_executions: 0,
                },
                ModelType::MmBertEmbedding => PathMetrics {
                    avg_execution_time: Duration::from_millis(25), // ~25ms (1.6-3.1× faster than BGE-M3)
                    avg_confidence: 0.80,
                    success_rate: 0.96,
                    total_executions: 0,
                },
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
            ModelType::Qwen3Embedding | ModelType::GemmaEmbedding | ModelType::MmBertEmbedding => {
                // FUTURE ENHANCEMENT: Optional support for manual embedding model preference
                // Current implementation: Intelligent automatic selection via UnifiedClassifier
                // This provides optimal quality-latency balance based on user priorities
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

        // Update success rate (using configurable threshold)
        let success_count = if confidence > self.router_config.success_confidence_threshold {
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

        // Factor 1: Multi-task vs Single-task (mutually exclusive)
        if requirements.tasks.len() > 1 {
            lora_score += self.router_config.multi_task_lora_weight; // LoRA excels at parallel processing
        } else {
            traditional_score += self.router_config.single_task_traditional_weight;
            // Traditional stable for single tasks
        }

        // Factor 2: Batch size efficiency (improved logic covering all cases)
        match requirements.batch_size {
            1 => {
                // Single item - Traditional advantage
                traditional_score += self.router_config.small_batch_traditional_weight;
            }
            2..=3 => {
                // Medium batch - slight advantage to both (neutral)
                lora_score += self.router_config.medium_batch_weight;
                traditional_score += self.router_config.medium_batch_weight;
            }
            _ if requirements.batch_size >= self.router_config.large_batch_threshold => {
                // Large batch - LoRA advantage
                lora_score += self.router_config.large_batch_lora_weight;
            }
            _ => {
                // Default case for other sizes - neutral
                lora_score += self.router_config.medium_batch_weight;
                traditional_score += self.router_config.medium_batch_weight;
            }
        }

        // Factor 3: Confidence requirements (mutually exclusive)
        if requirements.confidence_threshold >= self.router_config.high_confidence_threshold {
            lora_score += self.router_config.high_confidence_lora_weight; // LoRA provides ultra-high confidence
        } else if requirements.confidence_threshold <= 0.9 {
            traditional_score += self.router_config.low_confidence_traditional_weight;
            // Traditional sufficient for lower requirements
        }
        // Note: Medium confidence (0.9 < threshold < high_threshold) gets no bonus - neutral

        // Factor 4: Latency requirements (mutually exclusive)
        if requirements.max_latency
            <= Duration::from_millis(self.router_config.low_latency_threshold_ms)
        {
            lora_score += self.router_config.low_latency_lora_weight; // LoRA is faster
        } else {
            traditional_score += self.router_config.high_latency_traditional_weight;
            // Traditional acceptable for relaxed timing
        }

        // Factor 5: Historical performance (conditional, not always present)
        if let Some(lora_metrics) = self.current_metrics.get(&ModelType::LoRA) {
            if let Some(traditional_metrics) = self.current_metrics.get(&ModelType::Traditional) {
                if lora_metrics.avg_execution_time < traditional_metrics.avg_execution_time {
                    lora_score += self.router_config.performance_history_weight;
                } else {
                    traditional_score += self.router_config.performance_history_weight;
                }
            }
        }

        // Make intelligent decision with detailed scoring info
        let total_score = lora_score + traditional_score;
        let (selected_path, confidence, reasoning) = if lora_score > traditional_score {
            (
                ModelType::LoRA,
                if total_score > 0.0 { (lora_score / total_score).min(1.0) } else { 0.5 },
                format!("LoRA selected (score: {:.3} vs {:.3}): tasks={}, batch={}, confidence≥{:.2}, latency≤{}ms",
                    lora_score, traditional_score,
                    requirements.tasks.len(),
                    requirements.batch_size,
                    requirements.confidence_threshold,
                    requirements.max_latency.as_millis())
            )
        } else if traditional_score > lora_score {
            (
                ModelType::Traditional,
                if total_score > 0.0 { (traditional_score / total_score).min(1.0) } else { 0.5 },
                format!("Traditional selected (score: {:.3} vs {:.3}): tasks={}, batch={}, confidence≥{:.2}, latency≤{}ms",
                    traditional_score, lora_score,
                    requirements.tasks.len(),
                    requirements.batch_size,
                    requirements.confidence_threshold,
                    requirements.max_latency.as_millis())
            )
        } else {
            // Tie case - default to LoRA for performance, use configurable confidence
            (
                ModelType::LoRA,
                self.router_config.tie_break_confidence,
                format!(
                    "Tie (both score {:.3}) - defaulting to LoRA for performance",
                    lora_score
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
                    Duration::from_millis(self.router_config.lora_default_execution_time_ms)
                } else {
                    Duration::from_millis(self.router_config.traditional_default_execution_time_ms)
                },
                avg_confidence: if selected_path == ModelType::LoRA {
                    self.router_config.lora_default_confidence
                } else {
                    self.router_config.traditional_default_confidence
                },
                success_rate: if selected_path == ModelType::LoRA {
                    self.router_config.lora_default_success_rate
                } else {
                    self.router_config.traditional_default_success_rate
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
    fn calculate_average_performance(
        &self,
        model_type: ModelType,
        success_threshold: f32,
    ) -> Option<PathMetrics> {
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
        let success_count = records
            .iter()
            .filter(|r| r.confidence > success_threshold)
            .count();

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
        let router_config = RouterConfig::default();
        Self {
            confidence_threshold: router_config.default_confidence_threshold,
            max_latency: Duration::from_millis(router_config.default_max_latency_ms),
            batch_size: router_config.default_batch_size,
            tasks: vec![TaskType::Intent],
            priority: ProcessingPriority::Balanced,
        }
    }
}

impl Default for PathMetrics {
    fn default() -> Self {
        let router_config = RouterConfig::default();
        Self {
            avg_execution_time: Duration::from_millis(router_config.default_avg_execution_time_ms),
            avg_confidence: router_config.default_confidence_threshold,
            success_rate: router_config.traditional_default_success_rate, // Use traditional as default
            total_executions: 0,
        }
    }
}
