//! Traditional batch processor
//!
//! Provides efficient batch processing capabilities for traditional models
//! in the dual-path architecture.

use crate::core::processing_errors;
use candle_core::{Device, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Traditional batch processor for sequential processing
pub struct TraditionalBatchProcessor {
    device: Device,
    config: BatchProcessorConfig,
    metrics: ProcessingMetrics,
}

impl TraditionalBatchProcessor {
    /// Create new batch processor
    pub fn new(device: Device, config: BatchProcessorConfig) -> Self {
        Self {
            device,
            config,
            metrics: ProcessingMetrics::new(),
        }
    }

    /// Process batch of texts with single task
    pub fn process_batch<T, F>(&mut self, texts: &[&str], processor: F) -> Result<BatchResult<T>>
    where
        F: Fn(&str) -> Result<T>,
    {
        let start_time = Instant::now();
        let mut results = Vec::with_capacity(texts.len());
        let mut failed_indices = Vec::new();

        // Sequential processing for traditional path
        for (idx, &text) in texts.iter().enumerate() {
            match processor(text) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // Convert to unified error for consistent logging
                    let unified_err =
                        processing_errors::batch_processing(1, &format!("item {}: {}", idx, e));
                    failed_indices.push((idx, unified_err.to_string()));
                    // Continue processing other items in batch
                }
            }
        }

        let processing_time = start_time.elapsed();
        self.metrics
            .record_batch(texts.len(), processing_time, failed_indices.len());
        let success_rate = (texts.len() - failed_indices.len()) as f32 / texts.len() as f32;

        Ok(BatchResult {
            results,
            failed_indices,
            processing_time,
            batch_size: texts.len(),
            success_rate,
        })
    }

    /// Process batch with chunking for large batches
    pub fn process_large_batch<T, F>(
        &mut self,
        texts: &[&str],
        processor: F,
    ) -> Result<BatchResult<T>>
    where
        F: Fn(&str) -> Result<T> + Copy,
    {
        if texts.len() <= self.config.max_batch_size {
            return self.process_batch(texts, processor);
        }

        let mut all_results = Vec::new();
        let mut all_failed = Vec::new();
        let total_start = Instant::now();

        // Process in chunks
        for (chunk_idx, chunk) in texts.chunks(self.config.max_batch_size).enumerate() {
            let chunk_result = self.process_batch(chunk, processor)?;

            // Merge results
            all_results.extend(chunk_result.results);

            // Adjust failed indices for global indexing
            for (local_idx, error) in chunk_result.failed_indices {
                let global_idx = chunk_idx * self.config.max_batch_size + local_idx;
                all_failed.push((global_idx, error));
            }

            // Optional delay between chunks to prevent overload
            if chunk_idx > 0 && self.config.chunk_delay_ms > 0 {
                std::thread::sleep(Duration::from_millis(self.config.chunk_delay_ms));
            }
        }

        let total_time = total_start.elapsed();
        let success_rate = (texts.len() - all_failed.len()) as f32 / texts.len() as f32;

        Ok(BatchResult {
            results: all_results,
            failed_indices: all_failed,
            processing_time: total_time,
            batch_size: texts.len(),
            success_rate,
        })
    }

    /// Process batch with timeout per item
    pub fn process_batch_with_timeout<T, F>(
        &mut self,
        texts: &[&str],
        processor: F,
        timeout_per_item: Duration,
    ) -> Result<BatchResult<T>>
    where
        F: Fn(&str) -> Result<T>,
    {
        let start_time = Instant::now();
        let mut results = Vec::with_capacity(texts.len());
        let mut failed_indices = Vec::new();

        for (idx, &text) in texts.iter().enumerate() {
            let item_start = Instant::now();

            // Simple timeout simulation (in real implementation, would use proper async/timeout)
            match processor(text) {
                Ok(result) => {
                    if item_start.elapsed() <= timeout_per_item {
                        results.push(result);
                    } else {
                        failed_indices.push((idx, "Timeout".to_string()));
                    }
                }
                Err(e) => {
                    // Convert to unified error for consistent logging
                    let unified_err =
                        processing_errors::batch_processing(1, &format!("item {}: {}", idx, e));
                    failed_indices.push((idx, unified_err.to_string()));
                }
            }
        }

        let processing_time = start_time.elapsed();
        self.metrics
            .record_batch(texts.len(), processing_time, failed_indices.len());
        let success_rate = (texts.len() - failed_indices.len()) as f32 / texts.len() as f32;

        Ok(BatchResult {
            results,
            failed_indices,
            processing_time,
            batch_size: texts.len(),
            success_rate,
        })
    }

    /// Get processing metrics
    pub fn get_metrics(&self) -> &ProcessingMetrics {
        &self.metrics
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = ProcessingMetrics::new();
    }

    /// Get optimal batch size based on historical performance
    pub fn get_optimal_batch_size(&self) -> usize {
        if self.metrics.total_batches == 0 {
            return self.config.default_batch_size;
        }

        // Simple heuristic: find batch size with best throughput
        let avg_time_per_item =
            self.metrics.total_processing_time.as_millis() as f32 / self.metrics.total_items as f32;

        if avg_time_per_item < 50.0 {
            // Fast processing
            self.config.max_batch_size
        } else if avg_time_per_item < 200.0 {
            // Medium processing
            self.config.max_batch_size / 2
        } else {
            // Slow processing
            self.config.default_batch_size
        }
    }
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchProcessorConfig {
    pub max_batch_size: usize,
    pub default_batch_size: usize,
    pub chunk_delay_ms: u64,
    pub enable_metrics: bool,
    pub retry_failed_items: bool,
    pub max_retries: usize,
}

impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            default_batch_size: 8,
            chunk_delay_ms: 10,
            enable_metrics: true,
            retry_failed_items: false,
            max_retries: 3,
        }
    }
}

/// Batch processing result
#[derive(Debug, Clone)]
pub struct BatchResult<T> {
    pub results: Vec<T>,
    pub failed_indices: Vec<(usize, String)>,
    pub processing_time: Duration,
    pub batch_size: usize,
    pub success_rate: f32,
}

impl<T> BatchResult<T> {
    /// Check if batch processing was successful
    pub fn is_success(&self) -> bool {
        self.failed_indices.is_empty()
    }

    /// Get throughput (items per second)
    pub fn get_throughput(&self) -> f32 {
        self.batch_size as f32 / self.processing_time.as_secs_f32()
    }

    /// Get average processing time per item
    pub fn get_avg_time_per_item(&self) -> Duration {
        Duration::from_millis(self.processing_time.as_millis() as u64 / self.batch_size as u64)
    }

    /// Get failure rate
    pub fn get_failure_rate(&self) -> f32 {
        self.failed_indices.len() as f32 / self.batch_size as f32
    }
}

/// Processing metrics for batch processor
#[derive(Debug, Clone)]
pub struct ProcessingMetrics {
    pub total_batches: usize,
    pub total_items: usize,
    pub total_failures: usize,
    pub total_processing_time: Duration,
    pub fastest_batch_time: Duration,
    pub slowest_batch_time: Duration,
    pub batch_size_distribution: HashMap<usize, usize>,
}

impl ProcessingMetrics {
    fn new() -> Self {
        Self {
            total_batches: 0,
            total_items: 0,
            total_failures: 0,
            total_processing_time: Duration::from_millis(0),
            fastest_batch_time: Duration::from_secs(u64::MAX),
            slowest_batch_time: Duration::from_millis(0),
            batch_size_distribution: HashMap::new(),
        }
    }

    fn record_batch(&mut self, batch_size: usize, processing_time: Duration, failures: usize) {
        self.total_batches += 1;
        self.total_items += batch_size;
        self.total_failures += failures;
        self.total_processing_time += processing_time;

        if processing_time < self.fastest_batch_time {
            self.fastest_batch_time = processing_time;
        }
        if processing_time > self.slowest_batch_time {
            self.slowest_batch_time = processing_time;
        }

        *self.batch_size_distribution.entry(batch_size).or_insert(0) += 1;
    }

    /// Get average processing time per batch
    pub fn avg_batch_time(&self) -> Duration {
        if self.total_batches == 0 {
            return Duration::from_millis(0);
        }
        Duration::from_millis(
            self.total_processing_time.as_millis() as u64 / self.total_batches as u64,
        )
    }

    /// Get average processing time per item
    pub fn avg_item_time(&self) -> Duration {
        if self.total_items == 0 {
            return Duration::from_millis(0);
        }
        Duration::from_millis(
            self.total_processing_time.as_millis() as u64 / self.total_items as u64,
        )
    }

    /// Get overall success rate
    pub fn success_rate(&self) -> f32 {
        if self.total_items == 0 {
            return 0.0;
        }
        (self.total_items - self.total_failures) as f32 / self.total_items as f32
    }

    /// Get throughput (items per second)
    pub fn throughput(&self) -> f32 {
        if self.total_processing_time.as_secs_f32() == 0.0 {
            return 0.0;
        }
        self.total_items as f32 / self.total_processing_time.as_secs_f32()
    }
}
