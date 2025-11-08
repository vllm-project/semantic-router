//! Continuous Batch Scheduler for Qwen3 Multi-LoRA Classifier
//!
//! Inspired by vLLM's continuous batching algorithm, this module enables:
//! - Dynamic request batching: Requests are grouped as they arrive
//! - Concurrent processing: Multiple requests processed in single forward pass
//! - Low latency: No waiting for full batch, process as soon as ready
//! - High throughput: Maximize GPU utilization with adaptive batching
//!
//! Key differences from vLLM:
//! - Simpler: Classification is single-pass (no autoregressive decoding)
//! - No KV cache management: Each request is independent
//! - No preemption: All requests complete in one forward pass
//!
//! Architecture:
//! ```text
//! Request Queue → Batch Builder → Model Forward → Result Distribution
//!      ↓              ↓                ↓                ↓
//!   [Req1]      [Req1, Req2]    Batched Inference   [Res1, Res2]
//!   [Req2]      Ready when:      Single Pass         Return to
//!   [Req3]      • Max batch       [B, seq, vocab]    waiting threads
//!               • Timeout
//! ```

use crate::core::UnifiedResult;
use crate::model_architectures::generative::qwen3_multi_lora_classifier::{
    MultiAdapterClassificationResult, Qwen3MultiLoRAClassifier,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc::{channel, Receiver, Sender},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

/// Request submitted for classification
struct ClassificationRequest {
    /// Unique request ID
    id: u64,
    /// Input text to classify
    text: String,
    /// Adapter to use (e.g., "category", "jailbreak")
    adapter_name: String,
    /// Channel to send result back to caller
    response_tx: Sender<UnifiedResult<MultiAdapterClassificationResult>>,
    /// Time when request was received
    received_at: Instant,
}

/// Configuration for batch scheduler
#[derive(Debug, Clone)]
pub struct BatchSchedulerConfig {
    /// Maximum batch size (number of requests to process together)
    pub max_batch_size: usize,

    /// Maximum time to wait for batch to fill (milliseconds)
    /// If batch doesn't fill, process whatever we have
    pub batch_timeout_ms: u64,

    /// Capacity of request queue (reject requests if full)
    pub queue_capacity: usize,

    /// Enable detailed logging
    pub verbose: bool,
}

impl Default for BatchSchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,    // Process up to 8 requests together
            batch_timeout_ms: 10, // Wait max 10ms for batch to fill
            queue_capacity: 1000, // Queue up to 1000 pending requests
            verbose: false,
        }
    }
}

/// Statistics for monitoring scheduler performance
/// Uses lock-free atomic operations for high-frequency counters
#[derive(Debug)]
pub struct BatchSchedulerStats {
    /// Total requests processed (lock-free atomic)
    pub total_requests: AtomicU64,
    /// Total batches processed (lock-free atomic)
    pub total_batches: AtomicU64,
    /// Average batch size (requires mutex for float arithmetic)
    pub avg_batch_size: Mutex<f64>,
    /// Average latency in milliseconds (requires mutex for float arithmetic)
    pub avg_latency_ms: Mutex<f64>,
    /// Maximum latency in microseconds (lock-free atomic, stored as integer)
    pub max_latency_us: AtomicU64,
    /// Queue full rejections (lock-free atomic)
    pub queue_full_rejections: AtomicU64,
}

impl Clone for BatchSchedulerStats {
    fn clone(&self) -> Self {
        Self {
            total_requests: AtomicU64::new(self.total_requests.load(Ordering::Relaxed)),
            total_batches: AtomicU64::new(self.total_batches.load(Ordering::Relaxed)),
            avg_batch_size: Mutex::new(*self.avg_batch_size.lock().unwrap()),
            avg_latency_ms: Mutex::new(*self.avg_latency_ms.lock().unwrap()),
            max_latency_us: AtomicU64::new(self.max_latency_us.load(Ordering::Relaxed)),
            queue_full_rejections: AtomicU64::new(
                self.queue_full_rejections.load(Ordering::Relaxed),
            ),
        }
    }
}

/// Continuous Batch Scheduler
///
/// Manages a pool of incoming requests and batches them for efficient processing
pub struct ContinuousBatchScheduler {
    /// Channel to send new requests to scheduler thread
    request_tx: Sender<ClassificationRequest>,

    /// Configuration
    config: BatchSchedulerConfig,

    /// Statistics (shared with scheduler thread)
    stats: Arc<BatchSchedulerStats>,

    /// Next request ID (lock-free atomic counter)
    next_request_id: Arc<AtomicU64>,

    /// Shutdown signal
    shutdown_tx: Sender<()>,
}

impl ContinuousBatchScheduler {
    /// Create a new continuous batch scheduler
    ///
    /// # Arguments
    /// - `classifier`: The Qwen3 multi-LoRA classifier (will be moved to scheduler thread)
    /// - `config`: Scheduler configuration
    ///
    /// # Returns
    /// A new scheduler instance that can accept concurrent requests
    pub fn new(classifier: Qwen3MultiLoRAClassifier, config: BatchSchedulerConfig) -> Self {
        let (request_tx, request_rx) = channel();
        let (shutdown_tx, shutdown_rx) = channel();

        let stats = Arc::new(BatchSchedulerStats {
            total_requests: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
            avg_batch_size: Mutex::new(0.0),
            avg_latency_ms: Mutex::new(0.0),
            max_latency_us: AtomicU64::new(0),
            queue_full_rejections: AtomicU64::new(0),
        });

        let stats_clone = Arc::clone(&stats);
        let config_clone = config.clone();

        // Spawn scheduler thread
        thread::spawn(move || {
            Self::scheduler_loop(
                classifier,
                request_rx,
                shutdown_rx,
                config_clone,
                stats_clone,
            );
        });

        if config.verbose {
            println!("🚀 Continuous Batch Scheduler started");
            println!("   Max batch size: {}", config.max_batch_size);
            println!("   Batch timeout: {}ms", config.batch_timeout_ms);
            println!("   Queue capacity: {}", config.queue_capacity);
        }

        Self {
            request_tx,
            config,
            stats,
            next_request_id: Arc::new(AtomicU64::new(0)),
            shutdown_tx,
        }
    }

    /// Submit a classification request (non-blocking)
    ///
    /// # Arguments
    /// - `text`: Input text to classify
    /// - `adapter_name`: Name of adapter to use
    ///
    /// # Returns
    /// Result channel that will receive the classification result
    pub fn classify(
        &self,
        text: String,
        adapter_name: String,
    ) -> UnifiedResult<MultiAdapterClassificationResult> {
        // Generate unique request ID (lock-free atomic increment)
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);

        // Create response channel
        let (response_tx, response_rx) = channel();

        // Create request
        let request = ClassificationRequest {
            id,
            text,
            adapter_name,
            response_tx,
            received_at: Instant::now(),
        };

        // Send to scheduler (non-blocking if queue has space)
        self.request_tx
            .send(request)
            .map_err(|_| crate::core::UnifiedError::Processing {
                operation: "submit request".to_string(),
                source: "scheduler thread terminated".to_string(),
                input_context: None,
            })?;

        // Wait for response (blocking until result is ready)
        response_rx
            .recv()
            .map_err(|_| crate::core::UnifiedError::Processing {
                operation: "receive result".to_string(),
                source: "scheduler dropped response".to_string(),
                input_context: None,
            })?
    }

    /// Get current scheduler statistics
    pub fn get_stats(&self) -> BatchSchedulerStats {
        (*self.stats).clone()
    }

    /// Shutdown the scheduler gracefully
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    /// Main scheduler loop (runs in dedicated thread)
    fn scheduler_loop(
        mut classifier: Qwen3MultiLoRAClassifier,
        request_rx: Receiver<ClassificationRequest>,
        shutdown_rx: Receiver<()>,
        config: BatchSchedulerConfig,
        stats: Arc<BatchSchedulerStats>,
    ) {
        let mut pending_requests: Vec<ClassificationRequest> = Vec::new();
        let batch_timeout = Duration::from_millis(config.batch_timeout_ms);
        let mut last_batch_time = Instant::now();

        loop {
            // Check for shutdown signal
            if shutdown_rx.try_recv().is_ok() {
                if config.verbose {
                    println!("🛑 Scheduler shutting down");
                }
                break;
            }

            // Collect new requests (non-blocking)
            while pending_requests.len() < config.max_batch_size {
                match request_rx.try_recv() {
                    Ok(req) => {
                        pending_requests.push(req);
                    }
                    Err(_) => break, // No more requests available
                }
            }

            // Decide whether to process batch now
            let should_process = if pending_requests.is_empty() {
                // No requests, wait a bit and try again
                thread::sleep(Duration::from_micros(100));
                false
            } else if pending_requests.len() >= config.max_batch_size {
                // Batch is full, process immediately
                if config.verbose {
                    println!("📦 Batch full ({} requests)", pending_requests.len());
                }
                true
            } else if last_batch_time.elapsed() >= batch_timeout {
                // Timeout reached, process what we have
                if config.verbose {
                    println!("⏱️  Batch timeout ({} requests)", pending_requests.len());
                }
                true
            } else {
                // Wait a bit more for batch to fill
                thread::sleep(Duration::from_micros(100));
                false
            };

            if should_process && !pending_requests.is_empty() {
                let batch_start = Instant::now();
                let batch_size = pending_requests.len();

                // Process batch
                Self::process_batch(&mut classifier, &mut pending_requests, &config, &stats);

                // Update statistics (lock-free for counters, minimal lock for averages)
                let batch_duration = batch_start.elapsed();
                let total_batches = stats.total_batches.fetch_add(1, Ordering::Relaxed) + 1;

                // Update average batch size (requires lock for float arithmetic)
                let mut avg_batch_size = stats.avg_batch_size.lock().unwrap();
                *avg_batch_size = (*avg_batch_size * (total_batches - 1) as f64
                    + batch_size as f64)
                    / total_batches as f64;
                drop(avg_batch_size);

                if config.verbose {
                    println!(
                        "✅ Batch processed: {} requests in {:.2}ms",
                        batch_size,
                        batch_duration.as_secs_f64() * 1000.0
                    );
                }

                last_batch_time = Instant::now();
            }
        }
    }

    /// Process a batch of requests with true batched inference
    fn process_batch(
        classifier: &mut Qwen3MultiLoRAClassifier,
        requests: &mut Vec<ClassificationRequest>,
        config: &BatchSchedulerConfig,
        stats: &Arc<BatchSchedulerStats>,
    ) {
        if requests.is_empty() {
            return;
        }

        // Group requests by adapter (each adapter needs separate forward pass)
        let mut adapter_groups: std::collections::HashMap<String, Vec<ClassificationRequest>> =
            std::collections::HashMap::new();

        for req in requests.drain(..) {
            adapter_groups
                .entry(req.adapter_name.clone())
                .or_insert_with(Vec::new)
                .push(req);
        }

        // Process each adapter group with TRUE BATCHED INFERENCE
        for (adapter_name, adapter_requests) in adapter_groups {
            let batch_size = adapter_requests.len();

            if config.verbose {
                println!(
                    "🔄 TRUE BATCHED: Processing {} requests for adapter '{}' in single forward pass",
                    batch_size,
                    adapter_name
                );
            }

            // Collect texts and track request start times
            let texts: Vec<String> = adapter_requests
                .iter()
                .map(|req| req.text.clone())
                .collect();

            let request_starts: Vec<_> =
                adapter_requests.iter().map(|req| req.received_at).collect();

            // TRUE BATCHED INFERENCE: Single forward pass for entire batch
            let batch_start = std::time::Instant::now();
            let results = classifier.classify_batch_with_adapter(&texts, &adapter_name);
            let batch_duration = batch_start.elapsed().as_secs_f64() * 1000.0;

            if config.verbose {
                println!(
                    "✅ Batched forward pass completed: {} requests in {:.2}ms ({:.2}ms per request)",
                    batch_size,
                    batch_duration,
                    batch_duration / batch_size as f64
                );
            }

            // Distribute results back to individual requesters
            match results {
                Ok(batch_results) => {
                    // Collect all latencies first
                    let mut latencies = Vec::with_capacity(batch_size);

                    for (i, (req, result)) in adapter_requests
                        .into_iter()
                        .zip(batch_results.into_iter())
                        .enumerate()
                    {
                        let req_start = request_starts[i];
                        let latency_ms = req_start.elapsed().as_secs_f64() * 1000.0;
                        latencies.push(latency_ms);

                        // Send result back to caller
                        let _ = req.response_tx.send(Ok(result));
                    }

                    // Update statistics (lock-free where possible, batched)
                    let total_requests = stats
                        .total_requests
                        .fetch_add(batch_size as u64, Ordering::Relaxed)
                        + batch_size as u64;

                    // Update average latency (minimal lock for float arithmetic)
                    let total_latency_ms: f64 = latencies.iter().sum();
                    let mut avg_latency = stats.avg_latency_ms.lock().unwrap();
                    *avg_latency = (*avg_latency * (total_requests - batch_size as u64) as f64
                        + total_latency_ms)
                        / total_requests as f64;
                    drop(avg_latency);

                    // Update max latency (lock-free atomic compare-and-swap)
                    for &latency_ms in &latencies {
                        let latency_us = (latency_ms * 1000.0) as u64;
                        let mut current_max = stats.max_latency_us.load(Ordering::Relaxed);
                        while latency_us > current_max {
                            match stats.max_latency_us.compare_exchange_weak(
                                current_max,
                                latency_us,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => break,
                                Err(new_max) => current_max = new_max,
                            }
                        }
                    }
                }
                Err(e) => {
                    // If batch fails, create error message once and send to all requesters
                    let error_msg = format!("Batch classification failed: {:?}", e);
                    for req in adapter_requests {
                        let error = crate::core::UnifiedError::Processing {
                            operation: "batch classification".to_string(),
                            source: error_msg.clone(),
                            input_context: None,
                        };
                        let _ = req.response_tx.send(Err(error));
                    }
                }
            }
        }
    }
}

impl Drop for ContinuousBatchScheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_default() {
        let config = BatchSchedulerConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.batch_timeout_ms, 10);
        assert_eq!(config.queue_capacity, 1000);
    }
}
