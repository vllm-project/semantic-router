//! Continuous Batching Scheduler for Embedding Models
//!
//! Inspired by vLLM's continuous batching, this module enables:
//! - Dynamic request batching: Requests are grouped as they arrive
//! - Concurrent processing: Multiple requests processed in single forward pass
//! - Low latency: No waiting for full batch, process as soon as ready
//! - High throughput: Maximize GPU utilization with adaptive batching
//!
//! Key differences from text generation:
//! - Simpler: Embedding is single-pass (no autoregressive decoding)
//! - No KV cache management: Each request is independent
//! - No preemption: All requests complete in one forward pass
//!
//! Architecture:
//! ```text
//! Request Queue → Batch Builder → Model Forward → Result Distribution
//!      ↓              ↓                ↓                ↓
//!   [Req1]      [Req1, Req2]    Batched Inference   [Res1, Res2]
//!   [Req2]      Ready when:      Single Pass         Return to
//!   [Req3]      • Max batch       [B, hidden_size]   waiting threads
//!               • Timeout
//! ```

use crate::core::{UnifiedError, UnifiedResult};
use crate::model_architectures::embedding::qwen3_embedding::Qwen3EmbeddingModel;
use candle_core::Tensor;
use crossbeam_channel::{bounded, select, tick, unbounded, Receiver, Sender};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

/// Request submitted for embedding
struct EmbeddingRequest {
    /// Unique request ID
    id: u64,
    /// Input token IDs as raw vec (tensors created on scheduler thread to avoid CUDA context issues)
    input_ids_raw: Vec<u32>,
    /// Attention mask as raw vec
    attention_mask_raw: Vec<u32>,
    /// Sequence length (for batching optimization)
    seq_len: usize,
    /// Channel to send result back to caller (now sends Vec<f32> to avoid CUDA context issues)
    response_tx: Sender<UnifiedResult<Vec<f32>>>,
    /// Time when request was received
    received_at: Instant,
}

/// Configuration for batch scheduler
#[derive(Debug, Clone)]
pub struct ContinuousBatchConfig {
    /// Maximum batch size (number of requests to process together)
    pub max_batch_size: usize,

    /// Maximum time to wait for batch to fill (milliseconds)
    /// If batch doesn't fill, process whatever we have
    pub max_wait_time_ms: u64,

    /// Minimum batch size before processing (set to 1 for immediate processing)
    pub min_batch_size: usize,

    /// Enable dynamic batching (vs fixed batching)
    pub enable_dynamic: bool,

    /// Maximum sequence length difference to batch together
    /// Helps avoid excessive padding overhead
    pub max_seq_len_diff: usize,

    /// Enable detailed logging
    pub verbose: bool,
}

impl Default for ContinuousBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,  // Process up to 32 requests together
            max_wait_time_ms: 5, // Wait max 5ms for batch to fill
            min_batch_size: 1,   // Process immediately if any request arrives
            enable_dynamic: true,
            max_seq_len_diff: 128, // Group similar lengths
            verbose: false,
        }
    }
}

/// Statistics for monitoring scheduler performance
#[derive(Debug)]
pub struct BatchSchedulerStats {
    /// Total batches processed (lock-free atomic)
    pub total_batches: AtomicU64,
    /// Average batch size (requires mutex for float arithmetic)
    pub avg_batch_size: Mutex<f64>,
    /// Maximum latency in microseconds (lock-free atomic, stored as integer)
    pub max_latency_us: AtomicU64,
    /// Combined statistics requiring atomic updates (protected by single mutex)
    pub inner: Mutex<StatsInner>,
}

/// Inner stats that must be updated atomically together to avoid race conditions
#[derive(Debug, Clone, Copy)]
pub struct StatsInner {
    /// Total requests processed
    pub total_requests: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
}

impl Clone for BatchSchedulerStats {
    fn clone(&self) -> Self {
        Self {
            total_batches: AtomicU64::new(self.total_batches.load(Ordering::Relaxed)),
            avg_batch_size: Mutex::new(*self.avg_batch_size.lock().unwrap()),
            max_latency_us: AtomicU64::new(self.max_latency_us.load(Ordering::Relaxed)),
            inner: Mutex::new(*self.inner.lock().unwrap()),
        }
    }
}

/// Continuous Batch Scheduler
///
/// Manages a pool of incoming requests and batches them for efficient processing
pub struct ContinuousBatchScheduler {
    /// Channel to send new requests to scheduler thread
    request_tx: Sender<EmbeddingRequest>,

    /// Configuration
    config: ContinuousBatchConfig,

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
    /// - `model`: The Qwen3 embedding model (will be moved to scheduler thread)
    /// - `config`: Scheduler configuration
    ///
    /// # Returns
    /// A new scheduler instance that can accept concurrent requests
    pub fn new(model: Qwen3EmbeddingModel, config: ContinuousBatchConfig) -> Self {
        // Use unbounded channels for flexibility (crossbeam-channel)
        let (request_tx, request_rx) = unbounded();
        let (shutdown_tx, shutdown_rx) = bounded(1);

        let stats = Arc::new(BatchSchedulerStats {
            total_batches: AtomicU64::new(0),
            avg_batch_size: Mutex::new(0.0),
            max_latency_us: AtomicU64::new(0),
            inner: Mutex::new(StatsInner {
                total_requests: 0,
                avg_latency_ms: 0.0,
            }),
        });

        let stats_clone = Arc::clone(&stats);
        let config_clone = config.clone();

        // Spawn scheduler thread
        thread::spawn(move || {
            Self::scheduler_loop(model, request_rx, shutdown_rx, config_clone, stats_clone);
        });

        if config.verbose {
            println!("🚀 Continuous Batch Scheduler started (Embeddings)");
            println!("   Max batch size: {}", config.max_batch_size);
            println!("   Batch timeout: {}ms", config.max_wait_time_ms);
        }

        Self {
            request_tx,
            config,
            stats,
            next_request_id: Arc::new(AtomicU64::new(0)),
            shutdown_tx,
        }
    }

    /// Submit an embedding request using raw token vectors (avoids CUDA context issues)
    ///
    /// # Arguments
    /// - `input_ids_raw`: Token IDs as Vec<u32>
    /// - `attention_mask_raw`: Attention mask as Vec<u32>
    ///
    /// # Returns
    /// Embedding tensor [1, hidden_size]
    ///
    /// # Note
    /// This method accepts raw vectors instead of Tensors to avoid CUDA context errors.
    /// Tensors are created on the scheduler thread which owns the model and CUDA context.
    pub fn embed_from_raw(
        &self,
        input_ids_raw: Vec<u32>,
        attention_mask_raw: Vec<u32>,
    ) -> UnifiedResult<Vec<f32>> {
        // Generate unique request ID (lock-free atomic increment)
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);

        // Get sequence length
        let seq_len = input_ids_raw.len();

        // Create response channel (bounded for backpressure)
        let (response_tx, response_rx) = bounded(1);

        // Create request
        let request = EmbeddingRequest {
            id,
            input_ids_raw,
            attention_mask_raw,
            seq_len,
            response_tx,
            received_at: Instant::now(),
        };

        // Send to scheduler (non-blocking if queue has space)
        self.request_tx
            .send(request)
            .map_err(|_| UnifiedError::Processing {
                operation: "submit request".to_string(),
                source: "scheduler thread terminated".to_string(),
                input_context: None,
            })?;

        // Wait for response (blocking until result is ready)
        // Response is now Vec<f32> to avoid CUDA context issues
        response_rx.recv().map_err(|_| UnifiedError::Processing {
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
    /// Uses crossbeam-channel's select! for efficient multi-channel wakeup
    fn scheduler_loop(
        model: Qwen3EmbeddingModel,
        request_rx: Receiver<EmbeddingRequest>,
        shutdown_rx: Receiver<()>,
        config: ContinuousBatchConfig,
        stats: Arc<BatchSchedulerStats>,
    ) {
        let mut pending_requests: Vec<EmbeddingRequest> = Vec::new();
        let batch_timeout = Duration::from_millis(config.max_wait_time_ms);

        // Create a ticker for timeout events (efficient wakeup mechanism)
        let ticker = tick(batch_timeout);

        loop {
            // Wait for events using select! (no busy-waiting, no recv_timeout polling)
            select! {
                recv(shutdown_rx) -> _ => {
                    // Shutdown signal received
                    if config.verbose {
                        println!("🛑 Embedding scheduler shutting down");
                    }
                    break;
                }
                recv(request_rx) -> msg => {
                    // New request arrived
                    match msg {
                        Ok(req) => {
                            pending_requests.push(req);

                            // Drain any additional pending requests (non-blocking)
                            while pending_requests.len() < config.max_batch_size {
                                match request_rx.try_recv() {
                                    Ok(req) => pending_requests.push(req),
                                    Err(_) => break,
                                }
                            }

                            // If batch is full, process immediately
                            if pending_requests.len() >= config.max_batch_size {
                                if config.verbose {
                                    println!("📦 Batch full ({} requests)", pending_requests.len());
                                }
                                Self::process_and_update_stats(
                                    &model,
                                    &mut pending_requests,
                                    &config,
                                    &stats,
                                );
                            }
                        }
                        Err(_) => {
                            // Channel disconnected, shutdown
                            break;
                        }
                    }
                }
                recv(ticker) -> _ => {
                    // Timeout tick - process pending requests if any
                    if !pending_requests.is_empty() {
                        if config.verbose {
                            println!("⏱️  Batch timeout ({} requests)", pending_requests.len());
                        }
                        Self::process_and_update_stats(
                            &model,
                            &mut pending_requests,
                            &config,
                            &stats,
                        );
                    }
                }
            }
        }
    }

    /// Helper to process batch and update statistics
    fn process_and_update_stats(
        model: &Qwen3EmbeddingModel,
        pending_requests: &mut Vec<EmbeddingRequest>,
        config: &ContinuousBatchConfig,
        stats: &Arc<BatchSchedulerStats>,
    ) {
        let batch_start = Instant::now();
        let batch_size = pending_requests.len();

        // Process batch
        Self::process_batch(model, pending_requests, config, stats);

        // Update statistics (lock-free for counters, minimal lock for averages)
        let batch_duration = batch_start.elapsed();
        let total_batches = stats.total_batches.fetch_add(1, Ordering::Relaxed) + 1;

        // Update average batch size (requires lock for float arithmetic)
        let mut avg_batch_size = stats.avg_batch_size.lock().unwrap();
        *avg_batch_size = (*avg_batch_size * (total_batches - 1) as f64 + batch_size as f64)
            / total_batches as f64;
        drop(avg_batch_size);

        if config.verbose {
            println!(
                "✅ Batch processed: {} requests in {:.2}ms",
                batch_size,
                batch_duration.as_secs_f64() * 1000.0
            );
        }
    }

    /// Process a batch of requests with true batched inference
    fn process_batch(
        model: &Qwen3EmbeddingModel,
        requests: &mut Vec<EmbeddingRequest>,
        config: &ContinuousBatchConfig,
        stats: &Arc<BatchSchedulerStats>,
    ) {
        if requests.is_empty() {
            return;
        }

        let batch_size = requests.len();

        if config.verbose {
            println!(
                "🔄 TRUE BATCHED: Processing {} requests in single forward pass",
                batch_size
            );
        }

        // Find max sequence length in this batch
        let max_seq_len = requests.iter().map(|r| r.seq_len).max().unwrap();

        // Collect input_ids and attention_masks with padding (working with raw Vec<u32>)
        // IMPORTANT: drain() immediately takes ownership of ALL requests from the vector,
        // so we must ensure every request gets either a result or an error response.
        let mut batch_requests = Vec::new();
        let mut all_input_ids = Vec::new();
        let mut all_attention_masks = Vec::new();

        // Drain all requests at once (they're now owned by us - must handle each one)
        for request in requests.drain(..) {
            let seq_len = request.seq_len;

            // Pad to max_seq_len if needed (left padding for Qwen3)
            // Working with raw vectors - no CUDA context issues!
            let padded_input_ids = if seq_len < max_seq_len {
                let pad_len = max_seq_len - seq_len;
                let mut padded = vec![0u32; pad_len]; // Padding token ID = 0
                padded.extend_from_slice(&request.input_ids_raw);
                padded
            } else {
                request.input_ids_raw.clone()
            };

            let padded_attention_mask = if seq_len < max_seq_len {
                let pad_len = max_seq_len - seq_len;
                let mut padded = vec![0u32; pad_len]; // Padding mask = 0
                padded.extend_from_slice(&request.attention_mask_raw);
                padded
            } else {
                request.attention_mask_raw.clone()
            };

            // Add to batch
            all_input_ids.push(padded_input_ids);
            all_attention_masks.push(padded_attention_mask);
            batch_requests.push(request);
        }

        // If no requests were collected, return early
        if batch_requests.is_empty() {
            return;
        }

        let actual_batch_size = batch_requests.len();

        // Create batched tensors
        let device = &model.device();
        let batched_input_ids = match Tensor::new(all_input_ids, device) {
            Ok(t) => t,
            Err(e) => {
                // Send error to ALL requests (we own them now via drain)
                for req in batch_requests {
                    let _ = req.response_tx.send(Err(UnifiedError::Processing {
                        operation: "create batched input_ids".to_string(),
                        source: e.to_string(),
                        input_context: None,
                    }));
                }
                return;
            }
        };

        let batched_attention_mask = match Tensor::new(all_attention_masks, device) {
            Ok(t) => t,
            Err(e) => {
                // Send error to ALL requests (we own them now via drain)
                for req in batch_requests {
                    let _ = req.response_tx.send(Err(UnifiedError::Processing {
                        operation: "create batched attention_mask".to_string(),
                        source: e.to_string(),
                        input_context: None,
                    }));
                }
                return;
            }
        };

        // TRUE BATCHED INFERENCE: Single forward pass for entire batch
        let batch_start = Instant::now();
        let batch_result = model.embedding_forward(&batched_input_ids, &batched_attention_mask);
        let batch_duration = batch_start.elapsed().as_secs_f64() * 1000.0;

        if config.verbose {
            println!(
                "✅ Batched forward pass completed: {} requests in {:.2}ms ({:.2}ms per request)",
                actual_batch_size,
                batch_duration,
                batch_duration / actual_batch_size as f64
            );
        }

        // Distribute results back to individual requesters
        match batch_result {
            Ok(batch_embeddings) => {
                // CRITICAL: Convert tensor to Vec<Vec<f32>> ON scheduler thread (where CUDA context is valid)
                let embeddings_vecs: Vec<Vec<f32>> = match batch_embeddings.to_vec2::<f32>() {
                    Ok(vecs) => vecs,
                    Err(e) => {
                        // Conversion failed - send error to ALL requesters
                        for req in batch_requests {
                            let _ = req.response_tx.send(Err(UnifiedError::Processing {
                                operation: "convert batch tensor to vec".to_string(),
                                source: e.to_string(),
                                input_context: None,
                            }));
                        }
                        return;
                    }
                };

                // Collect all latencies first
                let mut latencies = Vec::with_capacity(actual_batch_size);

                // Send individual embedding vectors to each requester
                for (i, req) in batch_requests.into_iter().enumerate() {
                    let req_start = req.received_at;
                    let latency_ms = req_start.elapsed().as_secs_f64() * 1000.0;
                    latencies.push(latency_ms);

                    // Send embedding vector (no more tensor - no CUDA context issues!)
                    if i < embeddings_vecs.len() {
                        let _ = req.response_tx.send(Ok(embeddings_vecs[i].clone()));
                    } else {
                        let _ = req.response_tx.send(Err(UnifiedError::Processing {
                            operation: format!("extract embedding {}", i),
                            source: "index out of bounds".to_string(),
                            input_context: None,
                        }));
                    }
                }

                // Update statistics atomically (single mutex prevents race conditions)
                let total_latency_ms: f64 = latencies.iter().sum();
                let mut inner = stats.inner.lock().unwrap();
                inner.total_requests += actual_batch_size as u64;
                inner.avg_latency_ms = (inner.avg_latency_ms
                    * (inner.total_requests - actual_batch_size as u64) as f64
                    + total_latency_ms)
                    / inner.total_requests as f64;
                drop(inner);

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
                // If batch fails, send error to ALL requesters
                let error_msg = format!("Batch embedding failed: {:?}", e);
                for req in batch_requests {
                    let error = UnifiedError::Processing {
                        operation: "batch embedding".to_string(),
                        source: error_msg.clone(),
                        input_context: None,
                    };
                    let _ = req.response_tx.send(Err(error));
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

/// Extract individual embeddings from batched results
pub fn unbatch_embeddings(
    batch_embeddings: &Tensor,
    batch_size: usize,
) -> Result<Vec<Tensor>, UnifiedError> {
    let mut embeddings = Vec::new();

    for i in 0..batch_size {
        let embedding = batch_embeddings
            .get(i)
            .map_err(|e| UnifiedError::Processing {
                operation: format!("extract embedding {}", i),
                source: e.to_string(),
                input_context: None,
            })?;
        embeddings.push(embedding);
    }

    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_batch_config() {
        let config = ContinuousBatchConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_wait_time_ms, 5);
        assert!(config.enable_dynamic);
    }
}
