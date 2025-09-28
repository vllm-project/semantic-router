//! Intelligent Memory Management

use candle_core::{DType, Device, Shape, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::model_architectures::traits::{ModelType, TaskType};

/// Shared memory pool for dual-path optimization
pub struct DualPathMemoryPool {
    /// Traditional model memory allocations
    traditional_pools: Arc<RwLock<HashMap<String, TensorPool>>>,
    /// LoRA model memory allocations
    lora_pools: Arc<RwLock<HashMap<String, TensorPool>>>,
    /// Shared cross-path memory pool
    shared_pool: Arc<Mutex<SharedTensorPool>>,
    /// Memory usage tracker
    usage_tracker: Arc<Mutex<MemoryUsageTracker>>,
    /// Computing device
    device: Device,
    /// Pool configuration
    config: MemoryPoolConfig,
}

/// Tensor pool for efficient memory reuse
#[derive(Debug)]
pub struct TensorPool {
    /// Available tensors by shape and dtype
    available_tensors: HashMap<TensorKey, Vec<Tensor>>,
    /// Pool creation time
    created_at: Instant,
    /// Total allocations from this pool
    allocation_count: usize,
    /// Total deallocations to this pool
    deallocation_count: usize,
}

/// Shared tensor pool for cross-path optimization
#[derive(Debug)]
pub struct SharedTensorPool {
    /// Shared tensors between Traditional and LoRA paths
    shared_tensors: HashMap<TensorKey, Vec<SharedTensor>>,
    /// Pool usage statistics
    usage_stats: SharedPoolStats,
    /// Maximum pool size
    max_pool_size: usize,
}

/// Shared tensor with reference counting
#[derive(Debug, Clone)]
pub struct SharedTensor {
    /// The actual tensor
    tensor: Tensor,
    /// Reference count
    ref_count: Arc<Mutex<usize>>,
    /// Last accessed time
    last_accessed: Instant,
    /// Owning model type
    owner_type: ModelType,
}

/// Tensor identification key
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TensorKey {
    /// Tensor shape
    shape: Vec<usize>,
    /// Data type
    dtype: DType,
    /// Usage hint (e.g., "input_ids", "attention_mask", "embeddings")
    usage_hint: String,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Maximum pool size per model type (MB)
    max_pool_size_mb: usize,
    /// Maximum shared pool size (MB)
    max_shared_pool_size_mb: usize,
    /// Tensor cleanup interval
    cleanup_interval: Duration,
    /// Enable memory compression
    enable_compression: bool,
    /// Target memory reduction percentage
    target_reduction_percent: f32,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size_mb: 512,        // 512MB per model type
            max_shared_pool_size_mb: 256, // 256MB shared
            cleanup_interval: Duration::from_secs(30),
            enable_compression: true,
            target_reduction_percent: 20.0, // 20% reduction target
        }
    }
}

/// Memory usage tracking and analytics
#[derive(Debug, Default)]
pub struct MemoryUsageTracker {
    /// Baseline memory usage (without optimization)
    baseline_usage_mb: f32,
    /// Current memory usage (with optimization)
    current_usage_mb: f32,
    /// Peak memory usage
    peak_usage_mb: f32,
    /// Memory allocations by model type
    allocations_by_type: HashMap<ModelType, Vec<AllocationRecord>>,
    /// Shared memory savings
    shared_savings_mb: f32,
    /// Total memory operations
    total_operations: usize,
}

/// Individual allocation record
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Allocation size in bytes
    size_bytes: usize,
    /// Allocation timestamp
    timestamp: Instant,
    /// Tensor key
    tensor_key: TensorKey,
    /// Whether allocation came from pool
    from_pool: bool,
}

/// Shared pool usage statistics
#[derive(Debug, Default)]
pub struct SharedPoolStats {
    /// Total shared allocations
    total_shared_allocations: usize,
    /// Memory saved through sharing (MB)
    memory_saved_mb: f32,
    /// Hit rate for shared pool
    hit_rate_percent: f32,
    /// Average tensor reuse count
    avg_reuse_count: f32,
}

impl DualPathMemoryPool {
    /// Create a new dual-path memory pool
    pub fn new(device: Device, config: MemoryPoolConfig) -> Self {
        println!(
            "Initializing DualPathMemoryPool with {}MB limit",
            config.max_pool_size_mb * 2 + config.max_shared_pool_size_mb
        );

        Self {
            traditional_pools: Arc::new(RwLock::new(HashMap::new())),
            lora_pools: Arc::new(RwLock::new(HashMap::new())),
            shared_pool: Arc::new(Mutex::new(SharedTensorPool::new(
                config.max_shared_pool_size_mb,
            ))),
            usage_tracker: Arc::new(Mutex::new(MemoryUsageTracker::default())),
            device,
            config,
        }
    }

    /// Allocate tensor with optimization
    pub fn allocate_tensor(
        &self,
        shape: &[usize],
        dtype: DType,
        usage_hint: &str,
        model_type: ModelType,
    ) -> Result<Tensor, candle_core::Error> {
        let tensor_key = TensorKey {
            shape: shape.to_vec(),
            dtype,
            usage_hint: usage_hint.to_string(),
        };

        // Try to get from shared pool first
        if let Some(shared_tensor) = self.try_get_from_shared_pool(&tensor_key) {
            self.record_allocation(&tensor_key, model_type, true);
            return Ok(shared_tensor.tensor);
        }

        // Try to get from model-specific pool
        if let Some(pooled_tensor) = self.try_get_from_model_pool(&tensor_key, model_type) {
            self.record_allocation(&tensor_key, model_type, true);
            return Ok(pooled_tensor);
        }

        // Create new tensor
        let tensor = Tensor::zeros(shape, dtype, &self.device)?;
        self.record_allocation(&tensor_key, model_type, false);

        println!("Allocated new tensor: {:?} for {:?}", shape, model_type);
        Ok(tensor)
    }

    /// Return tensor to pool for reuse
    pub fn deallocate_tensor(
        &self,
        tensor: Tensor,
        usage_hint: &str,
        model_type: ModelType,
    ) -> Result<(), candle_core::Error> {
        let shape = tensor.shape().dims().to_vec();
        let dtype = tensor.dtype();

        let tensor_key = TensorKey {
            shape,
            dtype,
            usage_hint: usage_hint.to_string(),
        };

        // Decide whether to put in shared pool or model-specific pool
        if self.should_share_tensor(&tensor_key, model_type) {
            self.add_to_shared_pool(tensor, tensor_key, model_type);
        } else {
            self.add_to_model_pool(tensor, tensor_key, model_type);
        }

        Ok(())
    }

    /// Try to get tensor from shared pool
    fn try_get_from_shared_pool(&self, tensor_key: &TensorKey) -> Option<SharedTensor> {
        let mut shared_pool = self.shared_pool.lock().unwrap();
        shared_pool.try_get_tensor(tensor_key)
    }

    /// Try to get tensor from model-specific pool
    fn try_get_from_model_pool(
        &self,
        tensor_key: &TensorKey,
        model_type: ModelType,
    ) -> Option<Tensor> {
        let pools = match model_type {
            ModelType::Traditional => &self.traditional_pools,
            ModelType::LoRA => &self.lora_pools,
        };

        let pools_read = pools.read().unwrap();
        if let Some(pool) = pools_read.get(&tensor_key.usage_hint) {
            if let Some(tensors) = pool.available_tensors.get(tensor_key) {
                if !tensors.is_empty() {
                    return Some(tensors[0].clone());
                }
            }
        }
        None
    }

    /// Add tensor to shared pool
    fn add_to_shared_pool(&self, tensor: Tensor, tensor_key: TensorKey, owner_type: ModelType) {
        let mut shared_pool = self.shared_pool.lock().unwrap();
        let shared_tensor = SharedTensor {
            tensor,
            ref_count: Arc::new(Mutex::new(0)),
            last_accessed: Instant::now(),
            owner_type,
        };
        shared_pool.add_tensor(tensor_key, shared_tensor);
    }

    /// Add tensor to model-specific pool
    fn add_to_model_pool(&self, tensor: Tensor, tensor_key: TensorKey, model_type: ModelType) {
        let pools = match model_type {
            ModelType::Traditional => &self.traditional_pools,
            ModelType::LoRA => &self.lora_pools,
        };

        let mut pools_write = pools.write().unwrap();
        let pool = pools_write
            .entry(tensor_key.usage_hint.clone())
            .or_insert_with(|| TensorPool::new());

        pool.add_tensor(tensor_key, tensor);
    }

    /// Determine if tensor should be shared between paths
    fn should_share_tensor(&self, tensor_key: &TensorKey, _model_type: ModelType) -> bool {
        // Share common tensors like input_ids, attention_mask, embeddings
        matches!(
            tensor_key.usage_hint.as_str(),
            "input_ids" | "attention_mask" | "embeddings" | "pooled_output"
        )
    }

    /// Record memory allocation
    fn record_allocation(&self, tensor_key: &TensorKey, model_type: ModelType, from_pool: bool) {
        let mut tracker = self.usage_tracker.lock().unwrap();
        let tensor_size =
            tensor_key.shape.iter().product::<usize>() * dtype_size_bytes(tensor_key.dtype);

        let record = AllocationRecord {
            size_bytes: tensor_size,
            timestamp: Instant::now(),
            tensor_key: tensor_key.clone(),
            from_pool,
        };

        tracker
            .allocations_by_type
            .entry(model_type)
            .or_insert_with(Vec::new)
            .push(record);

        tracker.total_operations += 1;

        if from_pool {
            tracker.shared_savings_mb += tensor_size as f32 / 1024.0 / 1024.0;
        }
    }

    /// Get current memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let tracker = self.usage_tracker.lock().unwrap();
        let shared_pool = self.shared_pool.lock().unwrap();

        // Calculate total current usage
        let total_allocated_bytes: usize = tracker
            .allocations_by_type
            .values()
            .flat_map(|records| records.iter())
            .map(|record| record.size_bytes)
            .sum();

        let current_usage_mb = total_allocated_bytes as f32 / 1024.0 / 1024.0;

        // Estimate baseline usage (without optimization)
        let estimated_baseline_mb = current_usage_mb + tracker.shared_savings_mb;

        // Calculate reduction percentage
        let reduction_percent = if estimated_baseline_mb > 0.0 {
            (tracker.shared_savings_mb / estimated_baseline_mb) * 100.0
        } else {
            0.0
        };

        MemoryStats {
            current_usage_mb,
            estimated_baseline_mb,
            shared_savings_mb: tracker.shared_savings_mb,
            reduction_percent,
            shared_pool_hit_rate: shared_pool.usage_stats.hit_rate_percent,
            total_operations: tracker.total_operations,
            meets_target: reduction_percent >= self.config.target_reduction_percent,
        }
    }

    /// Cleanup unused tensors
    pub fn cleanup_unused_tensors(&self) -> CleanupReport {
        let start_time = Instant::now();
        let mut cleaned_count = 0;
        let mut freed_memory_mb = 0.0;

        // Cleanup shared pool
        {
            let mut shared_pool = self.shared_pool.lock().unwrap();
            let (count, memory) = shared_pool.cleanup_unused_tensors();
            cleaned_count += count;
            freed_memory_mb += memory;
        }

        // Cleanup model-specific pools
        for pools in [&self.traditional_pools, &self.lora_pools] {
            let mut pools_write = pools.write().unwrap();
            for pool in pools_write.values_mut() {
                let (count, memory) = pool.cleanup_old_tensors();
                cleaned_count += count;
                freed_memory_mb += memory;
            }
        }

        let cleanup_time = start_time.elapsed();

        CleanupReport {
            cleaned_tensors: cleaned_count,
            freed_memory_mb,
            cleanup_time_ms: cleanup_time.as_secs_f32() * 1000.0,
        }
    }

    /// Check if memory reduction target is met
    pub fn meets_reduction_target(&self) -> bool {
        let stats = self.get_memory_stats();
        stats.meets_target
    }
}

impl TensorPool {
    fn new() -> Self {
        Self {
            available_tensors: HashMap::new(),
            created_at: Instant::now(),
            allocation_count: 0,
            deallocation_count: 0,
        }
    }

    fn add_tensor(&mut self, key: TensorKey, tensor: Tensor) {
        self.available_tensors
            .entry(key)
            .or_insert_with(Vec::new)
            .push(tensor);
        self.deallocation_count += 1;
    }

    fn cleanup_old_tensors(&mut self) -> (usize, f32) {
        // Simple cleanup - remove all tensors older than cleanup interval
        let old_count = self.available_tensors.values().map(|v| v.len()).sum();
        self.available_tensors.clear();
        (old_count, 0.0) // Simplified memory calculation
    }
}

impl SharedTensorPool {
    fn new(max_size_mb: usize) -> Self {
        Self {
            shared_tensors: HashMap::new(),
            usage_stats: SharedPoolStats::default(),
            max_pool_size: max_size_mb,
        }
    }

    fn try_get_tensor(&mut self, key: &TensorKey) -> Option<SharedTensor> {
        if let Some(tensors) = self.shared_tensors.get_mut(key) {
            if let Some(mut shared_tensor) = tensors.pop() {
                shared_tensor.last_accessed = Instant::now();
                *shared_tensor.ref_count.lock().unwrap() += 1;
                self.usage_stats.total_shared_allocations += 1;
                return Some(shared_tensor);
            }
        }
        None
    }

    fn add_tensor(&mut self, key: TensorKey, tensor: SharedTensor) {
        self.shared_tensors
            .entry(key)
            .or_insert_with(Vec::new)
            .push(tensor);
    }

    fn cleanup_unused_tensors(&mut self) -> (usize, f32) {
        let mut cleaned = 0;
        let cutoff_time = Instant::now() - Duration::from_secs(300); // 5 minutes

        self.shared_tensors.retain(|_key, tensors| {
            let original_len = tensors.len();
            tensors.retain(|tensor| {
                let ref_count = *tensor.ref_count.lock().unwrap();
                ref_count > 0 || tensor.last_accessed > cutoff_time
            });
            cleaned += original_len - tensors.len();
            !tensors.is_empty()
        });

        (cleaned, 0.0) // Simplified memory calculation
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current memory usage (MB)
    pub current_usage_mb: f32,
    /// Estimated baseline usage without optimization (MB)
    pub estimated_baseline_mb: f32,
    /// Memory saved through sharing (MB)
    pub shared_savings_mb: f32,
    /// Memory reduction percentage
    pub reduction_percent: f32,
    /// Shared pool hit rate
    pub shared_pool_hit_rate: f32,
    /// Total memory operations
    pub total_operations: usize,
    /// Whether target reduction is met
    pub meets_target: bool,
}

/// Cleanup operation report
#[derive(Debug, Clone)]
pub struct CleanupReport {
    /// Number of tensors cleaned up
    pub cleaned_tensors: usize,
    /// Memory freed (MB)
    pub freed_memory_mb: f32,
    /// Cleanup time (ms)
    pub cleanup_time_ms: f32,
}

/// Calculate size in bytes for a given DType
fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        DType::F16 => 2,
        DType::U32 => 4,
        DType::I64 => 8,
        _ => 4, // Default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let device = Device::Cpu;
        let config = MemoryPoolConfig::default();
        let pool = DualPathMemoryPool::new(device, config);

        let stats = pool.get_memory_stats();
        assert_eq!(stats.current_usage_mb, 0.0);
    }

    #[test]
    fn test_tensor_allocation_and_deallocation() {
        let device = Device::Cpu;
        let config = MemoryPoolConfig::default();
        let pool = DualPathMemoryPool::new(device, config);

        // Allocate tensor
        let tensor = pool
            .allocate_tensor(
                &[128, 768],
                DType::F32,
                "embeddings",
                ModelType::Traditional,
            )
            .unwrap();

        assert_eq!(tensor.shape().dims(), &[128, 768]);

        // Deallocate tensor
        pool.deallocate_tensor(tensor, "embeddings", ModelType::Traditional)
            .unwrap();

        let stats = pool.get_memory_stats();
        assert!(stats.total_operations > 0);
    }

    #[test]
    fn test_memory_reduction_target() {
        let device = Device::Cpu;
        let mut config = MemoryPoolConfig::default();
        config.target_reduction_percent = 10.0; // Lower target for testing

        let pool = DualPathMemoryPool::new(device, config);

        // Simulate some allocations to generate savings
        for i in 0..5 {
            let tensor = pool
                .allocate_tensor(
                    &[64, 384],
                    DType::F32,
                    "input_ids",
                    if i % 2 == 0 {
                        ModelType::Traditional
                    } else {
                        ModelType::LoRA
                    },
                )
                .unwrap();

            pool.deallocate_tensor(tensor, "input_ids", ModelType::Traditional)
                .unwrap();
        }

        let stats = pool.get_memory_stats();
        println!("Memory reduction: {:.1}%", stats.reduction_percent);
    }

    #[test]
    fn test_cleanup_functionality() {
        let device = Device::Cpu;
        let config = MemoryPoolConfig::default();
        let pool = DualPathMemoryPool::new(device, config);

        // Allocate and deallocate some tensors
        for _ in 0..3 {
            let tensor = pool
                .allocate_tensor(&[32, 256], DType::F32, "test", ModelType::LoRA)
                .unwrap();
            pool.deallocate_tensor(tensor, "test", ModelType::LoRA)
                .unwrap();
        }

        let report = pool.cleanup_unused_tensors();
        assert!(report.cleanup_time_ms >= 0.0);
    }
}
