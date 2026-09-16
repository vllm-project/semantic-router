//! Device resolution shared by every model loader.

use candle_core::Device;

/// Resolve the inference device: CPU when `use_cpu`, otherwise the first
/// available accelerator (Metal when built with the `metal` feature, CUDA
/// otherwise), falling back to CPU when none is usable.
pub fn resolve_device(use_cpu: bool) -> Device {
    if use_cpu {
        return Device::Cpu;
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "metal"))]
    {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    }
}

#[cfg(feature = "metal")]
mod metal_pool {
    use std::sync::OnceLock;

    /// candle's Metal backend keeps one open command buffer per OS thread on
    /// the device's single command queue, which caps in-flight buffers at 64.
    /// Every distinct thread that ever runs a forward permanently occupies one
    /// slot, so unbounded caller threads (one per CGo call) deadlock the queue
    /// regardless of how few run concurrently. Confining inference to a fixed
    /// pool keeps the number of distinct threads, and therefore occupied
    /// slots, small.
    const DEFAULT_THREADS: usize = 8;
    const MAX_THREADS: usize = 32;

    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

    pub fn pool() -> &'static rayon::ThreadPool {
        POOL.get_or_init(|| {
            let threads = std::env::var("METAL_MAX_CONCURRENCY")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value > 0)
                .map(|value| value.min(MAX_THREADS))
                .unwrap_or(DEFAULT_THREADS);
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("metal-inference-{index}"))
                .build()
                .expect("failed to build Metal inference pool")
        })
    }
}

/// Run `operation` on the bounded Metal inference pool when `device` is
/// Metal (`METAL_MAX_CONCURRENCY` threads, default 8, max 32); run it inline
/// on every other device.
pub fn run_on_inference_pool<R, F>(device: &Device, operation: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    #[cfg(feature = "metal")]
    {
        if device.is_metal() {
            return metal_pool::pool().install(operation);
        }
        operation()
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = device;
        operation()
    }
}

/// Commit and drain any Metal work the calling thread queued during model
/// loading. candle enqueues the device's first command buffer at creation,
/// and an enqueued-but-uncommitted buffer blocks every later committed buffer
/// on the same queue; without this drain, the loader thread's leftover buffer
/// hangs the first pooled forward forever.
pub fn drain_loader_queue(device: &Device) {
    #[cfg(feature = "metal")]
    if device.is_metal() {
        let _ = device.synchronize();
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = device;
    }
}
