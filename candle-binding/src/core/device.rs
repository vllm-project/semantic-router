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
