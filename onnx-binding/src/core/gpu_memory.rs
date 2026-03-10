//! GPU memory detection for ORT session configuration.
//!
//! Determines the per-session GPU memory limit via:
//! 1. `ORT_GPU_MEM_LIMIT` env var (supports bytes, "60GB", "60000MB")
//! 2. Auto-probe **free** VRAM via HIP/CUDA + sysfs fallback for total
//! 3. Fallback default (4 GB)
//!
//! `ORT_GPU_NUM_SESSIONS` controls how free VRAM is divided (default: 3).

use std::env;
use std::sync::OnceLock;

const DEFAULT_GPU_SESSIONS: usize = 3;
const GPU_MEM_RESERVE_FRACTION: f64 = 0.05;
const FALLBACK_MEM_LIMIT: usize = 4 * 1024 * 1024 * 1024;

static GPU_MEM_LIMIT: OnceLock<usize> = OnceLock::new();

struct GpuMemInfo {
    free: usize,
    total: usize,
}

/// Returns the per-session GPU memory limit in bytes.
/// Cached after first call — safe to call from multiple session inits.
pub fn get_gpu_mem_limit() -> usize {
    *GPU_MEM_LIMIT.get_or_init(compute_gpu_mem_limit)
}

fn compute_gpu_mem_limit() -> usize {
    // 1. Explicit env var takes priority
    if let Ok(val) = env::var("ORT_GPU_MEM_LIMIT") {
        if let Some(limit) = parse_mem_size(&val) {
            println!(
                "INFO: [gpu_memory] ORT_GPU_MEM_LIMIT={} → {:.1} GB per session",
                val,
                limit as f64 / GB_F
            );
            return limit;
        }
        println!(
            "WARN: [gpu_memory] Invalid ORT_GPU_MEM_LIMIT='{}', falling back to auto-probe",
            val
        );
    }

    let num_sessions = env::var("ORT_GPU_NUM_SESSIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_GPU_SESSIONS);

    // 2. Auto-probe: prefer runtime APIs (HIP/CUDA) for free+total, sysfs as fallback
    if let Some(info) = probe_gpu_memory() {
        let usable = (info.free as f64 * (1.0 - GPU_MEM_RESERVE_FRACTION)) as usize;
        let per_session = usable / num_sessions;
        println!(
            "INFO: [gpu_memory] Probed VRAM: total={:.1} GB, free={:.1} GB, usable={:.1} GB, sessions={}, per_session={:.1} GB",
            info.total as f64 / GB_F,
            info.free as f64 / GB_F,
            usable as f64 / GB_F,
            num_sessions,
            per_session as f64 / GB_F,
        );
        return per_session;
    }

    // 3. Fallback
    println!(
        "WARN: [gpu_memory] Probe failed, using fallback {:.1} GB per session",
        FALLBACK_MEM_LIMIT as f64 / GB_F
    );
    FALLBACK_MEM_LIMIT
}

const GB_F: f64 = (1024 * 1024 * 1024) as f64;

fn parse_mem_size(s: &str) -> Option<usize> {
    let s = s.trim();
    if let Some(num) = s
        .strip_suffix("GB")
        .or_else(|| s.strip_suffix("gb"))
        .or_else(|| s.strip_suffix("Gb"))
    {
        num.trim().parse::<f64>().ok().map(|n| (n * GB_F) as usize)
    } else if let Some(num) = s
        .strip_suffix("MB")
        .or_else(|| s.strip_suffix("mb"))
        .or_else(|| s.strip_suffix("Mb"))
    {
        num.trim()
            .parse::<f64>()
            .ok()
            .map(|n| (n * 1024.0 * 1024.0) as usize)
    } else {
        s.parse::<usize>().ok()
    }
}

/// Probe GPU memory via the best available method.
fn probe_gpu_memory() -> Option<GpuMemInfo> {
    // Method 1: HIP runtime — gives accurate free+total
    #[cfg(feature = "rocm")]
    if let Some(info) = probe_hip() {
        return Some(info);
    }

    // Method 2: CUDA runtime — gives accurate free+total
    #[cfg(feature = "cuda")]
    if let Some(info) = probe_cuda() {
        return Some(info);
    }

    // Method 3: sysfs — only gives total VRAM, treat 90% as "free"
    if let Some(total) = probe_sysfs_total() {
        let estimated_free = (total as f64 * 0.90) as usize;
        return Some(GpuMemInfo {
            free: estimated_free,
            total,
        });
    }

    None
}

/// Read total VRAM size from AMD KFD sysfs topology.
fn probe_sysfs_total() -> Option<usize> {
    let topology = std::path::Path::new("/sys/class/kfd/kfd/topology/nodes");
    if !topology.exists() {
        return None;
    }

    let mut max_vram: usize = 0;

    if let Ok(nodes) = std::fs::read_dir(topology) {
        for node in nodes.flatten() {
            let mem_banks = node.path().join("mem_banks");
            if !mem_banks.exists() {
                continue;
            }
            if let Ok(banks) = std::fs::read_dir(&mem_banks) {
                for bank in banks.flatten() {
                    let props_path = bank.path().join("properties");
                    if let Ok(content) = std::fs::read_to_string(&props_path) {
                        let mut size: usize = 0;
                        let mut is_vram = false;
                        for line in content.lines() {
                            if let Some(val) = line.strip_prefix("size_in_bytes") {
                                size = val.trim().parse().unwrap_or(0);
                            }
                            if let Some(val) = line.strip_prefix("heap_type") {
                                is_vram = val.trim() == "1";
                            }
                            if let Some(val) = line.strip_prefix("flags") {
                                let flags: u64 = val.trim().parse().unwrap_or(0);
                                if flags & 0x2 != 0 {
                                    is_vram = true;
                                }
                            }
                        }
                        if is_vram && size > max_vram {
                            max_vram = size;
                        }
                    }
                }
            }
        }
    }

    if max_vram > 0 {
        Some(max_vram)
    } else {
        None
    }
}

/// Probe free+total via HIP runtime (dlopen libamdhip64.so).
#[cfg(feature = "rocm")]
fn probe_hip() -> Option<GpuMemInfo> {
    type MemGetInfoFn = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;

    unsafe {
        let mut lib = libc::dlopen(
            b"libamdhip64.so\0".as_ptr() as *const libc::c_char,
            libc::RTLD_NOW | libc::RTLD_NOLOAD,
        );
        if lib.is_null() {
            lib = libc::dlopen(
                b"libamdhip64.so\0".as_ptr() as *const libc::c_char,
                libc::RTLD_NOW,
            );
            if lib.is_null() {
                return None;
            }
        }
        let sym = libc::dlsym(lib, b"hipMemGetInfo\0".as_ptr() as *const libc::c_char);
        if sym.is_null() {
            return None;
        }
        let func: MemGetInfoFn = std::mem::transmute(sym);

        let mut free: usize = 0;
        let mut total: usize = 0;
        let ret = func(&mut free, &mut total);
        if ret == 0 && total > 0 {
            Some(GpuMemInfo { free, total })
        } else {
            None
        }
    }
}

/// Probe free+total via CUDA runtime (dlopen libcudart.so).
#[cfg(feature = "cuda")]
fn probe_cuda() -> Option<GpuMemInfo> {
    type MemGetInfoFn = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;

    unsafe {
        let lib = libc::dlopen(
            b"libcudart.so\0".as_ptr() as *const libc::c_char,
            libc::RTLD_NOW,
        );
        if lib.is_null() {
            return None;
        }
        let sym = libc::dlsym(lib, b"cudaMemGetInfo\0".as_ptr() as *const libc::c_char);
        if sym.is_null() {
            return None;
        }
        let func: MemGetInfoFn = std::mem::transmute(sym);

        let mut free: usize = 0;
        let mut total: usize = 0;
        let ret = func(&mut free, &mut total);
        if ret == 0 && total > 0 {
            Some(GpuMemInfo { free, total })
        } else {
            None
        }
    }
}
