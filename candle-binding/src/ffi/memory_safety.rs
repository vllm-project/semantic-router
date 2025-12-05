//! Dual-Path Memory Safety System
//!
//! This module provides comprehensive memory safety for the dual-path architecture,
//! including double-free protection, LoRA-specific memory management, and
//! path switching safety mechanisms.

use std::collections::HashMap;
use std::ffi::c_char;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

/// Memory allocation tracking for double-free protection
#[derive(Debug, Clone)]
pub struct AllocationTracker {
    pub ptr_addr: usize, // Store pointer as address for thread safety
    pub size: usize,
    pub allocation_type: AllocationType,
    pub path_type: PathType,
    pub timestamp: std::time::Instant,
}

/// Type of memory allocation
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationType {
    CString,
    FloatArray,
    IntArray,
    StructArray,
    LoRAAdapter,
    TensorBuffer,
}

/// Path type for allocation tracking
#[derive(Debug, Clone, PartialEq)]
pub enum PathType {
    Traditional,
    LoRA,
    Shared,
}

/// Memory safety result
#[derive(Debug)]
pub struct MemorySafetyResult {
    pub is_safe: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub leaked_allocations: usize,
    pub double_free_attempts: usize,
}

// Global memory tracker for dual-path safety using LazyLock
// These are lazily initialized mutable global state for runtime memory tracking
static MEMORY_TRACKER: LazyLock<Arc<RwLock<HashMap<usize, AllocationTracker>>>> =
    LazyLock::new(|| Arc::new(RwLock::new(HashMap::new())));
static DOUBLE_FREE_PROTECTION: LazyLock<Arc<Mutex<HashMap<usize, bool>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));
static LORA_MEMORY_POOL: LazyLock<Arc<Mutex<LoRAMemoryPool>>> =
    LazyLock::new(|| Arc::new(Mutex::new(LoRAMemoryPool::new())));
static PATH_SWITCH_GUARD: LazyLock<Arc<RwLock<PathSwitchState>>> =
    LazyLock::new(|| Arc::new(RwLock::new(PathSwitchState::new())));

/// LoRA-specific memory pool for high-performance allocations
#[derive(Debug)]
pub struct LoRAMemoryPool {
    adapters: HashMap<String, Vec<u8>>,
    tensor_buffers: Vec<Vec<f32>>,
    reusable_strings: Vec<String>,
    total_allocated: usize,
    peak_usage: usize,
}

impl LoRAMemoryPool {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            tensor_buffers: Vec::new(),
            reusable_strings: Vec::new(),
            total_allocated: 0,
            peak_usage: 0,
        }
    }

    /// Allocate LoRA adapter memory with tracking
    pub fn allocate_adapter(&mut self, name: &str, size: usize) -> *mut u8 {
        let buffer = vec![0u8; size];
        let ptr = buffer.as_ptr() as *mut u8;

        self.adapters.insert(name.to_string(), buffer);
        self.total_allocated += size;
        self.peak_usage = self.peak_usage.max(self.total_allocated);

        // Track allocation
        track_allocation(ptr, size, AllocationType::LoRAAdapter, PathType::LoRA);

        ptr
    }

    /// Reuse tensor buffer to avoid frequent allocations
    pub fn get_tensor_buffer(&mut self, size: usize) -> *mut f32 {
        // Try to reuse existing buffer
        for buffer in &mut self.tensor_buffers {
            if buffer.len() >= size {
                return buffer.as_mut_ptr();
            }
        }

        // Create new buffer if none suitable
        let mut buffer = vec![0.0f32; size];
        let ptr = buffer.as_mut_ptr();
        self.tensor_buffers.push(buffer);

        // Track allocation
        track_allocation(
            ptr as *mut u8,
            size * 4,
            AllocationType::TensorBuffer,
            PathType::LoRA,
        );

        ptr
    }

    /// Get reusable string to avoid allocations
    pub fn get_reusable_string(&mut self, content: &str) -> *mut c_char {
        // Try to reuse existing string
        for existing in &mut self.reusable_strings {
            if existing.capacity() >= content.len() {
                existing.clear();
                existing.push_str(content);
                return existing.as_ptr() as *mut c_char;
            }
        }

        // Create new string
        let mut string = String::with_capacity(content.len() + 32); // Extra capacity
        string.push_str(content);
        let ptr = string.as_ptr() as *mut c_char;
        self.reusable_strings.push(string);

        ptr
    }

    /// Clean up unused allocations
    pub fn cleanup(&mut self) {
        self.adapters.retain(|_, buffer| !buffer.is_empty());
        self.tensor_buffers.retain(|buffer| !buffer.is_empty());
        self.reusable_strings.retain(|s| !s.is_empty());
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> LoRAMemoryStats {
        LoRAMemoryStats {
            total_allocated: self.total_allocated,
            peak_usage: self.peak_usage,
            active_adapters: self.adapters.len(),
            tensor_buffers: self.tensor_buffers.len(),
            reusable_strings: self.reusable_strings.len(),
        }
    }
}

/// LoRA memory statistics
#[derive(Debug, Clone)]
pub struct LoRAMemoryStats {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub active_adapters: usize,
    pub tensor_buffers: usize,
    pub reusable_strings: usize,
}

/// Path switching state for memory safety during transitions
#[derive(Debug)]
pub struct PathSwitchState {
    pub current_path: PathType,
    pub switching_in_progress: bool,
    pub pending_deallocations: Vec<usize>, // Store addresses instead of pointers
    pub switch_count: usize,
}

impl PathSwitchState {
    pub fn new() -> Self {
        Self {
            current_path: PathType::Traditional,
            switching_in_progress: false,
            pending_deallocations: Vec::new(),
            switch_count: 0,
        }
    }

    /// Begin path switch with memory safety
    pub fn begin_switch(&mut self, new_path: PathType) -> bool {
        if self.switching_in_progress {
            return false; // Already switching
        }

        self.switching_in_progress = true;
        self.current_path = new_path;
        self.switch_count += 1;
        true
    }

    /// Complete path switch and process pending deallocations
    pub fn complete_switch(&mut self) {
        if !self.switching_in_progress {
            return;
        }

        // Process pending deallocations safely
        for &ptr_addr in &self.pending_deallocations {
            unsafe_deallocation(ptr_addr as *mut u8);
        }
        self.pending_deallocations.clear();

        self.switching_in_progress = false;
    }

    /// Add deallocation to pending list during switch
    pub fn defer_deallocation(&mut self, ptr: *mut u8) {
        self.pending_deallocations.push(ptr as usize);
    }
}

/// Track memory allocation with double-free protection
pub fn track_allocation(
    ptr: *mut u8,
    size: usize,
    alloc_type: AllocationType,
    path_type: PathType,
) {
    let ptr_addr = ptr as usize;
    let tracker = AllocationTracker {
        ptr_addr,
        size,
        allocation_type: alloc_type,
        path_type,
        timestamp: std::time::Instant::now(),
    };

    // Add to memory tracker
    if let Ok(mut memory_map) = MEMORY_TRACKER.write() {
        memory_map.insert(ptr_addr, tracker);
    }

    // Mark as allocated for double-free protection
    if let Ok(mut protection_map) = DOUBLE_FREE_PROTECTION.lock() {
        protection_map.insert(ptr_addr, true);
    }
}

/// Safe deallocation with double-free protection
pub fn safe_deallocation(ptr: *mut u8) -> bool {
    let ptr_addr = ptr as usize;

    // Check if switching is in progress
    if let Ok(mut switch_state) = PATH_SWITCH_GUARD.write() {
        if switch_state.switching_in_progress {
            switch_state.defer_deallocation(ptr);
            return true;
        }
    }

    // Check double-free protection
    if let Ok(mut protection_map) = DOUBLE_FREE_PROTECTION.lock() {
        if let Some(&is_allocated) = protection_map.get(&ptr_addr) {
            if !is_allocated {
                // Double-free attempt detected!
                eprintln!("Double-free attempt detected for pointer: {:?}", ptr);
                return false;
            }
            protection_map.insert(ptr_addr, false); // Mark as freed
        } else {
            // Pointer not tracked - potential issue
            eprintln!("Attempting to free untracked pointer: {:?}", ptr);
            return false;
        }
    }

    // Remove from memory tracker
    if let Ok(mut memory_map) = MEMORY_TRACKER.write() {
        memory_map.remove(&ptr_addr);
    }

    // Perform actual deallocation
    unsafe_deallocation(ptr);
    true
}

/// Unsafe deallocation (internal use only)
fn unsafe_deallocation(ptr: *mut u8) {
    if !ptr.is_null() {
        let ptr_addr = ptr as usize;
        unsafe {
            // Determine allocation type and deallocate appropriately
            if let Ok(memory_map) = MEMORY_TRACKER.read() {
                if let Some(tracker) = memory_map.get(&ptr_addr) {
                    match tracker.allocation_type {
                        AllocationType::CString => {
                            let _ = std::ffi::CString::from_raw(ptr as *mut c_char);
                        }
                        AllocationType::FloatArray => {
                            let _ = Vec::from_raw_parts(ptr as *mut f32, 0, tracker.size / 4);
                        }
                        AllocationType::IntArray => {
                            let _ = Vec::from_raw_parts(ptr as *mut i32, 0, tracker.size / 4);
                        }
                        _ => {
                            // Generic deallocation
                            let _ = Vec::from_raw_parts(ptr, 0, tracker.size);
                        }
                    }
                }
            }
        }
    }
}

/// Begin safe path switch
pub fn begin_path_switch(new_path: PathType) -> bool {
    if let Ok(mut switch_state) = PATH_SWITCH_GUARD.write() {
        switch_state.begin_switch(new_path)
    } else {
        false
    }
}

/// Complete safe path switch
pub fn complete_path_switch() {
    if let Ok(mut switch_state) = PATH_SWITCH_GUARD.write() {
        switch_state.complete_switch();
    }
}

/// Get LoRA memory pool statistics
pub fn get_lora_memory_stats() -> LoRAMemoryStats {
    if let Ok(pool) = LORA_MEMORY_POOL.lock() {
        pool.get_stats()
    } else {
        LoRAMemoryStats {
            total_allocated: 0,
            peak_usage: 0,
            active_adapters: 0,
            tensor_buffers: 0,
            reusable_strings: 0,
        }
    }
}

/// Perform comprehensive memory safety check
pub fn perform_memory_safety_check() -> MemorySafetyResult {
    let mut result = MemorySafetyResult {
        is_safe: true,
        warnings: Vec::new(),
        errors: Vec::new(),
        leaked_allocations: 0,
        double_free_attempts: 0,
    };

    // Check for memory leaks
    if let Ok(memory_map) = MEMORY_TRACKER.read() {
        result.leaked_allocations = memory_map.len();

        if result.leaked_allocations > 0 {
            result.warnings.push(format!(
                "Detected {} potential memory leaks",
                result.leaked_allocations
            ));
        }

        // Check for old allocations (potential leaks)
        let now = std::time::Instant::now();
        for (ptr_addr, tracker) in memory_map.iter() {
            let age = now.duration_since(tracker.timestamp);
            if age.as_secs() > 300 {
                // 5 minutes
                result.warnings.push(format!(
                    "Long-lived allocation detected: 0x{:x} (age: {}s, type: {:?})",
                    ptr_addr,
                    age.as_secs(),
                    tracker.allocation_type
                ));
            }
        }
    }

    // Check double-free protection status
    if let Ok(protection_map) = DOUBLE_FREE_PROTECTION.lock() {
        let freed_count = protection_map.values().filter(|&&freed| !freed).count();
        if freed_count > protection_map.len() / 2 {
            result.warnings.push(format!(
                "High number of freed pointers still tracked: {}",
                freed_count
            ));
        }
    }

    // Check path switching state
    if let Ok(switch_state) = PATH_SWITCH_GUARD.read() {
        if switch_state.switching_in_progress {
            result
                .warnings
                .push("Path switching in progress - some operations may be deferred".to_string());
        }

        if !switch_state.pending_deallocations.is_empty() {
            result.warnings.push(format!(
                "Pending deallocations during path switch: {}",
                switch_state.pending_deallocations.len()
            ));
        }
    }

    // Overall safety assessment
    result.is_safe = result.errors.is_empty() && result.leaked_allocations < 100;

    result
}

/// Clean up all memory tracking (for shutdown)
pub fn cleanup_memory_tracking() {
    if let Ok(mut memory_map) = MEMORY_TRACKER.write() {
        memory_map.clear();
    }

    if let Ok(mut protection_map) = DOUBLE_FREE_PROTECTION.lock() {
        protection_map.clear();
    }

    if let Ok(mut pool) = LORA_MEMORY_POOL.lock() {
        pool.cleanup();
    }
}

/// FFI-safe memory allocation for traditional path
#[no_mangle]
pub extern "C" fn safe_alloc_traditional(size: usize) -> *mut u8 {
    let buffer = vec![0u8; size];
    let ptr = buffer.as_ptr() as *mut u8;
    std::mem::forget(buffer); // Prevent automatic deallocation

    track_allocation(
        ptr,
        size,
        AllocationType::StructArray,
        PathType::Traditional,
    );
    ptr
}

/// FFI-safe memory allocation for LoRA path
#[no_mangle]
pub extern "C" fn safe_alloc_lora(size: usize) -> *mut u8 {
    let buffer = vec![0u8; size];
    let ptr = buffer.as_ptr() as *mut u8;
    std::mem::forget(buffer);

    track_allocation(ptr, size, AllocationType::StructArray, PathType::LoRA);
    ptr
}

/// FFI-safe memory deallocation
#[no_mangle]
pub extern "C" fn safe_free(ptr: *mut u8) -> bool {
    safe_deallocation(ptr)
}

/// FFI function to get memory safety status
#[no_mangle]
pub extern "C" fn get_memory_safety_status() -> bool {
    let result = perform_memory_safety_check();
    result.is_safe
}

/// FFI function to get LoRA memory usage
#[no_mangle]
pub extern "C" fn get_lora_memory_usage() -> usize {
    let stats = get_lora_memory_stats();
    stats.total_allocated
}

/// FFI function to cleanup memory tracking
#[no_mangle]
pub extern "C" fn cleanup_dual_path_memory() {
    cleanup_memory_tracking();
}
