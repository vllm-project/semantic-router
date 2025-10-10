//! Tests for FFI memory_safety module

use super::memory_safety::*;
use rstest::*;

/// Test safe_alloc_traditional function
#[rstest]
#[case(1024, "1KB allocation")]
#[case(4096, "4KB allocation")]
#[case(64, "Small allocation")]
fn test_memory_safety_safe_alloc_traditional(#[case] size: usize, #[case] _description: &str) {
    let ptr = safe_alloc_traditional(size);

    // Verify pointer is not null
    assert!(!ptr.is_null(), "Allocated pointer should not be null");

    // Test that we can write to the allocated memory
    unsafe {
        *ptr = 42;
        assert_eq!(*ptr, 42, "Should be able to write to allocated memory");
    }

    // Clean up
    let freed = safe_free(ptr);
    assert!(freed, "Memory should be successfully freed");

    println!("Safe alloc traditional test passed for size: {}", size);
}

/// Test safe_alloc_lora function
#[rstest]
#[case(2048, "2KB LoRA allocation")]
#[case(512, "Small LoRA allocation")]
fn test_memory_safety_safe_alloc_lora(#[case] size: usize, #[case] _description: &str) {
    let ptr = safe_alloc_lora(size);

    // Verify pointer is not null
    assert!(!ptr.is_null(), "Allocated LoRA pointer should not be null");

    // Test that we can write to the allocated memory
    unsafe {
        *ptr = 123;
        assert_eq!(
            *ptr, 123,
            "Should be able to write to LoRA allocated memory"
        );
    }

    // Clean up
    let freed = safe_free(ptr);
    assert!(freed, "LoRA memory should be successfully freed");

    println!("Safe alloc LoRA test passed for size: {}", size);
}

/// Test safe_free function with null pointer
#[rstest]
fn test_memory_safety_safe_free_null_pointer() {
    let result = safe_free(std::ptr::null_mut());

    // Freeing null pointer should be safe and return false
    assert!(!result, "Freeing null pointer should return false");

    println!("Safe free null pointer test passed");
}

/// Test memory cleanup
#[rstest]
fn test_memory_safety_memory_cleanup() {
    // Allocate some memory
    let ptr1 = safe_alloc_traditional(1024);
    let ptr2 = safe_alloc_lora(2048);

    // Cleanup memory tracking
    cleanup_dual_path_memory();

    // Note: We don't free ptr1 and ptr2 here because cleanup_dual_path_memory
    // should handle the tracking cleanup, but the actual memory might still need
    // to be freed explicitly in a real scenario
    safe_free(ptr1);
    safe_free(ptr2);

    println!("Memory cleanup test passed");
}
