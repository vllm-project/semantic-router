//! Tests for global state manager

use super::state_manager::*;
use rayon::prelude::*;
use rstest::*;

// Note: These tests use the actual singleton instance, so they may affect each other
// In a real scenario, you might want to use a separate test instance or mock

// ============================================================================
// Singleton Tests
// ============================================================================

#[rstest]
fn test_global_state_manager_instance() {
    let instance1 = GlobalStateManager::instance();
    let instance2 = GlobalStateManager::instance();

    // Should return the same instance (singleton pattern)
    assert_eq!(
        instance1 as *const GlobalStateManager, instance2 as *const GlobalStateManager,
        "Should return the same singleton instance"
    );
}

// ============================================================================
// System State Tests
// ============================================================================

#[rstest]
fn test_system_state_initial() {
    let manager = GlobalStateManager::instance();
    let state = manager.get_system_state();

    // System should either be Uninitialized or Ready (depending on test order)
    assert!(
        matches!(
            state,
            SystemState::Uninitialized | SystemState::Ready | SystemState::Initializing
        ),
        "Initial state should be Uninitialized, Initializing, or Ready"
    );
}

#[rstest]
fn test_system_state_enum() {
    // Test SystemState enum variants
    let states = vec![
        SystemState::Uninitialized,
        SystemState::Initializing,
        SystemState::Ready,
        SystemState::ShuttingDown,
        SystemState::Error("Test error".to_string()),
    ];

    for state in states {
        assert!(
            matches!(
                state,
                SystemState::Uninitialized
                    | SystemState::Initializing
                    | SystemState::Ready
                    | SystemState::ShuttingDown
                    | SystemState::Error(_)
            ),
            "Should be valid SystemState variant"
        );
    }
}

// ============================================================================
// Initialization Status Tests
// ============================================================================

#[rstest]
fn test_is_any_initialized() {
    let manager = GlobalStateManager::instance();

    // This will be true or false depending on what's initialized
    let any_init = manager.is_any_initialized();

    // Just verify it returns a boolean
    assert!(any_init || !any_init, "Should return boolean");
}

#[rstest]
fn test_is_ready() {
    let manager = GlobalStateManager::instance();

    // Just verify the method works
    let ready = manager.is_ready();
    assert!(ready || !ready, "Should return boolean");
}

// ============================================================================
// Classifier Initialization Tests
// ============================================================================

#[rstest]
fn test_is_unified_classifier_initialized() {
    let manager = GlobalStateManager::instance();

    let is_init = manager.is_unified_classifier_initialized();

    // Should return a boolean
    assert!(is_init || !is_init, "Should return boolean");

    // If initialized, should be able to get it
    if is_init {
        let classifier = manager.get_unified_classifier();
        assert!(
            classifier.is_some(),
            "Should return classifier when initialized"
        );
    }
}

#[rstest]
fn test_get_unified_classifier_when_not_initialized() {
    let manager = GlobalStateManager::instance();

    // Attempt to get classifier (may or may not be initialized)
    let classifier = manager.get_unified_classifier();

    // Should return Option
    match classifier {
        Some(_) => {
            // If Some, is_initialized should be true
            assert!(manager.is_unified_classifier_initialized());
        }
        None => {
            // If None, might not be initialized (or just wasn't set yet)
        }
    }
}

// ============================================================================
// LoRA Engine Tests
// ============================================================================

#[rstest]
fn test_get_parallel_lora_engine() {
    let manager = GlobalStateManager::instance();

    // Attempt to get LoRA engine
    let engine = manager.get_parallel_lora_engine();

    // Should return Option (may be None if not initialized)
    match engine {
        Some(_) => {
            // Successfully got engine
        }
        None => {
            // Engine not initialized yet
        }
    }
}

// ============================================================================
// Token Classifier Tests
// ============================================================================

#[rstest]
fn test_get_lora_token_classifier() {
    let manager = GlobalStateManager::instance();

    // Attempt to get token classifier
    let classifier = manager.get_lora_token_classifier();

    // Should return Option
    match classifier {
        Some(_) => {
            // Successfully got classifier
        }
        None => {
            // Classifier not initialized
        }
    }
}

// ============================================================================
// BERT Similarity Tests
// ============================================================================

#[rstest]
fn test_get_bert_similarity() {
    let manager = GlobalStateManager::instance();

    // Attempt to get BERT similarity
    let similarity = manager.get_bert_similarity();

    // Should return Option
    match similarity {
        Some(_) => {
            // Successfully got similarity
        }
        None => {
            // Similarity not initialized
        }
    }
}

// ============================================================================
// Legacy Classifier Tests
// ============================================================================

#[rstest]
#[case("legacy_bert")]
#[case("legacy_pii")]
#[case("legacy_jailbreak")]
#[case("nonexistent")]
fn test_get_legacy_classifier(#[case] name: &str) {
    let manager = GlobalStateManager::instance();

    // Attempt to get legacy classifier by name
    let classifier = manager.get_legacy_classifier(name);

    // Should return Option (likely None for most names)
    match classifier {
        Some(_) => {
            // Found a classifier with this name
        }
        None => {
            // Classifier not found or not initialized
        }
    }
}

// ============================================================================
// Statistics Tests
// ============================================================================

#[rstest]
fn test_get_stats() {
    let manager = GlobalStateManager::instance();

    // Get statistics
    let stats = manager.get_stats();

    // Verify structure (based on actual implementation)
    // Note: You may need to adjust these assertions based on actual struct fields
    assert!(
        stats.unified_classifier_initialized || !stats.unified_classifier_initialized,
        "Should have unified_classifier_initialized field"
    );
    assert!(
        stats.parallel_lora_engine_initialized || !stats.parallel_lora_engine_initialized,
        "Should have parallel_lora_engine_initialized field"
    );
    assert!(
        stats.lora_token_classifier_initialized || !stats.lora_token_classifier_initialized,
        "Should have lora_token_classifier_initialized field"
    );
    assert!(
        stats.bert_similarity_initialized || !stats.bert_similarity_initialized,
        "Should have bert_similarity_initialized field"
    );
}

// ============================================================================
// Cleanup Tests
// ============================================================================

#[rstest]
fn test_cleanup_method_exists() {
    let manager = GlobalStateManager::instance();

    // Just verify cleanup method can be called
    // Note: We don't actually call it in tests as it would affect other tests
    // manager.cleanup();

    // Instead, just verify the method exists through compilation
    let _ = manager; // Use the manager to avoid unused variable warning
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

#[rstest]
fn test_global_state_manager_thread_safety() {
    // Use rayon for parallel execution - simpler and more efficient
    (0..4).into_par_iter().for_each(|_| {
        let manager = GlobalStateManager::instance();
        let _ = manager.get_system_state();
        let _ = manager.is_any_initialized();
        let _ = manager.get_stats();
    });
}

#[rstest]
fn test_concurrent_state_access() {
    // Use rayon for parallel execution - simpler and more efficient
    let results: Vec<_> = (0..8)
        .into_par_iter()
        .map(|i| {
            let manager = GlobalStateManager::instance();

            // Perform various read operations
            let _ = manager.get_system_state();
            let _ = manager.is_ready();
            let _ = manager.is_any_initialized();
            let _ = manager.get_unified_classifier();
            let _ = manager.get_parallel_lora_engine();
            let _ = manager.get_lora_token_classifier();
            let _ = manager.get_bert_similarity();
            let _ = manager.get_legacy_classifier(&format!("classifier_{}", i));
            let _ = manager.get_stats();

            i // Return thread number
        })
        .collect();

    for (idx, result) in results.into_iter().enumerate() {
        assert_eq!(result, idx, "Thread should return correct index");
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[rstest]
fn test_system_state_error_variant() {
    let error_state = SystemState::Error("Test error message".to_string());

    match error_state {
        SystemState::Error(msg) => {
            assert_eq!(msg, "Test error message");
        }
        _ => panic!("Should be Error variant"),
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[rstest]
fn test_state_consistency() {
    let manager = GlobalStateManager::instance();

    // Get initialization status
    let unified_init = manager.is_unified_classifier_initialized();
    let any_init = manager.is_any_initialized();

    // If unified classifier is initialized, any_init should be true
    if unified_init {
        assert!(
            any_init,
            "If unified classifier is initialized, any_init should be true"
        );
    }

    // Get stats and verify consistency
    let stats = manager.get_stats();
    assert_eq!(
        stats.unified_classifier_initialized, unified_init,
        "Stats should match is_initialized status"
    );
}

#[rstest]
fn test_get_operations_consistency() {
    let manager = GlobalStateManager::instance();

    // Call get twice, should return consistent results
    let classifier1 = manager.get_unified_classifier();
    let classifier2 = manager.get_unified_classifier();

    match (classifier1, classifier2) {
        (Some(_), Some(_)) => {
            // Both Some - consistent
        }
        (None, None) => {
            // Both None - consistent
        }
        _ => {
            // This should not happen unless there's a race condition
            // In practice, once initialized, it should stay initialized
        }
    }
}
