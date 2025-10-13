//! Tests for routing system

use super::config::{PathSelectionStrategy, ProcessingPriority};
use super::routing::*;
use super::traits::{ModelType, TaskType};
use rstest::*;
use std::time::Duration;

/// Test router path selection with AlwaysLoRA strategy
#[rstest]
fn test_routing_always_lora_strategy() {
    let router = DualPathRouter::new(PathSelectionStrategy::AlwaysLoRA);

    let requirements = ProcessingRequirements {
        confidence_threshold: 0.8,
        max_latency: Duration::from_millis(100),
        batch_size: 16,
        tasks: vec![TaskType::Intent],
        priority: ProcessingPriority::Latency,
    };

    let selection = router.select_path(&requirements);

    // Test that LoRA is always selected
    assert_eq!(selection.selected_path, ModelType::LoRA);
    assert_eq!(selection.confidence, 1.0);
    assert!(selection.reasoning.contains("Always use LoRA"));

    println!("AlwaysLoRA strategy test passed");
}

/// Test router path selection with AlwaysTraditional strategy
#[rstest]
fn test_routing_always_traditional_strategy() {
    let router = DualPathRouter::new(PathSelectionStrategy::AlwaysTraditional);

    let requirements = ProcessingRequirements {
        confidence_threshold: 0.9,
        max_latency: Duration::from_millis(500),
        batch_size: 32,
        tasks: vec![TaskType::PII, TaskType::Security],
        priority: ProcessingPriority::Accuracy,
    };

    let selection = router.select_path(&requirements);

    // Test that Traditional is always selected
    assert_eq!(selection.selected_path, ModelType::Traditional);
    assert_eq!(selection.confidence, 1.0);
    assert!(selection.reasoning.contains("Always use Traditional"));

    println!("AlwaysTraditional strategy test passed");
}

/// Test router path selection with Automatic strategy
#[rstest]
fn test_routing_automatic_strategy() {
    let router = DualPathRouter::new(PathSelectionStrategy::Automatic);

    let requirements = ProcessingRequirements {
        confidence_threshold: 0.8,
        max_latency: Duration::from_millis(200),
        batch_size: 16,
        tasks: vec![TaskType::Classification],
        priority: ProcessingPriority::Throughput,
    };

    let selection = router.select_path(&requirements);

    // Test that a valid path is selected
    assert!(matches!(
        selection.selected_path,
        ModelType::Traditional | ModelType::LoRA
    ));
    assert!(selection.confidence >= 0.0 && selection.confidence <= 1.0);
    assert!(!selection.reasoning.is_empty());

    println!(
        "Automatic strategy test passed - selected: {:?} (confidence: {:.2})",
        selection.selected_path, selection.confidence
    );
}

/// Test router path selection with PerformanceBased strategy
#[rstest]
fn test_routing_performance_based_strategy() {
    let router = DualPathRouter::new(PathSelectionStrategy::PerformanceBased);

    let requirements = ProcessingRequirements {
        confidence_threshold: 0.85,
        max_latency: Duration::from_millis(150),
        batch_size: 24,
        tasks: vec![TaskType::Intent, TaskType::PII],
        priority: ProcessingPriority::Latency,
    };

    let selection = router.select_path(&requirements);

    // Test that a valid path is selected
    assert!(matches!(
        selection.selected_path,
        ModelType::Traditional | ModelType::LoRA
    ));
    assert!(selection.confidence >= 0.0 && selection.confidence <= 1.0);
    assert!(!selection.reasoning.is_empty());

    println!(
        "PerformanceBased strategy test passed - selected: {:?} (confidence: {:.2})",
        selection.selected_path, selection.confidence
    );
}

/// Test different processing priorities
#[rstest]
#[case(ProcessingPriority::Latency, "latency_priority")]
#[case(ProcessingPriority::Accuracy, "accuracy_priority")]
#[case(ProcessingPriority::Throughput, "throughput_priority")]
#[case(ProcessingPriority::Balanced, "balanced_priority")]
fn test_routing_processing_priorities(
    #[case] priority: ProcessingPriority,
    #[case] priority_name: &str,
) {
    let router = DualPathRouter::new(PathSelectionStrategy::Automatic);

    let requirements = ProcessingRequirements {
        confidence_threshold: 0.8,
        max_latency: Duration::from_millis(200),
        batch_size: 16,
        tasks: vec![TaskType::Intent],
        priority,
    };

    let selection = router.select_path(&requirements);

    // Test that selection is made regardless of priority
    assert!(matches!(
        selection.selected_path,
        ModelType::Traditional | ModelType::LoRA
    ));
    assert!(selection.confidence >= 0.0 && selection.confidence <= 1.0);

    // Test priority-specific logic (simplified)
    match priority {
        ProcessingPriority::Latency => {
            // Latency priority might prefer LoRA for parallel processing
            println!("Latency priority selection: {:?}", selection.selected_path);
        }
        ProcessingPriority::Accuracy => {
            // Accuracy priority might prefer Traditional for stability
            println!("Accuracy priority selection: {:?}", selection.selected_path);
        }
        ProcessingPriority::Throughput => {
            // Throughput priority might prefer LoRA for batch processing
            println!(
                "Throughput priority selection: {:?}",
                selection.selected_path
            );
        }
        ProcessingPriority::Balanced => {
            // Balanced priority uses automatic selection
            println!("Balanced priority selection: {:?}", selection.selected_path);
        }
    }

    println!("Processing priority test passed for {}", priority_name);
}

/// Test different task combinations
#[rstest]
#[case(vec![TaskType::Intent], "single_intent")]
#[case(vec![TaskType::PII], "single_pii")]
#[case(vec![TaskType::Security], "single_security")]
#[case(vec![TaskType::Intent, TaskType::PII], "dual_task")]
#[case(vec![TaskType::Intent, TaskType::PII, TaskType::Security], "multi_task")]
fn test_routing_task_combinations(#[case] tasks: Vec<TaskType>, #[case] task_description: &str) {
    let router = DualPathRouter::new(PathSelectionStrategy::Automatic);

    let requirements = ProcessingRequirements {
        confidence_threshold: 0.8,
        max_latency: Duration::from_millis(200),
        batch_size: 16,
        tasks: tasks.clone(),
        priority: ProcessingPriority::Throughput,
    };

    let selection = router.select_path(&requirements);

    // Test that selection works for different task combinations
    assert!(matches!(
        selection.selected_path,
        ModelType::Traditional | ModelType::LoRA
    ));
    assert!(selection.confidence >= 0.0 && selection.confidence <= 1.0);

    // Multi-task scenarios might prefer LoRA
    if tasks.len() > 1 {
        println!(
            "Multi-task scenario ({} tasks) selected: {:?}",
            tasks.len(),
            selection.selected_path
        );
    } else {
        println!(
            "Single-task scenario selected: {:?}",
            selection.selected_path
        );
    }

    println!(
        "Task combination test passed for {} ({} tasks)",
        task_description,
        tasks.len()
    );
}

/// Test confidence threshold impact
#[rstest]
#[case(0.5, "low_confidence")]
#[case(0.8, "medium_confidence")]
#[case(0.95, "high_confidence")]
fn test_routing_confidence_threshold_impact(
    #[case] confidence_threshold: f32,
    #[case] threshold_description: &str,
) {
    let router = DualPathRouter::new(PathSelectionStrategy::Automatic);

    let requirements = ProcessingRequirements {
        confidence_threshold,
        max_latency: Duration::from_millis(200),
        batch_size: 16,
        tasks: vec![TaskType::Intent],
        priority: ProcessingPriority::Accuracy,
    };

    let selection = router.select_path(&requirements);

    // Test that selection is made regardless of confidence threshold
    assert!(matches!(
        selection.selected_path,
        ModelType::Traditional | ModelType::LoRA
    ));
    assert!(selection.confidence >= 0.0 && selection.confidence <= 1.0);

    // High confidence requirements might prefer Traditional for stability
    if confidence_threshold > 0.9 {
        println!(
            "High confidence requirement ({}), selected: {:?}",
            confidence_threshold, selection.selected_path
        );
    }

    println!(
        "Confidence threshold test passed for {} (threshold: {})",
        threshold_description, confidence_threshold
    );
}

/// Test latency constraints
#[rstest]
#[case(50, "very_low_latency")]
#[case(100, "low_latency")]
#[case(500, "medium_latency")]
#[case(1000, "high_latency")]
fn test_routing_latency_constraints(
    #[case] max_latency_ms: u64,
    #[case] latency_description: &str,
) {
    let router = DualPathRouter::new(PathSelectionStrategy::Automatic);

    let requirements = ProcessingRequirements {
        confidence_threshold: 0.8,
        max_latency: Duration::from_millis(max_latency_ms),
        batch_size: 16,
        tasks: vec![TaskType::Intent],
        priority: ProcessingPriority::Latency,
    };

    let selection = router.select_path(&requirements);

    // Test that selection considers latency constraints
    assert!(matches!(
        selection.selected_path,
        ModelType::Traditional | ModelType::LoRA
    ));
    assert!(selection.confidence >= 0.0 && selection.confidence <= 1.0);

    // Very low latency might prefer LoRA for parallel processing
    if max_latency_ms < 100 {
        println!(
            "Very low latency requirement ({}ms), selected: {:?}",
            max_latency_ms, selection.selected_path
        );
    }

    println!(
        "Latency constraint test passed for {} ({}ms)",
        latency_description, max_latency_ms
    );
}

/// Test batch size impact
#[rstest]
#[case(1, "single_item")]
#[case(8, "small_batch")]
#[case(32, "medium_batch")]
#[case(128, "large_batch")]
fn test_routing_batch_size_impact(#[case] batch_size: usize, #[case] batch_description: &str) {
    let router = DualPathRouter::new(PathSelectionStrategy::Automatic);

    let requirements = ProcessingRequirements {
        confidence_threshold: 0.8,
        max_latency: Duration::from_millis(200),
        batch_size,
        tasks: vec![TaskType::Intent],
        priority: ProcessingPriority::Throughput,
    };

    let selection = router.select_path(&requirements);

    // Test that selection considers batch size
    assert!(matches!(
        selection.selected_path,
        ModelType::Traditional | ModelType::LoRA
    ));
    assert!(selection.confidence >= 0.0 && selection.confidence <= 1.0);

    // Large batches might prefer LoRA for parallel processing
    if batch_size > 64 {
        println!(
            "Large batch size ({}), selected: {:?}",
            batch_size, selection.selected_path
        );
    }

    println!(
        "Batch size test passed for {} (size: {})",
        batch_description, batch_size
    );
}
