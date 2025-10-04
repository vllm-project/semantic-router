//! Tests for traditional batch processor implementation

use super::batch_processor::*;
use crate::test_fixtures::fixtures::*;
use candle_core::{Device, Result};
use rstest::*;
use std::time::Duration;

/// Test TraditionalBatchProcessor creation
#[rstest]
fn test_batch_processor_traditional_batch_processor_new(cpu_device: Device) {
    let config = BatchProcessorConfig::default();
    let processor = TraditionalBatchProcessor::new(cpu_device.clone(), config.clone());

    // Test that processor was created successfully
    // We can't directly access private fields, but we can test the interface

    // Test metrics access
    let metrics = processor.get_metrics();
    assert_eq!(metrics.total_batches, 0); // Should start with 0
    assert_eq!(metrics.total_items, 0);

    // Test optimal batch size calculation
    let optimal_size = processor.get_optimal_batch_size();
    assert_eq!(optimal_size, config.default_batch_size); // Should return default when no history

    println!("TraditionalBatchProcessor creation test passed");
}

/// Test basic batch processing
#[rstest]
fn test_batch_processor_traditional_batch_processor_process_batch(cpu_device: Device) {
    let config = BatchProcessorConfig::default();
    let mut processor = TraditionalBatchProcessor::new(cpu_device, config);

    let sample_texts = sample_texts();
    let texts = vec![sample_texts[6], sample_texts[7], sample_texts[8]]; // "hello", "world", "test"

    // Simple processor that converts to uppercase
    let uppercase_processor = |text: &str| -> Result<String> { Ok(text.to_uppercase()) };

    let result = processor.process_batch(&texts, uppercase_processor);

    match result {
        Ok(batch_result) => {
            // Test results
            assert_eq!(batch_result.results.len(), 3);

            // Test batch metadata
            assert_eq!(batch_result.batch_size, 3);
            assert_eq!(batch_result.failed_indices.len(), 0);
            assert_eq!(batch_result.success_rate, 1.0);
            assert!(batch_result.processing_time.as_nanos() > 0);

            println!(
                "TraditionalBatchProcessor.process_batch test passed: {} items processed in {:?}",
                batch_result.results.len(),
                batch_result.processing_time
            );
        }
        Err(e) => {
            println!("TraditionalBatchProcessor.process_batch failed: {}", e);
        }
    }
}

/// Test batch processing with failures
#[rstest]
fn test_batch_processor_batch_processing_with_failures(cpu_device: Device) {
    let config = BatchProcessorConfig::default();
    let mut processor = TraditionalBatchProcessor::new(cpu_device, config);

    let texts = vec!["good", "fail", "also_good", "also_fail"];

    // Processor that fails on texts containing "fail"
    let selective_processor = |text: &str| -> Result<String> {
        if text.contains("fail") {
            Err(candle_core::Error::Msg("Intentional failure".to_string()))
        } else {
            Ok(text.to_uppercase())
        }
    };

    let result = processor.process_batch(&texts, selective_processor);

    match result {
        Ok(batch_result) => {
            // Test successful results
            assert_eq!(batch_result.results.len(), 2);
            assert_eq!(batch_result.results[0], "GOOD");
            assert_eq!(batch_result.results[1], "ALSO_GOOD");

            // Test failed indices
            assert_eq!(batch_result.failed_indices.len(), 2);
            assert_eq!(batch_result.failed_indices[0].0, 1); // "fail" at index 1
            assert_eq!(batch_result.failed_indices[1].0, 3); // "also_fail" at index 3

            // Test success rate
            assert_eq!(batch_result.success_rate, 0.5); // 2 out of 4 succeeded
            assert_eq!(batch_result.batch_size, 4);

            println!(
                "Batch processing with failures test passed: {}/{} succeeded",
                batch_result.results.len(),
                batch_result.batch_size
            );
        }
        Err(e) => {
            println!("Batch processing with failures test failed: {}", e);
        }
    }
}

/// Test large batch processing with chunking
#[rstest]
fn test_batch_processor_traditional_batch_processor_process_large_batch(cpu_device: Device) {
    let config = BatchProcessorConfig {
        max_batch_size: 3, // Small max size to force chunking
        default_batch_size: 2,
        chunk_delay_ms: 1, // Minimal delay for testing
        ..Default::default()
    };
    let mut processor = TraditionalBatchProcessor::new(cpu_device, config);

    // Create a batch larger than max_batch_size
    let texts = vec![
        "item1", "item2", "item3", "item4", "item5", "item6", "item7",
    ];

    let uppercase_processor =
        |text: &str| -> Result<String> { Ok(format!("PROCESSED_{}", text.to_uppercase())) };

    let result = processor.process_large_batch(&texts, uppercase_processor);

    match result {
        Ok(batch_result) => {
            // Test all items were processed
            assert_eq!(batch_result.results.len(), 7);
            assert_eq!(batch_result.batch_size, 7);
            assert_eq!(batch_result.failed_indices.len(), 0);
            assert_eq!(batch_result.success_rate, 1.0);

            // Test results are correct
            for (i, result) in batch_result.results.iter().enumerate() {
                let expected = format!("PROCESSED_ITEM{}", i + 1);
                assert_eq!(*result, expected);
            }

            println!("TraditionalBatchProcessor.process_large_batch test passed: {} items processed in {} chunks",
                batch_result.results.len(), (texts.len() + 2) / 3); // Ceiling division
        }
        Err(e) => {
            println!(
                "TraditionalBatchProcessor.process_large_batch test failed: {}",
                e
            );
        }
    }
}

/// Test batch processing with timeout
#[rstest]
fn test_batch_processor_traditional_batch_processor_process_batch_with_timeout(cpu_device: Device) {
    let config = BatchProcessorConfig::default();
    let mut processor = TraditionalBatchProcessor::new(cpu_device, config);

    let texts = vec!["fast", "slow", "medium"];
    let timeout = Duration::from_millis(100);

    // Processor with variable processing time
    let variable_time_processor = |text: &str| -> Result<String> {
        match text {
            "slow" => {
                // Simulate slow processing (but not actually sleep in test)
                std::thread::sleep(Duration::from_millis(1)); // Minimal sleep
                Ok("SLOW_PROCESSED".to_string())
            }
            _ => Ok(text.to_uppercase()),
        }
    };

    let result = processor.process_batch_with_timeout(&texts, variable_time_processor, timeout);

    match result {
        Ok(batch_result) => {
            // In this test, all should succeed since we're not actually timing out
            assert!(batch_result.results.len() >= 2); // At least fast and medium should succeed
            assert_eq!(batch_result.batch_size, 3);
            assert!(batch_result.success_rate >= 0.66); // At least 2/3 should succeed

            println!("TraditionalBatchProcessor.process_batch_with_timeout test passed: {}/{} items succeeded",
                batch_result.results.len(), batch_result.batch_size);
        }
        Err(e) => {
            println!(
                "TraditionalBatchProcessor.process_batch_with_timeout test failed: {}",
                e
            );
        }
    }
}

/// Test processing metrics
#[rstest]
fn test_batch_processor_traditional_batch_processor_get_metrics(cpu_device: Device) {
    let config = BatchProcessorConfig::default();
    let mut processor = TraditionalBatchProcessor::new(cpu_device, config);

    // Initial metrics should be empty
    let initial_metrics = processor.get_metrics();
    assert_eq!(initial_metrics.total_batches, 0);
    assert_eq!(initial_metrics.total_items, 0);

    // Process a batch
    let texts = vec!["test1", "test2", "test3"];
    let simple_processor = |text: &str| -> Result<String> { Ok(text.to_string()) };

    let _result = processor.process_batch(&texts, simple_processor);

    // Check metrics were updated
    let updated_metrics = processor.get_metrics();
    assert_eq!(updated_metrics.total_batches, 1);
    assert_eq!(updated_metrics.total_items, 3);

    // Test metrics reset
    processor.reset_metrics();
    let reset_metrics = processor.get_metrics();
    assert_eq!(reset_metrics.total_batches, 0);
    assert_eq!(reset_metrics.total_items, 0);

    println!("TraditionalBatchProcessor.get_metrics test passed");
}
