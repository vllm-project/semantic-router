//! Unit tests for FFI vision transformer functions
//!
//! Following .cursorrules Line 20-25 specifications:
//! - Test framework: rstest (parameterized testing)
//! - Concurrency control: serial_test (#[serial] for serial execution)
//! - File naming: vision.rs → vision_test.rs
//! - Location: Same directory as source file
//!
//! Note: These tests require the vision encoder to be initialized.
//! Use the `setup_vision_encoder` fixture to initialize the encoder before testing.

use super::vision::*;
use rstest::*;
use serial_test::serial;
use std::ffi::CString;
use std::sync::Once;
use std::io::Cursor;
use image::{ImageBuffer, Rgb, RgbImage, DynamicImage, ImageFormat};

/// Global initializer to ensure vision encoder is initialized once
static INIT: Once = Once::new();

/// Create a simple test image (224x224 RGB JPEG)
///
/// This creates a minimal valid JPEG image that can be used for testing.
/// The image is a simple red square with white text "TEST" in the center.
fn create_test_image() -> Vec<u8> {
    // Create a 224x224 RGB image (CLIP's expected input size)
    let mut img: RgbImage = ImageBuffer::new(224, 224);
    
    // Fill with red color
    for pixel in img.pixels_mut() {
        *pixel = Rgb([255u8, 0u8, 0u8]);
    }
    
    // Draw a simple pattern (white square in center)
    let center_x = 112;
    let center_y = 112;
    let size = 50;
    for y in (center_y - size)..(center_y + size) {
        for x in (center_x - size)..(center_x + size) {
            if y < 224 && x < 224 {
                img.put_pixel(x, y, Rgb([255u8, 255u8, 255u8]));
            }
        }
    }
    
    // Encode as JPEG
    let dynamic_img = DynamicImage::ImageRgb8(img);
    let mut bytes = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    dynamic_img
        .write_to(&mut cursor, ImageFormat::Jpeg)
        .expect("Failed to encode test image");
    
    bytes
}

/// Setup fixture: Initialize vision encoder before tests
///
/// This fixture initializes the vision encoder with the default CLIP model.
/// It uses Once to ensure initialization happens only once across all tests.
#[fixture]
fn setup_vision_encoder() {
    INIT.call_once(|| {
        // Use default CLIP model
        let model_id = CString::new("openai/clip-vit-base-patch32").unwrap();
        let device = CString::new("cpu").unwrap();
        
        let success = init_vision_encoder_ffi(model_id.as_ptr(), device.as_ptr());
        
        if !success {
            panic!("Failed to initialize vision encoder for FFI tests");
        }
        
        println!("✅ Vision encoder initialized for FFI tests");
    });
}

/// Test init_vision_encoder_ffi with valid parameters
#[rstest]
#[serial]
fn test_init_vision_encoder_ffi_valid(_setup_vision_encoder: ()) {
    // This test verifies that initialization works
    // The setup fixture already initializes it, so we just verify it's working
    // by testing that we can get embeddings
    let test_image = create_test_image();
    let mime_type = CString::new("image/jpeg").unwrap();
    
    let result_ptr = get_image_embedding(
        test_image.as_ptr(),
        test_image.len(),
        mime_type.as_ptr(),
    );
    
    assert!(!result_ptr.is_null(), "Result pointer should not be null");
    
    unsafe {
        let result = &*result_ptr;
        assert_eq!(result.error, false, "Should not have error");
        assert_eq!(result.length, 512, "CLIP embeddings should be 512 dimensions");
        assert!(!result.data.is_null(), "Data pointer should not be null");
        assert!(result.processing_time_ms >= 0.0, "Should have valid processing time");
        
        // Cleanup
        free_image_embedding_result(result_ptr);
    }
}

/// Test get_image_embedding with valid JPEG image
#[rstest]
#[serial]
fn test_get_image_embedding_valid_jpeg(_setup_vision_encoder: ()) {
    let test_image = create_test_image();
    let mime_type = CString::new("image/jpeg").unwrap();
    
    let result_ptr = get_image_embedding(
        test_image.as_ptr(),
        test_image.len(),
        mime_type.as_ptr(),
    );
    
    assert!(!result_ptr.is_null(), "Result pointer should not be null");
    
    unsafe {
        let result = &*result_ptr;
        assert_eq!(result.error, false, "Should not have error");
        assert_eq!(result.length, 512, "CLIP embeddings should be 512 dimensions");
        assert!(!result.data.is_null(), "Data pointer should not be null");
        
        // Verify embedding values are reasonable (not all zeros, not NaN)
        let embedding_slice = std::slice::from_raw_parts(result.data, result.length as usize);
        let mut has_non_zero = false;
        for &val in embedding_slice {
            assert!(!val.is_nan(), "Embedding values should not be NaN");
            assert!(!val.is_infinite(), "Embedding values should not be infinite");
            if val != 0.0 {
                has_non_zero = true;
            }
        }
        assert!(has_non_zero, "Embedding should have at least some non-zero values");
        
        // Cleanup
        free_image_embedding_result(result_ptr);
    }
}

/// Test get_image_embedding with valid PNG image
#[rstest]
#[serial]
fn test_get_image_embedding_valid_png(_setup_vision_encoder: ()) {
    // Create a simple PNG image
    let mut img: RgbImage = ImageBuffer::new(224, 224);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0u8, 255u8, 0u8]); // Green
    }
    
    let dynamic_img = DynamicImage::ImageRgb8(img);
    let mut bytes = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    dynamic_img
        .write_to(&mut cursor, ImageFormat::Png)
        .expect("Failed to encode PNG");
    
    let mime_type = CString::new("image/png").unwrap();
    
    let result_ptr = get_image_embedding(
        bytes.as_ptr(),
        bytes.len(),
        mime_type.as_ptr(),
    );
    
    assert!(!result_ptr.is_null(), "Result pointer should not be null");
    
    unsafe {
        let result = &*result_ptr;
        assert_eq!(result.error, false, "Should not have error for PNG");
        assert_eq!(result.length, 512, "CLIP embeddings should be 512 dimensions");
        
        // Cleanup
        free_image_embedding_result(result_ptr);
    }
}

/// Test get_image_embedding with invalid image data
#[rstest]
#[serial]
fn test_get_image_embedding_invalid_data(_setup_vision_encoder: ()) {
    // Invalid image data (not a valid JPEG/PNG)
    let invalid_data = vec![0u8, 1u8, 2u8, 3u8, 4u8];
    let mime_type = CString::new("image/jpeg").unwrap();
    
    let result_ptr = get_image_embedding(
        invalid_data.as_ptr(),
        invalid_data.len(),
        mime_type.as_ptr(),
    );
    
    assert!(!result_ptr.is_null(), "Result pointer should not be null");
    
    unsafe {
        let result = &*result_ptr;
        // Should have error for invalid image data
        assert_eq!(result.error, true, "Should have error for invalid image data");
        
        // Cleanup
        free_image_embedding_result(result_ptr);
    }
}

/// Test get_image_embedding with invalid MIME type
#[rstest]
#[serial]
fn test_get_image_embedding_invalid_mime(_setup_vision_encoder: ()) {
    let test_image = create_test_image();
    let invalid_mime = CString::new("invalid/mime").unwrap();
    
    let result_ptr = get_image_embedding(
        test_image.as_ptr(),
        test_image.len(),
        invalid_mime.as_ptr(),
    );
    
    assert!(!result_ptr.is_null(), "Result pointer should not be null");
    
    unsafe {
        let result = &*result_ptr;
        // Should have error for invalid MIME type
        assert_eq!(result.error, true, "Should have error for invalid MIME type");
        
        // Cleanup
        free_image_embedding_result(result_ptr);
    }
}

/// Test get_image_embedding with empty image data
#[rstest]
#[serial]
fn test_get_image_embedding_empty_data(_setup_vision_encoder: ()) {
    let empty_data = Vec::<u8>::new();
    let mime_type = CString::new("image/jpeg").unwrap();
    
    let result_ptr = get_image_embedding(
        empty_data.as_ptr(),
        empty_data.len(),
        mime_type.as_ptr(),
    );
    
    assert!(!result_ptr.is_null(), "Result pointer should not be null");
    
    unsafe {
        let result = &*result_ptr;
        // Should have error for empty data
        assert_eq!(result.error, true, "Should have error for empty image data");
        
        // Cleanup
        free_image_embedding_result(result_ptr);
    }
}

/// Test that embeddings are consistent for the same image
#[rstest]
#[serial]
fn test_get_image_embedding_consistency(_setup_vision_encoder: ()) {
    let test_image = create_test_image();
    let mime_type = CString::new("image/jpeg").unwrap();
    
    // Get embedding twice
    let result_ptr1 = get_image_embedding(
        test_image.as_ptr(),
        test_image.len(),
        mime_type.as_ptr(),
    );
    
    let result_ptr2 = get_image_embedding(
        test_image.as_ptr(),
        test_image.len(),
        mime_type.as_ptr(),
    );
    
    assert!(!result_ptr1.is_null() && !result_ptr2.is_null());
    
    unsafe {
        let result1 = &*result_ptr1;
        let result2 = &*result_ptr2;
        
        assert_eq!(result1.error, false);
        assert_eq!(result2.error, false);
        assert_eq!(result1.length, result2.length);
        
        // Embeddings should be identical (deterministic)
        let length = result1.length as usize;
        let emb1 = std::slice::from_raw_parts(result1.data, length);
        let emb2 = std::slice::from_raw_parts(result2.data, length);
        
        for i in 0..length {
            assert!(
                (emb1[i] - emb2[i]).abs() < 1e-5,
                "Embeddings should be identical (within floating point precision)"
            );
        }
        
        // Cleanup
        free_image_embedding_result(result_ptr1);
        free_image_embedding_result(result_ptr2);
    }
}

/// Test free_image_embedding_result with null pointer (should not crash)
#[rstest]
#[serial]
fn test_free_image_embedding_result_null(_setup_vision_encoder: ()) {
    // Should not crash when freeing null pointer
    free_image_embedding_result(std::ptr::null_mut());
    // If we get here, the test passed
}

