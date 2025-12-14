//! Image preprocessing utilities for vision transformers
//!
//! Handles image decoding, resizing, normalization, and tensor conversion
//! for CLIP and other vision transformer models.

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};

#[derive(Debug, Clone)]
pub enum ImagePreprocessingError {
    DecodeError(String),
    ResizeError(String),
    ConversionError(String),
}

impl std::fmt::Display for ImagePreprocessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImagePreprocessingError::DecodeError(msg) => write!(f, "Decode error: {}", msg),
            ImagePreprocessingError::ResizeError(msg) => write!(f, "Resize error: {}", msg),
            ImagePreprocessingError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
        }
    }
}

impl std::error::Error for ImagePreprocessingError {}

/// Preprocess image for vision transformer (CLIP)
///
/// Steps:
/// 1. Decode image from bytes (JPEG/PNG)
/// 2. Resize to 224x224 (CLIP standard size) with center crop
/// 3. Normalize pixel values [0-255] → [-1.0, 1.0] (CLIP normalization)
/// 4. Convert to tensor [1, 3, 224, 224] (batch, channels, height, width)
///
/// # Arguments
/// - `image_data`: Raw image bytes (JPEG, PNG, etc.)
/// - `mime_type`: MIME type of the image (e.g., "image/jpeg", "image/png")
/// - `device`: Device to create tensor on
///
/// # Returns
/// - `Ok(Tensor)`: Preprocessed image tensor [1, 3, 224, 224]
/// - `Err(ImagePreprocessingError)`: If preprocessing fails
pub fn preprocess_image(
    image_data: &[u8],
    mime_type: &str,
    device: &Device,
) -> Result<Tensor, ImagePreprocessingError> {
    // 1. Decode image
    let img = decode_image(image_data, mime_type)?;

    // 2. Resize to 224x224 with center crop
    let resized = resize_and_center_crop(&img, 224, 224)?;

    // 3. Normalize and convert to tensor
    let tensor = image_to_tensor(&resized, device)?;

    Ok(tensor)
}

fn decode_image(data: &[u8], mime_type: &str) -> Result<DynamicImage, ImagePreprocessingError> {
    match mime_type {
        "image/jpeg" | "image/jpg" => {
            image::load_from_memory(data)
                .map_err(|e| ImagePreprocessingError::DecodeError(e.to_string()))
        }
        "image/png" => {
            image::load_from_memory(data)
                .map_err(|e| ImagePreprocessingError::DecodeError(e.to_string()))
        }
        _ => Err(ImagePreprocessingError::DecodeError(
            format!("Unsupported MIME type: {}", mime_type),
        )),
    }
}

/// Resize image to target size with center crop
///
/// CLIP preprocessing:
/// 1. Resize shorter side to target_size while maintaining aspect ratio
/// 2. Center crop to target_size x target_size
fn resize_and_center_crop(
    img: &DynamicImage,
    target_size: u32,
    crop_size: u32,
) -> Result<RgbImage, ImagePreprocessingError> {
    // Convert to RGB
    let rgb_img = img.to_rgb8();
    let (orig_width, orig_height) = rgb_img.dimensions();

    // Calculate resize dimensions (maintain aspect ratio, resize shorter side)
    let (resize_width, resize_height) = if orig_width < orig_height {
        (target_size, (orig_height * target_size) / orig_width)
    } else {
        ((orig_width * target_size) / orig_height, target_size)
    };

    // Resize using image crate's resize function (simpler and more reliable)
    let resized = image::imageops::resize(
        &rgb_img,
        resize_width,
        resize_height,
        image::imageops::FilterType::Lanczos3,
    );

    // Center crop
    let crop_x = (resize_width.saturating_sub(crop_size)) / 2;
    let crop_y = (resize_height.saturating_sub(crop_size)) / 2;

    let cropped = image::imageops::crop_imm(&resized, crop_x, crop_y, crop_size, crop_size)
        .to_image();

    Ok(cropped)
}

/// Convert image to tensor with CLIP normalization
///
/// CLIP normalization:
/// - Mean: [0.485, 0.456, 0.406] (ImageNet mean)
/// - Std: [0.229, 0.224, 0.225] (ImageNet std)
/// - Formula: (pixel / 255.0 - mean) / std
fn image_to_tensor(img: &RgbImage, device: &Device) -> Result<Tensor, ImagePreprocessingError> {
    let (width, height) = img.dimensions();
    let mut pixels = Vec::new();

    // CLIP normalization constants (ImageNet)
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    // Convert image to CHW format (Channels, Height, Width)
    // Normalize: (pixel / 255.0 - mean) / std
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                let value = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                pixels.push(value);
            }
        }
    }

    // Create tensor: [1, 3, 224, 224]
    let tensor = Tensor::from_vec(pixels, (1, 3, height as usize, width as usize), device)
        .map_err(|e| ImagePreprocessingError::ConversionError(e.to_string()))?;
    tensor
        .to_dtype(DType::F32)
        .map_err(|e| ImagePreprocessingError::ConversionError(e.to_string()))
}


