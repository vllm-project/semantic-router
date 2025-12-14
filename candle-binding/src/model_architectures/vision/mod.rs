//! Vision Transformer Module
//!
//! This module provides vision transformer integration for image feature extraction.
//! Currently implements CLIP (Contrastive Language-Image Pre-training) vision encoder.

pub mod image_utils;
pub mod clip_encoder;

pub use image_utils::{preprocess_image, ImagePreprocessingError};
pub use clip_encoder::{ClipVisionEncoder, VisionEncoder};


