//! Bounded JPEG/PNG decoding for multimodal image embeddings.

use image::{ImageFormat, ImageReader, Limits};
use std::io::Cursor;

pub(super) const MAX_MULTIMODAL_IMAGE_DIMENSION: u32 = 8192;
pub(super) const MAX_MULTIMODAL_IMAGE_PIXELS: u64 = 16_777_216;
pub(super) const MAX_MULTIMODAL_IMAGE_ENCODED_BYTES: usize = 20 * 1024 * 1024;
const MAX_MULTIMODAL_IMAGE_DECODE_BYTES: u64 = 64 * 1024 * 1024;

type BoundedImageReader<'a> = ImageReader<Cursor<&'a [u8]>>;

fn decoder_limits() -> Limits {
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_MULTIMODAL_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_MULTIMODAL_IMAGE_DIMENSION);
    limits.max_alloc = Some(MAX_MULTIMODAL_IMAGE_DECODE_BYTES);

    limits
}

fn image_reader(bytes: &[u8]) -> Result<(BoundedImageReader<'_>, ImageFormat), String> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("image format detection failed: {e:?}"))?;
    let format = reader
        .format()
        .ok_or_else(|| "image format could not be detected".to_string())?;
    if !matches!(format, ImageFormat::Jpeg | ImageFormat::Png) {
        return Err(format!("unsupported image format: {format:?}"));
    }
    reader.limits(decoder_limits());
    Ok((reader, format))
}

fn image_reader_with_format(bytes: &[u8], format: ImageFormat) -> BoundedImageReader<'_> {
    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    reader.limits(decoder_limits());
    reader
}

/// Decode bounded JPEG/PNG bytes, resize them, and return planar CHW f32 data.
///
/// Width/height and pixel-area preflight prevent compressed-image bombs before
/// allocating the source pixel buffer. The decoder allocation budget adds a
/// second bound for codec intermediates. Catmull-Rom resizing preserves the
/// established SigLIP preprocessing parity with PIL bicubic antialiasing.
pub(super) fn decode_resize_to_chw_f32(
    bytes: &[u8],
    target_w: u32,
    target_h: u32,
) -> Result<Vec<f32>, String> {
    if bytes.len() > MAX_MULTIMODAL_IMAGE_ENCODED_BYTES {
        return Err(format!(
            "encoded image has {} bytes; maximum is {MAX_MULTIMODAL_IMAGE_ENCODED_BYTES}",
            bytes.len()
        ));
    }

    let w = target_w as usize;
    let h = target_h as usize;
    let n_pixels = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| format!("target dimensions overflow usize: {target_w}x{target_h}x3"))?;

    let (metadata_reader, source_format) = image_reader(bytes)?;
    let (source_w, source_h) = metadata_reader
        .into_dimensions()
        .map_err(|e| format!("image decode failed during metadata preflight: {e:?}"))?;
    let source_pixels = u64::from(source_w)
        .checked_mul(u64::from(source_h))
        .ok_or_else(|| "source image pixel count overflow".to_string())?;
    if source_pixels > MAX_MULTIMODAL_IMAGE_PIXELS {
        return Err(format!(
            "source image has {source_pixels} pixels; maximum is {MAX_MULTIMODAL_IMAGE_PIXELS}"
        ));
    }

    let image = image_reader_with_format(bytes, source_format)
        .decode()
        .map_err(|e| format!("image decode failed: {e:?}"))?
        .into_rgb8();
    let resized = image::imageops::resize(
        &image,
        target_w,
        target_h,
        image::imageops::FilterType::CatmullRom,
    );

    let raw = resized.as_raw();
    debug_assert_eq!(raw.len(), n_pixels);
    let mut pixels = vec![0f32; n_pixels];
    let plane = h * w;
    let inv = 1.0f32 / 255.0;
    for index in 0..plane {
        let source = index * 3;
        pixels[index] = raw[source] as f32 * inv;
        pixels[plane + index] = raw[source + 1] as f32 * inv;
        pixels[2 * plane + index] = raw[source + 2] as f32 * inv;
    }
    Ok(pixels)
}

#[cfg(test)]
pub(crate) mod test_support {
    use image::{ImageBuffer, ImageFormat, Rgb};
    use std::io::Cursor;

    pub(crate) fn make_test_png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(width, height, |_, _| Rgb(rgb));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .expect("failed to encode test PNG");
        bytes
    }

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = u32::MAX;
        for &byte in bytes {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xedb8_8320 & mask);
            }
        }
        !crc
    }

    pub(crate) fn make_png_with_declared_dimensions(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = make_test_png(1, 1, [1, 2, 3]);
        bytes[16..20].copy_from_slice(&width.to_be_bytes());
        bytes[20..24].copy_from_slice(&height.to_be_bytes());
        let ihdr_crc = crc32(&bytes[12..29]);
        bytes[29..33].copy_from_slice(&ihdr_crc.to_be_bytes());
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{make_png_with_declared_dimensions, make_test_png};
    use super::*;

    #[test]
    fn decode_produces_chw_layout_in_unit_range() {
        let pixels = decode_resize_to_chw_f32(&make_test_png(4, 4, [255, 128, 64]), 8, 8)
            .expect("valid PNG should decode");
        let plane = 8 * 8;
        assert_eq!(pixels.len(), 3 * plane);
        assert!(pixels.iter().all(|value| (0.0..=1.0).contains(value)));
        assert!((pixels[0] - 1.0).abs() < 0.05);
        assert!((pixels[plane] - 128.0 / 255.0).abs() < 0.05);
        assert!((pixels[2 * plane] - 64.0 / 255.0).abs() < 0.05);
    }

    #[test]
    fn decode_rejects_invalid_bytes_and_target_overflow() {
        let invalid = decode_resize_to_chw_f32(b"not an image", 8, 8).unwrap_err();
        assert!(invalid.contains("format"));

        let png = make_test_png(2, 2, [128, 128, 128]);
        let overflow = decode_resize_to_chw_f32(&png, u32::MAX, u32::MAX).unwrap_err();
        assert!(overflow.contains("overflow"));
    }

    #[test]
    fn decode_rejects_oversized_dimensions_and_pixel_area() {
        let dimension_bomb =
            make_png_with_declared_dimensions(MAX_MULTIMODAL_IMAGE_DIMENSION + 1, 1);
        assert!(decode_resize_to_chw_f32(&dimension_bomb, 8, 8).is_err());

        let side = (MAX_MULTIMODAL_IMAGE_PIXELS as f64).sqrt() as u32 + 1;
        let pixel_bomb = make_png_with_declared_dimensions(side, side);
        let error = decode_resize_to_chw_f32(&pixel_bomb, 8, 8).unwrap_err();
        assert!(error.contains("maximum"));
    }
}
