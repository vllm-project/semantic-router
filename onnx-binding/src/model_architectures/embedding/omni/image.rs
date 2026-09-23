//! PIL-compatible RGB resize: antialiasing and 22-bit coefficients, with an
//! 8-bit intermediate after each pass, before the artifact's normalization.
use super::manifest::ImageProcessor;

/// Match Pillow's libjpeg-turbo decode before applying the published processor.
/// The Rust JPEG decoder differs in chroma upsampling on ordinary 4:2:0 images.
pub(super) fn decode_rgb(bytes: &[u8]) -> anyhow::Result<image::RgbImage> {
    if !bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Ok(image::load_from_memory(bytes)?.to_rgb8());
    }
    let mut decoder = turbojpeg::Decompressor::new()?;
    decoder.set_fast_upsample(false)?;
    let header = decoder.read_header(bytes)?;
    let cmyk = matches!(
        header.colorspace,
        turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK
    );
    let channels = if cmyk { 4 } else { 3 };
    let pitch = header
        .width
        .checked_mul(channels)
        .ok_or_else(|| anyhow::anyhow!("JPEG dimensions overflow"))?;
    let length = pitch
        .checked_mul(header.height)
        .ok_or_else(|| anyhow::anyhow!("JPEG dimensions overflow"))?;
    if let Some(limit) = image::Limits::default().max_alloc {
        anyhow::ensure!(
            length as u64 <= limit,
            "JPEG exceeds decoded image memory limit"
        );
    }
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(length)?;
    pixels.resize(length, 0);
    decoder.decompress(
        bytes,
        turbojpeg::Image {
            pixels: &mut pixels[..],
            width: header.width,
            height: header.height,
            pitch,
            format: if cmyk {
                turbojpeg::PixelFormat::CMYK
            } else {
                turbojpeg::PixelFormat::RGB
            },
        },
    )?;
    if cmyk {
        // Pillow reads CMYK JPEG with Adobe's inverted channels (CMYK;I),
        // then converts CMYK to RGB using rounded multiplication by K.
        pixels = pixels
            .chunks_exact(4)
            .flat_map(|p| {
                let k = u32::from(p[3]);
                [p[0], p[1], p[2]].map(|c| {
                    let product = (255 - u32::from(c)) * k + 128;
                    (k - ((product + (product >> 8)) >> 8)) as u8
                })
            })
            .collect();
    }
    image::RgbImage::from_raw(header.width.try_into()?, header.height.try_into()?, pixels)
        .ok_or_else(|| anyhow::anyhow!("invalid JPEG RGB dimensions"))
}

fn coefficients(input: usize, output: usize, cubic: bool) -> Vec<(usize, Vec<i64>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = if cubic { 2.0 } else { 1.0 } * filter_scale;
    (0..output)
        .map(|x| {
            let center = (x as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let weights: Vec<f64> = (start..end)
                .map(|i| {
                    let t = ((i as f64 - center + 0.5) / filter_scale).abs();
                    if !cubic {
                        (1.0 - t).max(0.0)
                    } else if t < 1.0 {
                        ((1.5 * t - 2.5) * t) * t + 1.0
                    } else if t < 2.0 {
                        ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let sum: f64 = weights.iter().sum();
            (
                start,
                weights
                    .into_iter()
                    .map(|v| (v / sum * (1u64 << 22) as f64).round() as i64)
                    .collect(),
            )
        })
        .collect()
}
fn clip(sum: i64) -> u8 {
    (sum >> 22).clamp(0, 255) as u8
}
pub fn pixels(bytes: &[u8], processor: &ImageProcessor) -> anyhow::Result<Vec<f32>> {
    let image = decode_rgb(bytes)?;
    let (width, height) = (image.width() as usize, image.height() as usize);
    anyhow::ensure!(width > 0 && height > 0, "empty image");
    let size = processor.size;
    let cubic = processor.resample == "bicubic";
    let horizontal = coefficients(width, size, cubic);
    let vertical = coefficients(height, size, cubic);
    let mut intermediate = vec![0u8; size * height * 3];
    let source = image.as_raw();
    for y in 0..height {
        for (x, (start, weights)) in horizontal.iter().enumerate() {
            for c in 0..3 {
                let sum = (1i64 << 21)
                    + weights
                        .iter()
                        .enumerate()
                        .map(|(i, w)| i64::from(source[(y * width + start + i) * 3 + c]) * w)
                        .sum::<i64>();
                intermediate[(y * size + x) * 3 + c] = clip(sum);
            }
        }
    }
    let mut output = vec![0f32; size * size * 3];
    for (y, (start, weights)) in vertical.iter().enumerate() {
        for x in 0..size {
            for c in 0..3 {
                let sum = (1i64 << 21)
                    + weights
                        .iter()
                        .enumerate()
                        .map(|(i, w)| i64::from(intermediate[((start + i) * size + x) * 3 + c]) * w)
                        .sum::<i64>();
                // HF rescales to float32 before normalization.
                let pixel = (f64::from(clip(sum)) / 255.0) as f32;
                output[c * size * size + y * size + x] =
                    (pixel - processor.mean[c]) / processor.std[c];
            }
        }
    }
    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coefficients_preserve_constants_and_use_antialiasing() {
        for cubic in [false, true] {
            for (from, to) in [(3, 17), (17, 3)] {
                for (_, weights) in coefficients(from, to, cubic) {
                    assert!((weights.iter().sum::<i64>() - (1 << 22)).abs() < 8);
                    assert_eq!(
                        clip((1 << 21) + weights.iter().map(|w| w * 127).sum::<i64>()),
                        127
                    );
                }
            }
        }
    }
}
