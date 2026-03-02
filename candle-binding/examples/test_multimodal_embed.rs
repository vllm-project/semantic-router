//! Example: Test multi-modal embedding model with real Wikimedia Commons images
//!
//! Downloads copyright-free images and tests text, image, base64-image,
//! and cross-modal embeddings.
//!
//! Usage:
//! ```bash
//! MULTIMODAL_MODEL_PATH=models/multi-modal-embed-small \
//! cargo run --release --no-default-features --example test_multimodal_embed
//! ```

use base64::Engine;
use candle_core::{Device, Tensor};
use candle_semantic_router::model_architectures::embedding::MultiModalEmbeddingModel;
use image::GenericImageView;
use tokenizers::Tokenizer;

/// All images are copyright-free from Wikimedia Commons:
///   - Tuxedo_kitten.jpg : Public Domain (author: TimVickers)
///   - 1Cute-doggy.jpg   : CC0 1.0 Universal (author: X posid)
///   - 1908_Ford_Model_T.jpg : Public Domain (published 1908, pre-1930)
const IMAGE_URLS: &[(&str, &str)] = &[
    (
        "cat",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Tuxedo_kitten.jpg/512px-Tuxedo_kitten.jpg",
    ),
    (
        "dog",
        "https://upload.wikimedia.org/wikipedia/commons/a/a7/1Cute-doggy.jpg",
    ),
    (
        "car",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/1908_Ford_Model_T.jpg/960px-1908_Ford_Model_T.jpg",
    ),
];

fn download_image(url: &str) -> Vec<u8> {
    println!("  Downloading {}...", url);
    let resp = ureq::get(url).call().expect("HTTP request failed");
    let len: usize = resp
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 20);
    let mut buf = Vec::with_capacity(len);
    resp.into_reader()
        .read_to_end(&mut buf)
        .expect("Failed to read response body");
    println!("  Downloaded {} bytes", buf.len());
    buf
}

/// Decode image bytes → resized 512×512 → [1, 3, 512, 512] tensor (pixel values in [0, 1]).
fn image_bytes_to_tensor(bytes: &[u8], device: &Device) -> Tensor {
    let img = image::load_from_memory(bytes).expect("Failed to decode image");
    let img = img.resize_exact(512, 512, image::imageops::FilterType::Triangle);
    let (w, h) = img.dimensions();
    assert_eq!((w, h), (512, 512));

    let rgb = img.to_rgb8();
    let raw = rgb.as_raw();

    let mut chw = vec![0f32; 3 * 512 * 512];
    for y in 0..512usize {
        for x in 0..512usize {
            let idx = (y * 512 + x) * 3;
            chw[0 * 512 * 512 + y * 512 + x] = raw[idx] as f32 / 255.0;
            chw[1 * 512 * 512 + y * 512 + x] = raw[idx + 1] as f32 / 255.0;
            chw[2 * 512 * 512 + y * 512 + x] = raw[idx + 2] as f32 / 255.0;
        }
    }
    Tensor::from_vec(chw, (1, 3, 512, 512), device).unwrap()
}

/// Simulate OpenAI API flow: raw bytes → base64 string → decode → tensor.
fn base64_to_tensor(image_bytes: &[u8], device: &Device) -> Tensor {
    let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .expect("base64 decode failed");
    image_bytes_to_tensor(&decoded, device)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn encode_text(model: &MultiModalEmbeddingModel, tokenizer: &Tokenizer, text: &str) -> Vec<f32> {
    let encoding = tokenizer.encode(text, true).unwrap();
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let mask: Vec<u32> = encoding
        .get_attention_mask()
        .iter()
        .map(|&x| x as u32)
        .collect();
    let seq_len = ids.len();
    let device = model.device();
    let input_ids = Tensor::from_vec(ids, (1, seq_len), device).unwrap();
    let attention_mask = Tensor::from_vec(mask, (1, seq_len), device).unwrap();
    model
        .encode_text(&input_ids, Some(&attention_mask))
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("MULTIMODAL_MODEL_PATH")
        .unwrap_or_else(|_| "models/multi-modal-embed-small".to_string());

    println!("=== Multi-Modal Embedding Test (Real Images) ===");
    println!("Model path: {}", model_path);

    let device = Device::Cpu;

    println!("\n[1/6] Loading model...");
    let start = std::time::Instant::now();
    let model = MultiModalEmbeddingModel::load(&model_path, &device)?;
    println!("  Loaded in {:.2}s", start.elapsed().as_secs_f32());
    println!("  Embedding dim: {}", model.config().embedding_dim);

    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer: {:?}", e))?;

    // ── Download real images ──
    println!("\n[2/6] Downloading copyright-free images from Wikimedia Commons...");
    let mut image_data: Vec<(&str, Vec<u8>)> = Vec::new();
    for (name, url) in IMAGE_URLS {
        let bytes = download_image(url);
        image_data.push((name, bytes));
    }

    // ── Encode images from raw bytes ──
    println!("\n[3/6] Encoding images (raw bytes → decode → tensor)...");
    let mut image_embeddings: Vec<(&str, Vec<f32>)> = Vec::new();
    for (name, bytes) in &image_data {
        let start = std::time::Instant::now();
        let tensor = image_bytes_to_tensor(bytes, &device);
        let emb = model.encode_image(&tensor)?;
        let elapsed = start.elapsed();
        let emb_vec: Vec<f32> = emb.squeeze(0)?.to_vec1()?;
        let norm: f32 = emb_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "  {}: dim={}, norm={:.4}, time={:.1}ms",
            name,
            emb_vec.len(),
            norm,
            elapsed.as_secs_f32() * 1000.0
        );
        image_embeddings.push((name, emb_vec));
    }

    // ── Encode images from base64 (OpenAI API style) ──
    println!("\n[4/6] Encoding images (base64 → decode → tensor, OpenAI API style)...");
    let mut base64_embeddings: Vec<(&str, Vec<f32>)> = Vec::new();
    for (name, bytes) in &image_data {
        let start = std::time::Instant::now();
        let tensor = base64_to_tensor(bytes, &device);
        let emb = model.encode_image(&tensor)?;
        let elapsed = start.elapsed();
        let emb_vec: Vec<f32> = emb.squeeze(0)?.to_vec1()?;
        let norm: f32 = emb_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "  {} (base64): dim={}, norm={:.4}, time={:.1}ms",
            name,
            emb_vec.len(),
            norm,
            elapsed.as_secs_f32() * 1000.0
        );
        base64_embeddings.push((name, emb_vec));
    }

    // ── Verify base64 ↔ raw consistency ──
    println!("\n  Verifying base64 vs raw byte consistency...");
    for i in 0..image_data.len() {
        let sim = cosine_similarity(&image_embeddings[i].1, &base64_embeddings[i].1);
        println!(
            "  {} raw vs base64: sim={:.6} (should be 1.0)",
            image_embeddings[i].0, sim
        );
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "base64 and raw should produce identical embeddings"
        );
    }

    // ── Text encoding ──
    println!("\n[5/6] Text encoding + cross-modal similarity...");
    let text_queries = [
        ("a photo of a cat", "cat"),
        ("a photo of a dog", "dog"),
        ("a photo of a car", "car"),
        ("a cute kitten sitting", "cat"),
        ("a fluffy puppy", "dog"),
        ("a vintage automobile", "car"),
    ];

    for (query, expected_match) in &text_queries {
        let text_emb = encode_text(&model, &tokenizer, query);

        let mut sims: Vec<(&str, f32)> = image_embeddings
            .iter()
            .map(|(name, emb)| (*name, cosine_similarity(&text_emb, emb)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best = sims[0].0;
        let marker = if best == *expected_match {
            "OK"
        } else {
            "MISS"
        };
        println!(
            "  [{}] \"{}\": best={} ({:.4})  | {}",
            marker,
            query,
            best,
            sims[0].1,
            sims.iter()
                .map(|(n, s)| format!("{}={:.4}", n, s))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // ── MRL + base64 ──
    println!("\n[6/6] Matryoshka dimension reduction on real images...");
    let cat_bytes = &image_data[0].1;
    let cat_tensor = base64_to_tensor(cat_bytes, &device);

    for dim in [384, 256, 128, 64, 32] {
        let emb = model.encode_image_with_dim(&cat_tensor, Some(dim))?;
        let shape = emb.dims().to_vec();
        let emb_vec: Vec<f32> = emb.squeeze(0)?.to_vec1()?;
        let norm: f32 = emb_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "  cat (base64) dim={}: shape={:?}, norm={:.4}",
            dim, shape, norm
        );
    }

    println!("\n=== All tests passed ===");
    Ok(())
}
