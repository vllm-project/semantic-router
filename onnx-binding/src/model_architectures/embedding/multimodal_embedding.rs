//! Multi-Modal Embedding Model using ONNX Runtime
//!
//! Loads three ONNX sessions (text, image, audio) from a single model directory.
//! Mirrors the candle-binding `MultiModalEmbeddingModel` API.
//!
//! ## Expected directory layout
//! ```text
//! <model_path>/
//! ├── text_encoder.onnx      — MiniLM-L6-v2 (input_ids, attention_mask → 384-dim)
//! ├── image_encoder.onnx     — SigLIP + projection (pixel_values → 384-dim)
//! ├── audio_encoder.onnx     — Whisper-tiny (mel_spectrogram → 384-dim)
//! ├── tokenizer.json
//! └── config.json            — {"embedding_dim": 384, ...}
//! ```

use crate::core::unified_error::{errors, UnifiedResult};
use ndarray::Array1;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct MultiModalConfig {
    pub embedding_dim: usize,
    pub matryoshka_dims: Vec<usize>,
    pub image_size: usize,
    pub n_mels: usize,
    pub max_seq_len: usize,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384,
            matryoshka_dims: vec![384, 256, 128, 64, 32],
            image_size: 512,
            n_mels: 80,
            max_seq_len: 512,
        }
    }
}

impl MultiModalConfig {
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> UnifiedResult<Self> {
        let config_path = model_path.as_ref().join("config.json");
        if !config_path.exists() {
            return Ok(Self::default());
        }
        let s = std::fs::read_to_string(&config_path)
            .map_err(|_| errors::file_not_found(&config_path.display().to_string()))?;
        let v: serde_json::Value = serde_json::from_str(&s)
            .map_err(|e| errors::invalid_json(&config_path.display().to_string(), &e.to_string()))?;
        let mut cfg = Self::default();
        if let Some(d) = v["embedding_dim"].as_u64() {
            cfg.embedding_dim = d as usize;
        }
        if let Some(s) = v.get("image_encoder").and_then(|o| o["image_size"].as_u64()) {
            cfg.image_size = s as usize;
        }
        if let Some(m) = v.get("audio_encoder").and_then(|o| o["n_mels"].as_u64()) {
            cfg.n_mels = m as usize;
        }
        if let Some(l) = v.get("text_encoder").and_then(|o| o["max_seq_len"].as_u64()) {
            cfg.max_seq_len = l as usize;
        }
        Ok(cfg)
    }
}

pub struct MultiModalEmbeddingModel {
    text_session: Mutex<Session>,
    image_session: Mutex<Session>,
    audio_session: Mutex<Session>,
    tokenizer: Arc<Tokenizer>,
    config: MultiModalConfig,
    model_path: String,
}

impl MultiModalEmbeddingModel {
    pub fn load<P: AsRef<Path>>(model_path: P, use_cpu: bool) -> UnifiedResult<Self> {
        let dir = model_path.as_ref();
        let model_path_str = dir.display().to_string();

        let config = MultiModalConfig::from_pretrained(dir)?;

        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            return Err(errors::file_not_found(&tok_path.display().to_string()));
        }
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        let text_path = dir.join("text_encoder.onnx");
        let image_path = dir.join("image_encoder.onnx");
        let audio_path = dir.join("audio_encoder.onnx");

        if !text_path.exists() {
            return Err(errors::file_not_found(&text_path.display().to_string()));
        }
        if !image_path.exists() {
            return Err(errors::file_not_found(&image_path.display().to_string()));
        }
        if !audio_path.exists() {
            return Err(errors::file_not_found(&audio_path.display().to_string()));
        }

        let text_session = Self::create_session(&text_path, use_cpu)?;
        let image_session = Self::create_session(&image_path, use_cpu)?;
        let audio_session = Self::create_session(&audio_path, use_cpu)?;

        println!("INFO: Multi-modal ONNX model loaded from {}", model_path_str);
        println!("INFO: embedding_dim={}, image_size={}, n_mels={}",
            config.embedding_dim, config.image_size, config.n_mels);

        Ok(Self {
            text_session: Mutex::new(text_session),
            image_session: Mutex::new(image_session),
            audio_session: Mutex::new(audio_session),
            tokenizer: Arc::new(tokenizer),
            config,
            model_path: model_path_str,
        })
    }

    fn create_session<P: AsRef<Path>>(onnx_path: P, use_cpu: bool) -> UnifiedResult<Session> {
        let path_str = onnx_path.as_ref().display().to_string();

        if use_cpu {
            return Session::builder()
                .map_err(|e| errors::ort_error(&e.to_string()))?
                .commit_from_file(onnx_path.as_ref())
                .map_err(|e| errors::model_load(&path_str, &e.to_string()));
        }

        #[cfg(feature = "migraphx")]
        {
            use ort::execution_providers::MIGraphXExecutionProvider;
            match Session::builder()
                .map_err(|e| errors::ort_error(&e.to_string()))
                .and_then(|b| b.with_execution_providers([MIGraphXExecutionProvider::default().with_fp16(true).build().error_on_failure()])
                    .map_err(|e| errors::ort_error(&e.to_string())))
                .and_then(|b| b.commit_from_file(onnx_path.as_ref())
                    .map_err(|e| errors::model_load(&path_str, &e.to_string())))
            {
                Ok(s) => return Ok(s),
                Err(e) => println!("WARN: MIGraphX EP failed: {}", e),
            }
        }

        #[cfg(feature = "rocm")]
        {
            use ort::execution_providers::{ROCmExecutionProvider, ArenaExtendStrategy};
            use crate::core::gpu_memory;
            let mem_limit = gpu_memory::get_gpu_mem_limit();
            match Session::builder()
                .map_err(|e| errors::ort_error(&e.to_string()))
                .and_then(|b| b.with_execution_providers([
                    ROCmExecutionProvider::default()
                        .with_mem_limit(mem_limit)
                        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                        .build()
                        .error_on_failure()
                    ])
                    .map_err(|e| errors::ort_error(&e.to_string())))
                .and_then(|b| b.commit_from_file(onnx_path.as_ref())
                    .map_err(|e| errors::model_load(&path_str, &e.to_string())))
            {
                Ok(s) => return Ok(s),
                Err(e) => println!("WARN: ROCm EP failed: {}", e),
            }
        }

        #[cfg(feature = "cuda")]
        {
            use ort::execution_providers::CUDAExecutionProvider;
            match Session::builder()
                .map_err(|e| errors::ort_error(&e.to_string()))
                .and_then(|b| b.with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
                    .map_err(|e| errors::ort_error(&e.to_string())))
                .and_then(|b| b.commit_from_file(onnx_path.as_ref())
                    .map_err(|e| errors::model_load(&path_str, &e.to_string())))
            {
                Ok(s) => return Ok(s),
                Err(e) => println!("WARN: CUDA EP failed: {}", e),
            }
        }

        #[allow(unreachable_code)]
        Session::builder()
            .map_err(|e| errors::ort_error(&e.to_string()))?
            .commit_from_file(onnx_path.as_ref())
            .map_err(|e| errors::model_load(&path_str, &e.to_string()))
    }

    pub fn config(&self) -> &MultiModalConfig { &self.config }
    pub fn model_path(&self) -> &str { &self.model_path }

    /// Encode text → L2-normalised embedding.
    pub fn encode_text(&self, text: &str, target_dim: Option<usize>) -> UnifiedResult<Array1<f32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        let max_len = encoding.len().min(self.config.max_seq_len);
        let ids: Vec<i64> = encoding.get_ids()[..max_len].iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask()[..max_len].iter().map(|&m| m as i64).collect();
        let seq_len = ids.len();

        let ids_tensor = Tensor::from_array(([1usize, seq_len], ids))
            .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?;
        let mask_tensor = Tensor::from_array(([1usize, seq_len], mask))
            .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?;

        let mut session = self.text_session.lock();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?;

        let emb = self.extract_embedding_from_outputs(&outputs)?;
        self.maybe_truncate(emb, target_dim)
    }

    /// Encode pre-processed image pixels → L2-normalised embedding.
    /// `pixel_data` is [3 * H * W] in row-major, float32, pixel values in [0, 1].
    pub fn encode_image(&self, pixel_data: &[f32], height: usize, width: usize, target_dim: Option<usize>) -> UnifiedResult<Array1<f32>> {
        let expected = 3 * height * width;
        if pixel_data.len() != expected {
            return Err(errors::validation("pixel_data length", &expected.to_string(), &pixel_data.len().to_string()));
        }

        let pixel_vec: Vec<f32> = pixel_data.to_vec();
        let tensor = Tensor::from_array(([1usize, 3, height, width], pixel_vec))
            .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?;

        let mut session = self.image_session.lock();
        let outputs = session
            .run(ort::inputs![
                "pixel_values" => tensor,
            ])
            .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?;

        let emb = self.extract_embedding_from_outputs(&outputs)?;
        self.maybe_truncate(emb, target_dim)
    }

    /// Encode mel spectrogram → L2-normalised embedding.
    /// `mel_data` is [n_mels * time_frames] in row-major.
    /// The Whisper ONNX model expects exactly 3000 time frames;
    /// shorter inputs are zero-padded, longer ones are truncated.
    pub fn encode_audio(&self, mel_data: &[f32], n_mels: usize, time_frames: usize, target_dim: Option<usize>) -> UnifiedResult<Array1<f32>> {
        let expected = n_mels * time_frames;
        if mel_data.len() != expected {
            return Err(errors::validation("mel_data length", &expected.to_string(), &mel_data.len().to_string()));
        }

        const WHISPER_FRAMES: usize = 3000;
        let padded: Vec<f32> = if time_frames == WHISPER_FRAMES {
            mel_data.to_vec()
        } else {
            let mut buf = vec![0.0f32; n_mels * WHISPER_FRAMES];
            let copy_frames = time_frames.min(WHISPER_FRAMES);
            for m in 0..n_mels {
                let src_start = m * time_frames;
                let dst_start = m * WHISPER_FRAMES;
                buf[dst_start..dst_start + copy_frames]
                    .copy_from_slice(&mel_data[src_start..src_start + copy_frames]);
            }
            buf
        };

        let tensor = Tensor::from_array(([1usize, n_mels, WHISPER_FRAMES], padded))
            .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?;

        let mut session = self.audio_session.lock();
        let outputs = session
            .run(ort::inputs![
                "mel_spectrogram" => tensor,
            ])
            .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?;

        let emb = self.extract_embedding_from_outputs(&outputs)?;
        self.maybe_truncate(emb, target_dim)
    }

    /// Extract a 1-D embedding from session outputs, matching the mmbert pattern.
    fn extract_embedding_from_outputs(&self, outputs: &ort::session::SessionOutputs) -> UnifiedResult<Array1<f32>> {
        let names = ["embedding", "sentence_embedding", "pooler_output", "last_hidden_state"];
        for name in &names {
            if let Some(output_value) = outputs.get(*name) {
                if let Ok((shape, data)) = output_value.try_extract_tensor::<f32>() {
                    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                    if dims.len() == 2 && dims[0] >= 1 {
                        let dim = dims[1];
                        let vec: Vec<f32> = data.iter().take(dim).copied().collect();
                        return Ok(Array1::from_vec(vec));
                    } else if dims.len() == 1 {
                        return Ok(Array1::from_vec(data.to_vec()));
                    }
                }
            }
        }
        // Fallback: first output
        if let Some((_, output_value)) = outputs.iter().next() {
            if let Ok((shape, data)) = output_value.try_extract_tensor::<f32>() {
                let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                let dim = *dims.last().unwrap_or(&0);
                if dim > 0 {
                    let vec: Vec<f32> = data.iter().take(dim).copied().collect();
                    return Ok(Array1::from_vec(vec));
                }
            }
        }
        Err(errors::ort_error("no valid embedding output found"))
    }

    fn maybe_truncate(&self, emb: Array1<f32>, target_dim: Option<usize>) -> UnifiedResult<Array1<f32>> {
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_safe = if norm > 1e-12 { norm } else { 1e-12 };
        let normalized = emb.mapv(|x| x / norm_safe);

        if let Some(dim) = target_dim {
            if dim > 0 && dim < normalized.len() {
                return Ok(normalized.slice(ndarray::s![..dim]).to_owned());
            }
        }
        Ok(normalized)
    }
}
