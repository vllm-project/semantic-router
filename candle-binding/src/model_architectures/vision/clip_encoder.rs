//! CLIP Vision Transformer Encoder
//!
//! Implements CLIP's vision transformer (ViT) architecture for image feature extraction.
//! Supports loading CLIP ViT-B/32 weights from HuggingFace.

use crate::core::{UnifiedError, UnifiedResult};
use crate::core::unified_error::ModelErrorType;
use anyhow::Result;
use candle_core::{Device, DType, IndexOp, Module, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::Deserialize;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// CLIP Vision Transformer configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ClipVisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub projection_dim: usize,
}

impl ClipVisionConfig {
    /// Load configuration from HuggingFace or local path
    pub fn from_pretrained(model_id: &str) -> UnifiedResult<Self> {
        let (config_path, _) = Self::resolve_model_files(model_id)?;
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| UnifiedError::IO {
                operation: "read config".to_string(),
                path: Some(config_path.clone()),
                source: e,
            })?;
        
        #[derive(Deserialize)]
        struct ClipConfig {
            vision_config: ClipVisionConfig,
        }
        
        let clip_config: ClipConfig = serde_json::from_str(&config_json)
            .map_err(|e| UnifiedError::Validation {
                field: "config.json".to_string(),
                expected: "valid CLIP config".to_string(),
                actual: e.to_string(),
                context: None,
            })?;
        
        Ok(clip_config.vision_config)
    }
    
    /// Resolve model files (local or HuggingFace)
    fn resolve_model_files(model_id: &str) -> UnifiedResult<(String, String)> {
        if Path::new(model_id).exists() {
            let config_path = Path::new(model_id).join("config.json").to_string_lossy().to_string();
            let weights_path = if Path::new(model_id).join("pytorch_model.bin").exists() {
                Path::new(model_id).join("pytorch_model.bin").to_string_lossy().to_string()
            } else if Path::new(model_id).join("model.safetensors").exists() {
                Path::new(model_id).join("model.safetensors").to_string_lossy().to_string()
            } else {
                Path::new(model_id).join("vision_model.safetensors").to_string_lossy().to_string()
            };
            Ok((config_path, weights_path))
        } else {
            let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());
            let api = Api::new().map_err(|e| UnifiedError::IO {
                operation: "create HuggingFace API".to_string(),
                path: None,
                source: std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
            })?;
            let api = api.repo(repo);

            let config = api.get("config.json").map_err(|e| {
                eprintln!("ERROR: Failed to download config.json: {}", e);
                UnifiedError::IO {
                    operation: "download config".to_string(),
                    path: Some("config.json".to_string()),
                    source: std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
                }
            })?;
            
            eprintln!("Successfully downloaded config.json");
            
            // CLIP model doesn't have model.safetensors in main branch
            // API shows only pytorch_model.bin, flax_model.msgpack, tf_model.h5
            // Use pytorch_model.bin (same as BERT fallback pattern)
            eprintln!("CLIP model uses pytorch_model.bin (safetensors not available in this model)");
            let weights_file = api.get("pytorch_model.bin").map_err(|e| {
                eprintln!("ERROR: Failed to download pytorch_model.bin: {}", e);
                UnifiedError::IO {
                    operation: "download weights".to_string(),
                    path: Some("pytorch_model.bin".to_string()),
                    source: std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e)),
                }
            })?;
            
            eprintln!("Successfully downloaded pytorch_model.bin");
            
            Ok((
                config.to_string_lossy().to_string(),
                weights_file.to_string_lossy().to_string(),
            ))
        }
    }
}

/// Vision Transformer Encoder trait
pub trait VisionEncoder {
    fn encode(&self, image: &Tensor) -> Result<Tensor>;
    fn embedding_dim(&self) -> usize;
}

/// CLIP Vision Transformer Encoder
pub struct ClipVisionEncoder {
    config: ClipVisionConfig,
    device: Device,
    patch_embedding: Linear,
    class_embedding: Tensor,
    position_embedding: Tensor,
    encoder_layers: Vec<VisionTransformerLayer>,
    final_layer_norm: LayerNorm, // CLIP uses "final_layer_norm", not "post_layernorm"
    projection: Linear,
}

impl ClipVisionEncoder {
    /// Create new CLIP vision encoder
    pub fn new(_config: ClipVisionConfig, _device: Device) -> Result<Self> {
        // This will be populated when loading weights
        // For now, create placeholder
        Err(anyhow::anyhow!("Use from_pretrained to load model"))
    }
    
    /// Load CLIP vision encoder from HuggingFace or local path
    pub fn from_pretrained(model_id: &str, device: Device) -> UnifiedResult<Self> {
        let config = ClipVisionConfig::from_pretrained(model_id)?;
        let (_, weights_path) = ClipVisionConfig::resolve_model_files(model_id)?;
        
        eprintln!("Loading weights from: {}", weights_path);
        
        // CLIP pytorch_model.bin contains both vision and text encoders
        // Vision encoder weights are prefixed with "vision_model."
        let use_prefix = weights_path.contains("pytorch_model.bin");
        eprintln!("Using weight prefix: {}", if use_prefix { "vision_model." } else { "none" });
        
        // Load weights - CLIP uses PyTorch format (not safetensors)
        eprintln!("Attempting to load PyTorch model from: {}", weights_path);
        let vb_result = unsafe {
            VarBuilder::from_pth(&weights_path, DType::F32, &device)
        };
        
        let mut vb = match vb_result {
            Ok(v) => {
                eprintln!("Successfully loaded PyTorch model file");
                v
            }
            Err(e) => {
                eprintln!("ERROR: Failed to load PyTorch model: {}", e);
                eprintln!("ERROR: File path: {}", weights_path);
                eprintln!("ERROR: This might indicate the file doesn't exist or is corrupted");
                return Err(UnifiedError::Model {
                    model_type: ModelErrorType::Traditional,
                    operation: "load CLIP weights".to_string(),
                    source: format!("Failed to load PyTorch model from {}: {}", weights_path, e),
                    context: None,
                });
            }
        };
        
        // visual_projection is at root level, not under vision_model
        // Load it before applying the prefix
        // Note: visual_projection might not have a bias
        eprintln!("Loading visual_projection (at root level)...");
        let projection = {
            // Try with bias first
            match linear(
                config.hidden_size,
                config.projection_dim,
                vb.pp("visual_projection"),
            ) {
                Ok(proj) => {
                    eprintln!("Successfully loaded visual_projection with bias");
                    proj
                }
                Err(_) => {
                    // Try without bias
                    eprintln!("visual_projection with bias failed, trying without bias...");
                    let weight = vb
                        .get(
                            (config.projection_dim, config.hidden_size),
                            "visual_projection.weight",
                        )
                        .map_err(|e| {
                            eprintln!("ERROR: Failed to load visual_projection.weight: {}", e);
                            UnifiedError::Model {
                                model_type: ModelErrorType::Traditional,
                                operation: "create projection".to_string(),
                                source: format!("Failed to load visual_projection.weight: {}", e),
                                context: None,
                            }
                        })?;
                    eprintln!("Successfully loaded visual_projection.weight, creating Linear without bias");
                    Linear::new(weight, None) // No bias
                }
            }
        };
        eprintln!("Successfully loaded visual_projection at root level");
        
        // Apply prefix if needed (PyTorch CLIP models use "vision_model." prefix)
        // But visual_projection is already loaded, so we skip it
        if use_prefix {
            eprintln!("Applying vision_model. prefix to weight names (except visual_projection)");
            vb = vb.pp("vision_model");
        }
        
        Self::load_with_weights(config, vb, device, projection)
    }
    
    /// Load model with weights
    fn load_with_weights(
        config: ClipVisionConfig,
        vb: VarBuilder,
        device: Device,
        projection: Linear, // visual_projection loaded separately (root level)
    ) -> UnifiedResult<Self> {
        // Patch embedding: projects patches to hidden dimension
        eprintln!("Loading patch embedding...");
        let patch_embedding = {
            // Try to load as Linear first (safetensors format)
            match linear(
                config.patch_size * config.patch_size * 3, // 3 channels (RGB) = 3072
                config.hidden_size, // 768
                vb.pp("embeddings.patch_embedding"),
            ) {
                Ok(l) => {
                    eprintln!("Patch embedding loaded as Linear (safetensors format)");
                    l
                }
                Err(_) => {
                    // If that fails, load as Conv2d and reshape
                    eprintln!("Patch embedding not in Linear format, loading as Conv2d [768, 3, 32, 32] and reshaping to [768, 3072]...");
                    let conv_weight = vb
                        .get(
                            (config.hidden_size, 3, config.patch_size, config.patch_size),
                            "embeddings.patch_embedding.weight",
                        )
                        .map_err(|e| UnifiedError::Model {
                            model_type: ModelErrorType::Traditional,
                            operation: "load patch embedding conv weight".to_string(),
                            source: e.to_string(),
                            context: None,
                        })?;
                    
                    eprintln!("Loaded conv weight shape: {:?}", conv_weight.shape());
                    
                    // So we just need to flatten the last 3 dimensions: [768, 3*32*32] = [768, 3072]
                    let reshaped = conv_weight
                        .reshape((config.hidden_size, config.patch_size * config.patch_size * 3))
                        .map_err(|e| UnifiedError::Model {
                            model_type: ModelErrorType::Traditional,
                            operation: "reshape patch embedding".to_string(),
                            source: format!("Failed to reshape from {:?}: {}", conv_weight.shape(), e),
                            context: None,
                        })?;
                    
                    eprintln!("Reshaped conv weight to: {:?} (should be [768, 3072])", reshaped.shape());
                    
                    // Load bias if it exists (CLIP patch embedding typically has no bias)
                    let bias = match vb.get((config.hidden_size,), "embeddings.patch_embedding.bias") {
                        Ok(b) => Some(b),
                        Err(_) => {
                            eprintln!("No bias found for patch embedding (this is normal for CLIP)");
                            None
                        }
                    };
                    
                    // Create Linear layer manually with reshaped weights
                    eprintln!("Creating Linear layer with weight shape: {:?}, bias: {:?}", reshaped.shape(), bias.as_ref().map(|b| b.shape()));
                    
                    Linear::new(reshaped, bias)
                }
            }
        };
        
        // Class embedding (CLS token)
        let class_embedding = vb
            .get((config.hidden_size,), "embeddings.class_embedding")
            .map_err(|e| UnifiedError::Model {
                model_type: ModelErrorType::Traditional,
                operation: "load class embedding".to_string(),
                source: e.to_string(),
                context: None,
            })?;
        
        // PyTorch CLIP stores as: vision_model.embeddings.position_embedding.weight
        let num_patches = (config.image_size / config.patch_size).pow(2);
        let num_positions = num_patches + 1; // +1 for CLS token
        eprintln!("Loading position embedding: expecting shape [{}, {}]", num_positions, config.hidden_size);
        let position_embedding = match vb.get(
            (num_positions, config.hidden_size),
            "embeddings.position_embedding.weight",
        ) {
            Ok(emb) => {
                eprintln!("Successfully loaded position embedding with .weight suffix");
                emb
            }
            Err(e) => {
                eprintln!("WARNING: Failed to load position embedding with .weight suffix: {}", e);
                eprintln!("Trying without .weight suffix...");
                vb.get(
                    (num_positions, config.hidden_size),
                    "embeddings.position_embedding",
                )
                .map_err(|e2| {
                    eprintln!("ERROR: Both attempts failed. With .weight: {}. Without: {}", e, e2);
                    UnifiedError::Model {
                        model_type: ModelErrorType::Traditional,
                        operation: "load position embedding".to_string(),
                        source: format!("Failed with .weight: {}. Failed without: {}", e, e2),
                        context: None,
                    }
                })?
            }
        };
        eprintln!("Successfully loaded position embedding with shape: {:?}", position_embedding.shape());
        
        // Encoder layers
        eprintln!("Loading {} encoder layers...", config.num_hidden_layers);
        let mut encoder_layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            eprintln!("Loading encoder layer {}...", i);
            encoder_layers.push(
                VisionTransformerLayer::new(&config, vb.pp(&format!("encoder.layers.{}", i)))
                    .map_err(|e| UnifiedError::Model {
                        model_type: ModelErrorType::Traditional,
                        operation: format!("create encoder layer {}", i),
                        source: e.to_string(),
                        context: None,
                    })?,
            );
        }
        
        eprintln!("Loading final layer norm...");
        let final_layer_norm = match layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("final_layer_norm"),
        ) {
            Ok(ln) => {
                eprintln!("Found final_layer_norm");
                ln
            }
            Err(_) => {
                eprintln!("final_layer_norm not found, trying post_layernorm...");
                match layer_norm(
                    config.hidden_size,
                    config.layer_norm_eps,
                    vb.pp("post_layernorm"),
                ) {
                    Ok(ln) => {
                        eprintln!("Found post_layernorm");
                        ln
                    }
                    Err(e) => {
                        eprintln!("ERROR: Neither final_layer_norm nor post_layernorm found");
                        return Err(UnifiedError::Model {
                            model_type: ModelErrorType::Traditional,
                            operation: "create final layer norm".to_string(),
                            source: format!("Failed to find final layer norm. Tried: final_layer_norm, post_layernorm. Error: {}", e),
                            context: None,
                        });
                    }
                }
            }
        };
        
        eprintln!("Using pre-loaded visual_projection");
        
        Ok(Self {
            config,
            device,
            patch_embedding,
            class_embedding,
            position_embedding,
            encoder_layers,
            final_layer_norm,
            projection,
        })
    }
    
    /// Encode image patches
    fn encode_patches(&self, image: &Tensor) -> Result<Tensor> {
        // image: [batch, 3, 224, 224]
        let (batch, channels, height, width) = image.dims4()?;
        let patch_size = self.config.patch_size;
        let num_patches_h = height / patch_size;
        let num_patches_w = width / patch_size;
        let num_patches = num_patches_h * num_patches_w;
        
        // Reshape image to extract patches
        // [batch, channels, height, width] -> [batch, channels, num_patches_h, patch_size, num_patches_w, patch_size]
        // -> [batch, num_patches, channels * patch_size * patch_size]
        let image_reshaped = image
            .reshape((batch, channels, num_patches_h, patch_size, num_patches_w, patch_size))?
            .permute((0, 2, 4, 1, 3, 5))? // [batch, num_patches_h, num_patches_w, channels, patch_size, patch_size]
            .reshape((batch, num_patches, channels * patch_size * patch_size))?;
        
        // Project patches to hidden dimension
        self.patch_embedding.forward(&image_reshaped)
            .map_err(|e| anyhow::anyhow!("Patch embedding forward failed: {}", e))
    }
}

impl VisionEncoder for ClipVisionEncoder {
    fn encode(&self, image: &Tensor) -> Result<Tensor> {
        // image: [batch, 3, 224, 224]
        
        // 1.Encode patches
        let patch_embeddings = self.encode_patches(image)?; // [batch, num_patches, hidden_size]
        
        // 2.Add CLS token
        let batch_size = patch_embeddings.dim(0)?;
        let class_emb = self.class_embedding.unsqueeze(0)?.expand((batch_size, 1, self.config.hidden_size))?;
        let embeddings = Tensor::cat(&[&class_emb, &patch_embeddings], 1)?; // [batch, num_patches+1, hidden_size]
        
        // 3. Add position embeddings
        let pos_emb = self.position_embedding.unsqueeze(0)?; // [1, num_positions, hidden_size]
        let embeddings = embeddings.broadcast_add(&pos_emb)?;
        
        // 4. Encoder layers (no pre-layernorm in CLIP - layer norms are inside each layer)
        let mut hidden_states = embeddings;
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        
        // 5. Final layer norm
        hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        
        // 7. Extract CLS token (first token)
        let cls_token = hidden_states.i((.., 0, ..))?; // [batch, hidden_size]
        
        // 8. Project to projection dimension
        let projected = self.projection.forward(&cls_token)?; // [batch, projection_dim]
        
        // 9. Normalize (L2 normalization)
        let squared = projected.sqr()?;
        let sum_squared = squared.sum_keepdim(1)?;
        let norm = sum_squared.sqrt()?;
        let normalized = projected.broadcast_div(&norm)?;
        
        Ok(normalized)
    }
    
    fn embedding_dim(&self) -> usize {
        self.config.projection_dim
    }
}

/// Vision Transformer Encoder Layer
struct VisionTransformerLayer {
    self_attn: VisionSelfAttention,
    layer_norm1: LayerNorm,
    mlp: VisionMLP,
    layer_norm2: LayerNorm,
}

impl VisionTransformerLayer {
    fn new(config: &ClipVisionConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = VisionSelfAttention::new(config, vb.pp("self_attn"))?;
        let layer_norm1 = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm1"),
        )?;
        let mlp = VisionMLP::new(config, vb.pp("mlp"))?;
        let layer_norm2 = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm2"),
        )?;
        
        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }
    
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Self-attention with residual
        let residual = hidden_states;
        let hidden_states = self.layer_norm1.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        
        // MLP with residual
        let residual = &hidden_states;
        let hidden_states = self.layer_norm2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        
        Ok(hidden_states)
    }
}

/// Vision Self-Attention
/// CLIP uses separate q_proj, k_proj, v_proj (not combined qkv)
struct VisionSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionSelfAttention {
    fn new(config: &ClipVisionConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        // CLIP uses separate projections: q_proj, k_proj, v_proj
        let q_proj = linear(config.hidden_size, config.hidden_size, vb.pp("q_proj"))
            .map_err(|e| anyhow::anyhow!("Failed to create q_proj: {}", e))?;
        let k_proj = linear(config.hidden_size, config.hidden_size, vb.pp("k_proj"))
            .map_err(|e| anyhow::anyhow!("Failed to create k_proj: {}", e))?;
        let v_proj = linear(config.hidden_size, config.hidden_size, vb.pp("v_proj"))
            .map_err(|e| anyhow::anyhow!("Failed to create v_proj: {}", e))?;
        let out_proj = linear(config.hidden_size, config.hidden_size, vb.pp("out_proj"))
            .map_err(|e| anyhow::anyhow!("Failed to create out_proj: {}", e))?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }
    
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = hidden_states.dims3()?;
        
        // Compute Q, K, V separately
        let q = self.q_proj.forward(hidden_states)?; // [batch, seq_len, hidden_size]
        let k = self.k_proj.forward(hidden_states)?; // [batch, seq_len, hidden_size]
        let v = self.v_proj.forward(hidden_states)?; // [batch, seq_len, hidden_size]
        
        // Reshape for multi-head attention
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        
        // Permute for attention computation: [batch, num_heads, seq_len, head_dim]
        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;
        let v = v.permute((0, 2, 1, 3))?;
        
        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.t()?)? * scale)?; // [batch, num_heads, seq_len, seq_len]
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?; // [batch, num_heads, seq_len, head_dim]
        
        // Reshape and project
        let attn_output = attn_output.permute((0, 2, 1, 3))?; // [batch, seq_len, num_heads, head_dim]
        let attn_output = attn_output.reshape((batch, seq_len, hidden_size))?;
        self.out_proj.forward(&attn_output)
            .map_err(|e| anyhow::anyhow!("Attention output projection failed: {}", e))
    }
}

/// Vision MLP
struct VisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMLP {
    fn new(config: &ClipVisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(config.hidden_size, config.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(config.intermediate_size, config.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
    
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.fc1.forward(hidden_states)
            .map_err(|e| anyhow::anyhow!("MLP fc1 forward failed: {}", e))?;
        let hidden_states = hidden_states.gelu()
            .map_err(|e| anyhow::anyhow!("GELU activation failed: {}", e))?;
        self.fc2.forward(&hidden_states)
            .map_err(|e| anyhow::anyhow!("MLP fc2 forward failed: {}", e))
    }
}

// Global cache
static mut VISION_ENCODER: Option<Arc<Mutex<ClipVisionEncoder>>> = None;

/// Initialize vision encoder once
pub fn init_vision_encoder(model_id: &str, device: Device) -> UnifiedResult<()> {
    eprintln!("init_vision_encoder: Starting initialization for model_id={}, device={:?}", model_id, device);
    unsafe {
        eprintln!("init_vision_encoder: Calling from_pretrained...");
        let encoder = match ClipVisionEncoder::from_pretrained(model_id, device) {
            Ok(e) => {
                eprintln!("init_vision_encoder: from_pretrained succeeded");
                e
            }
            Err(e) => {
                eprintln!("init_vision_encoder: from_pretrained failed: {}", e);
                eprintln!("init_vision_encoder: Error details: {:?}", e);
                return Err(e);
            }
        };
        eprintln!("init_vision_encoder: Creating Arc<Mutex<>> wrapper...");
        VISION_ENCODER = Some(Arc::new(Mutex::new(encoder)));
        eprintln!("init_vision_encoder: Initialization complete");
        Ok(())
    }
}

/// Get global vision encoder
pub fn get_vision_encoder() -> Option<Arc<Mutex<ClipVisionEncoder>>> {
    unsafe { 
        VISION_ENCODER.as_ref().map(|e| Arc::clone(e))
    }
}


