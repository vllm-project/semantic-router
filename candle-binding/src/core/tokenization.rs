//! Tokenization Core Module

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use tokenizers::{
    Encoding, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

/// Tokenization mode for different processing requirements
#[derive(Debug, Clone, PartialEq)]
pub enum TokenizationMode {
    /// Single text encoding (BERT-style)
    Single,
    /// Batch processing with padding
    Batch,
    /// ModernBERT-specific batch processing
    ModernBertBatch,
    /// LoRA-optimized tokenization
    LoRA,
}

/// Tokenization strategy enumeration
///
/// Renamed from ModelType to avoid confusion with the main ModelType enum.
/// This enum determines the tokenization strategy (padding, token type, etc.)
/// independent of the actual model architecture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenizationStrategy {
    /// Traditional BERT models (I32 tokens, standard padding)
    BERT,
    /// ModernBERT models (U32 tokens, optimized padding)
    ModernBERT,
    /// mmBERT models (multilingual ModernBERT, U32 tokens, 256k vocab, 8192 max length)
    MmBERT,
    /// LoRA-enabled models (I32 tokens, LoRA-specific handling)
    LoRA,
    /// Long-context embedding models (varies by model)
    LongContextEmbedding,
}

/// Data type for token IDs
#[derive(Debug, Clone, PartialEq)]
pub enum TokenDataType {
    /// 32-bit unsigned integers (ModernBERT)
    U32,
    /// 32-bit signed integers (BERT)
    I32,
}

/// Tokenization configuration
#[derive(Debug, Clone)]
pub struct TokenizationConfig {
    /// Maximum sequence length
    pub max_length: usize,
    /// Whether to add special tokens
    pub add_special_tokens: bool,
    /// Truncation strategy
    pub truncation_strategy: TruncationStrategy,
    /// Truncation direction
    pub truncation_direction: TruncationDirection,
    /// Padding token ID
    pub pad_token_id: u32,
    /// Padding token string
    pub pad_token: String,
    /// Tokenization strategy for this model
    pub tokenization_strategy: TokenizationStrategy,
    /// Expected token data type
    pub token_data_type: TokenDataType,
}

impl Default for TokenizationConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            add_special_tokens: true,
            truncation_strategy: TruncationStrategy::LongestFirst,
            truncation_direction: TruncationDirection::Right,
            pad_token_id: 0,
            pad_token: "[PAD]".to_string(),
            tokenization_strategy: TokenizationStrategy::BERT,
            token_data_type: TokenDataType::I32,
        }
    }
}

/// Tokenization result for single text
#[derive(Debug, Clone)]
pub struct TokenizationResult {
    /// Token IDs as i32 (for compatibility)
    pub token_ids: Vec<i32>,
    /// Token IDs as u32 (for ModernBERT)
    pub token_ids_u32: Vec<u32>,
    /// Attention mask
    pub attention_mask: Vec<u32>,
    /// Token strings
    pub tokens: Vec<String>,
    /// Character offsets for token mapping
    pub offsets: Vec<(usize, usize)>,
}

/// Batch tokenization result
#[derive(Debug, Clone)]
pub struct BatchTokenizationResult {
    /// Batch of token IDs (padded)
    pub token_ids: Vec<Vec<i32>>,
    /// Batch of token IDs as u32 (for ModernBERT)
    pub token_ids_u32: Vec<Vec<u32>>,
    /// Batch of attention masks
    pub attention_masks: Vec<Vec<u32>>,
    /// Batch of token strings
    pub tokens: Vec<Vec<String>>,
    /// Maximum sequence length in batch
    pub max_length: usize,
    /// Batch size
    pub batch_size: usize,
}

/// Unified tokenizer trait for dual-path architecture
pub trait DualPathTokenizer: Send + Sync + std::fmt::Debug {
    /// Tokenize single text with automatic strategy selection
    fn tokenize(&self, text: &str) -> Result<TokenizationResult>;

    /// Tokenize batch of texts efficiently
    fn tokenize_batch(&self, texts: &[&str]) -> Result<BatchTokenizationResult>;

    /// Tokenize for traditional model path
    fn tokenize_for_traditional(&self, text: &str) -> Result<TokenizationResult>;

    /// Tokenize for LoRA model path
    fn tokenize_for_lora(&self, text: &str) -> Result<TokenizationResult>;

    /// Smart batch tokenization with automatic padding optimization
    fn tokenize_batch_smart(
        &self,
        texts: &[&str],
        prefer_lora: bool,
    ) -> Result<BatchTokenizationResult>;

    /// Get tokenizer configuration
    fn get_config(&self) -> &TokenizationConfig;

    /// Check if tokenizer supports parallel processing
    fn supports_parallel(&self) -> bool;

    /// Create tensors from tokenization result
    fn create_tensors(&self, result: &TokenizationResult) -> Result<(Tensor, Tensor)>;

    /// Create batch tensors from batch tokenization result
    fn create_batch_tensors(&self, result: &BatchTokenizationResult) -> Result<(Tensor, Tensor)>;
}

/// Unified tokenizer implementation
#[derive(Debug)]
pub struct UnifiedTokenizer {
    /// Core tokenizer
    tokenizer: Tokenizer,
    /// Tokenization configuration
    config: TokenizationConfig,
    /// Device for tensor operations
    device: Device,
}

impl UnifiedTokenizer {
    /// Create a new unified tokenizer
    ///
    /// ## Arguments
    /// * `tokenizer` - Pre-configured tokenizer instance
    /// * `config` - Tokenization configuration
    /// * `device` - Computing device
    ///
    /// ## Returns
    /// * `Result<Self>` - Initialized unified tokenizer
    pub fn new(tokenizer: Tokenizer, config: TokenizationConfig, device: Device) -> Result<Self> {
        Ok(Self {
            tokenizer,
            config,
            device,
        })
    }

    /// Create from tokenizer path with automatic configuration
    ///
    /// ## Arguments
    /// * `tokenizer_path` - Path to tokenizer.json file
    /// * `tokenization_strategy` - Tokenization strategy for this model
    /// * `device` - Computing device
    ///
    /// ## Returns
    /// * `Result<Self>` - Initialized unified tokenizer
    pub fn from_file(
        tokenizer_path: &str,
        tokenization_strategy: TokenizationStrategy,
        device: Device,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        let config = TokenizationConfig {
            tokenization_strategy,
            token_data_type: match tokenization_strategy {
                TokenizationStrategy::ModernBERT | TokenizationStrategy::MmBERT => {
                    TokenDataType::U32
                }
                _ => TokenDataType::I32,
            },
            // mmBERT supports 8192 max length, ModernBERT uses 512
            max_length: match tokenization_strategy {
                TokenizationStrategy::MmBERT => 8192,
                _ => 512,
            },
            ..Default::default()
        };

        Self::new(tokenizer, config, device)
    }

    /// Configure tokenizer for specific mode
    fn configure_for_mode(&self, mode: TokenizationMode) -> Result<Tokenizer> {
        let mut tokenizer = self.tokenizer.clone();

        // Set truncation
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.config.max_length,
                strategy: self.config.truncation_strategy.clone(),
                stride: 0,
                direction: self.config.truncation_direction.clone(),
            }))
            .map_err(E::msg)?;

        // Set padding for batch modes
        if matches!(
            mode,
            TokenizationMode::Batch | TokenizationMode::ModernBertBatch
        ) {
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                direction: PaddingDirection::Right,
                pad_to_multiple_of: None,
                pad_id: self.config.pad_token_id,
                pad_type_id: 0,
                pad_token: self.config.pad_token.clone(),
            }));
        }

        Ok(tokenizer)
    }

    /// Convert encoding to tokenization result
    fn encoding_to_result(&self, encoding: &Encoding) -> TokenizationResult {
        let token_ids_u32 = encoding.get_ids().to_vec();
        let token_ids: Vec<i32> = token_ids_u32.iter().map(|&id| id as i32).collect();
        let attention_mask = encoding.get_attention_mask().to_vec();
        let tokens = encoding.get_tokens().to_vec();
        let offsets = encoding.get_offsets().to_vec();

        TokenizationResult {
            token_ids,
            token_ids_u32,
            attention_mask,
            tokens,
            offsets,
        }
    }

    /// Convert batch encodings to batch result
    fn encodings_to_batch_result(&self, encodings: &[Encoding]) -> BatchTokenizationResult {
        let mut token_ids = Vec::new();
        let mut token_ids_u32 = Vec::new();
        let mut attention_masks = Vec::new();
        let mut tokens = Vec::new();
        let mut max_length = 0;

        for encoding in encodings {
            let ids_u32 = encoding.get_ids().to_vec();
            let ids_i32: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
            let mask = encoding.get_attention_mask().to_vec();
            let toks = encoding.get_tokens().to_vec();

            max_length = max_length.max(ids_u32.len());

            token_ids.push(ids_i32);
            token_ids_u32.push(ids_u32);
            attention_masks.push(mask);
            tokens.push(toks);
        }

        BatchTokenizationResult {
            token_ids,
            token_ids_u32,
            attention_masks,
            tokens,
            max_length,
            batch_size: encodings.len(),
        }
    }

    /// Create tensors from tokenization result
    pub fn create_tensors(&self, result: &TokenizationResult) -> Result<(Tensor, Tensor)> {
        // Always use u32 for Tensor::new as it's the expected type
        let token_ids_tensor =
            Tensor::new(&result.token_ids_u32[..], &self.device)?.unsqueeze(0)?;
        let attention_mask_tensor =
            Tensor::new(&result.attention_mask[..], &self.device)?.unsqueeze(0)?;

        Ok((token_ids_tensor, attention_mask_tensor))
    }

    /// Create batch tensors from batch tokenization result
    pub fn create_batch_tensors(
        &self,
        result: &BatchTokenizationResult,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = result.batch_size;
        let max_length = result.max_length;

        // Always use u32 for Tensor::new - this is the required type
        let mut padded_token_ids = Vec::new();
        let mut padded_attention_masks = Vec::new();

        for i in 0..batch_size {
            let mut ids = result.token_ids_u32[i].clone();
            let mut mask = result.attention_masks[i].clone();

            // Pad to max_length
            ids.resize(max_length, self.config.pad_token_id);
            mask.resize(max_length, 0);

            padded_token_ids.extend(ids);
            padded_attention_masks.extend(mask);
        }

        let token_ids_tensor = Tensor::new(padded_token_ids.as_slice(), &self.device)?
            .reshape(&[batch_size, max_length])?;
        let attention_mask_tensor = Tensor::new(padded_attention_masks.as_slice(), &self.device)?
            .reshape(&[batch_size, max_length])?;

        Ok((token_ids_tensor, attention_mask_tensor))
    }
}

impl DualPathTokenizer for UnifiedTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenizationResult> {
        let mode = match self.config.tokenization_strategy {
            TokenizationStrategy::ModernBERT | TokenizationStrategy::MmBERT => {
                TokenizationMode::ModernBertBatch
            }
            TokenizationStrategy::LoRA => TokenizationMode::LoRA,
            _ => TokenizationMode::Single,
        };

        match mode {
            TokenizationMode::ModernBertBatch => {
                // ModernBERT and mmBERT use batch processing even for single text
                let tokenizer = self.configure_for_mode(mode)?;
                let encodings = tokenizer
                    .encode_batch(vec![text], self.config.add_special_tokens)
                    .map_err(E::msg)?;
                Ok(self.encoding_to_result(&encodings[0]))
            }
            _ => {
                // Standard single text encoding
                let tokenizer = self.configure_for_mode(TokenizationMode::Single)?;
                let encoding = tokenizer
                    .encode(text, self.config.add_special_tokens)
                    .map_err(E::msg)?;
                Ok(self.encoding_to_result(&encoding))
            }
        }
    }

    fn tokenize_batch(&self, texts: &[&str]) -> Result<BatchTokenizationResult> {
        let mode = match self.config.tokenization_strategy {
            TokenizationStrategy::ModernBERT | TokenizationStrategy::MmBERT => {
                TokenizationMode::ModernBertBatch
            }
            _ => TokenizationMode::Batch,
        };

        let tokenizer = self.configure_for_mode(mode)?;
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), self.config.add_special_tokens)
            .map_err(E::msg)?;

        Ok(self.encodings_to_batch_result(&encodings))
    }

    fn tokenize_for_traditional(&self, text: &str) -> Result<TokenizationResult> {
        // Force traditional BERT-style tokenization
        let tokenizer = self.configure_for_mode(TokenizationMode::Single)?;
        let encoding = tokenizer
            .encode(text, self.config.add_special_tokens)
            .map_err(E::msg)?;
        Ok(self.encoding_to_result(&encoding))
    }

    fn tokenize_for_lora(&self, text: &str) -> Result<TokenizationResult> {
        // LoRA-optimized tokenization (currently same as traditional, but extensible)
        let tokenizer = self.configure_for_mode(TokenizationMode::LoRA)?;
        let encoding = tokenizer
            .encode(text, self.config.add_special_tokens)
            .map_err(E::msg)?;

        // Explicitly enforce max_length truncation for LoRA models
        // This is a safety check to ensure we never exceed the model's position embedding size
        let mut result = self.encoding_to_result(&encoding);
        let max_len = self.config.max_length;
        if result.token_ids.len() > max_len {
            result.token_ids.truncate(max_len);
            result.token_ids_u32.truncate(max_len);
            result.attention_mask.truncate(max_len);
            result.tokens.truncate(max_len);
        }

        Ok(result)
    }

    fn tokenize_batch_smart(
        &self,
        texts: &[&str],
        prefer_lora: bool,
    ) -> Result<BatchTokenizationResult> {
        if prefer_lora && self.config.tokenization_strategy == TokenizationStrategy::LoRA {
            // Use LoRA-optimized batch processing
            let tokenizer = self.configure_for_mode(TokenizationMode::LoRA)?;
            let encodings = tokenizer
                .encode_batch(texts.to_vec(), self.config.add_special_tokens)
                .map_err(E::msg)?;
            Ok(self.encodings_to_batch_result(&encodings))
        } else {
            // Use standard batch processing
            self.tokenize_batch(texts)
        }
    }

    fn get_config(&self) -> &TokenizationConfig {
        &self.config
    }

    fn supports_parallel(&self) -> bool {
        // LoRA models support parallel tokenization
        matches!(
            self.config.tokenization_strategy,
            TokenizationStrategy::LoRA
        )
    }

    fn create_tensors(&self, result: &TokenizationResult) -> Result<(Tensor, Tensor)> {
        self.create_tensors(result)
    }

    fn create_batch_tensors(&self, result: &BatchTokenizationResult) -> Result<(Tensor, Tensor)> {
        self.create_batch_tensors(result)
    }
}

/// Create tokenizer for specific tokenization strategy
///
/// ## Arguments
/// * `tokenizer_path` - Path to tokenizer.json file
/// * `tokenization_strategy` - Tokenization strategy (BERT, ModernBERT, LoRA, etc.)
/// * `device` - Computing device
///
/// ## Returns
/// * `Result<Box<dyn DualPathTokenizer>>` - Boxed tokenizer implementing dual-path interface
pub fn create_tokenizer(
    tokenizer_path: &str,
    tokenization_strategy: TokenizationStrategy,
    device: Device,
) -> Result<Box<dyn DualPathTokenizer>> {
    let tokenizer = UnifiedTokenizer::from_file(tokenizer_path, tokenization_strategy, device)?;
    Ok(Box::new(tokenizer))
}

/// Utility function to detect tokenization strategy from tokenizer configuration
pub fn detect_tokenization_strategy(tokenizer_path: &str) -> Result<TokenizationStrategy> {
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    // Try to detect tokenization strategy from tokenizer properties
    // This is a heuristic approach - in practice, you'd pass strategy explicitly
    let vocab_size = tokenizer.get_vocab_size(false);

    // mmBERT has vocab_size of 256000 for multilingual support
    if vocab_size >= 200000 {
        Ok(TokenizationStrategy::MmBERT)
    } else if vocab_size > 50000 {
        Ok(TokenizationStrategy::ModernBERT)
    } else {
        Ok(TokenizationStrategy::BERT)
    }
}

/// Detect if a model is mmBERT from config.json
pub fn detect_mmbert_from_config(config_path: &str) -> Result<bool> {
    let config_str = std::fs::read_to_string(config_path).map_err(E::msg)?;
    let config: serde_json::Value = serde_json::from_str(&config_str).map_err(E::msg)?;

    // Check for mmBERT-specific characteristics:
    // 1. vocab_size >= 200000 (mmBERT uses 256000)
    // 2. model_type == "modernbert"
    // 3. position_embedding_type == "sans_pos" (RoPE-based)
    let vocab_size = config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let position_embedding_type = config
        .get("position_embedding_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // mmBERT has large vocab (256000) and uses modernbert architecture with sans_pos
    Ok(vocab_size >= 200000 && model_type == "modernbert" && position_embedding_type == "sans_pos")
}

/// Legacy C-compatible tokenization result structure
///
/// This matches the original TokenizationResult from lib.rs for API compatibility
#[repr(C)]
pub struct CTokenizationResult {
    pub token_ids: *mut i32,
    pub token_count: i32,
    pub tokens: *mut *mut std::ffi::c_char,
    pub error: bool,
}

/// Convert TokenizationResult to C-compatible format
///
/// ## Arguments
/// * `result` - Rust tokenization result
///
/// ## Returns
/// * `CTokenizationResult` - C-compatible result with allocated memory
///
/// ## Safety
/// The returned pointers must be freed using appropriate free functions
pub fn tokenization_result_to_c(result: TokenizationResult) -> CTokenizationResult {
    use std::ffi::CString;

    let count = result.token_ids.len() as i32;

    // Allocate memory for token IDs
    let ids_ptr = result.token_ids.as_ptr() as *mut i32;
    std::mem::forget(result.token_ids); // Prevent deallocation

    // Allocate memory for tokens
    let c_tokens: Vec<*mut std::ffi::c_char> = result
        .tokens
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap().into_raw())
        .collect();

    let tokens_ptr = c_tokens.as_ptr() as *mut *mut std::ffi::c_char;
    std::mem::forget(c_tokens); // Prevent deallocation

    CTokenizationResult {
        token_ids: ids_ptr,
        token_count: count,
        tokens: tokens_ptr,
        error: false,
    }
}

/// Create error result for C FFI
pub fn create_c_tokenization_error() -> CTokenizationResult {
    CTokenizationResult {
        token_ids: std::ptr::null_mut(),
        token_count: 0,
        tokens: std::ptr::null_mut(),
        error: true,
    }
}

/// Compatibility function to wrap BertSimilarity tokenization
///
/// This provides the same interface as the original BertSimilarity.tokenize_text
/// but uses the new dual-path tokenization system internally.
pub fn tokenize_text_compat(
    tokenizer: &dyn DualPathTokenizer,
    text: &str,
    _max_length: Option<usize>,
) -> Result<(Vec<i32>, Vec<String>)> {
    let result = tokenizer.tokenize(text)?;
    Ok((result.token_ids, result.tokens))
}

/// Create a tokenizer from BertSimilarity for migration compatibility
///
/// This function allows existing BertSimilarity instances to be wrapped
/// with the new dual-path tokenization interface.
pub fn create_bert_compatibility_tokenizer(
    tokenizer: Tokenizer,
    device: Device,
) -> Result<Box<dyn DualPathTokenizer>> {
    let config = TokenizationConfig {
        tokenization_strategy: TokenizationStrategy::BERT,
        token_data_type: TokenDataType::I32,
        ..Default::default()
    };

    let unified_tokenizer = UnifiedTokenizer::new(tokenizer, config, device)?;
    Ok(Box::new(unified_tokenizer))
}

/// Create a tokenizer for ModernBERT compatibility
pub fn create_modernbert_compatibility_tokenizer(
    tokenizer: Tokenizer,
    device: Device,
) -> Result<Box<dyn DualPathTokenizer>> {
    let config = TokenizationConfig {
        tokenization_strategy: TokenizationStrategy::ModernBERT,
        token_data_type: TokenDataType::U32,
        ..Default::default()
    };

    let unified_tokenizer = UnifiedTokenizer::new(tokenizer, config, device)?;
    Ok(Box::new(unified_tokenizer))
}

/// Create a tokenizer for LoRA compatibility
pub fn create_lora_compatibility_tokenizer(
    tokenizer: Tokenizer,
    device: Device,
) -> Result<Box<dyn DualPathTokenizer>> {
    let config = TokenizationConfig {
        tokenization_strategy: TokenizationStrategy::LoRA,
        token_data_type: TokenDataType::U32, // LoRA typically uses u32
        ..Default::default()
    };

    let unified_tokenizer = UnifiedTokenizer::new(tokenizer, config, device)?;
    Ok(Box::new(unified_tokenizer))
}

/// Create a tokenizer for mmBERT compatibility (multilingual ModernBERT)
///
/// mmBERT is a multilingual encoder built on ModernBERT architecture with:
/// - 256k vocabulary for 1800+ language support
/// - 8192 max sequence length
/// - RoPE positional embeddings (sans_pos)
pub fn create_mmbert_compatibility_tokenizer(
    tokenizer: Tokenizer,
    device: Device,
) -> Result<Box<dyn DualPathTokenizer>> {
    let config = TokenizationConfig {
        tokenization_strategy: TokenizationStrategy::MmBERT,
        token_data_type: TokenDataType::U32,
        max_length: 8192,               // mmBERT supports 8k context
        pad_token_id: 0,                // mmBERT pad_token_id from config
        pad_token: "<pad>".to_string(), // mmBERT uses <pad> token
        ..Default::default()
    };

    let unified_tokenizer = UnifiedTokenizer::new(tokenizer, config, device)?;
    Ok(Box::new(unified_tokenizer))
}

/// Create a tokenizer for mmBERT with custom max length
pub fn create_mmbert_compatibility_tokenizer_with_max_length(
    tokenizer: Tokenizer,
    device: Device,
    max_length: usize,
) -> Result<Box<dyn DualPathTokenizer>> {
    let config = TokenizationConfig {
        tokenization_strategy: TokenizationStrategy::MmBERT,
        token_data_type: TokenDataType::U32,
        max_length,
        pad_token_id: 0,
        pad_token: "<pad>".to_string(),
        ..Default::default()
    };

    let unified_tokenizer = UnifiedTokenizer::new(tokenizer, config, device)?;
    Ok(Box::new(unified_tokenizer))
}
