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

/// Model type for tokenization strategy selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    /// Traditional BERT models
    BERT,
    /// ModernBERT models
    ModernBERT,
    /// LoRA-enabled models
    LoRA,
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
    /// Model type for strategy selection
    pub model_type: ModelType,
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
            model_type: ModelType::BERT,
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
    /// * `model_type` - Model type for configuration
    /// * `device` - Computing device
    ///
    /// ## Returns
    /// * `Result<Self>` - Initialized unified tokenizer
    pub fn from_file(tokenizer_path: &str, model_type: ModelType, device: Device) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        let config = TokenizationConfig {
            model_type,
            token_data_type: match model_type {
                ModelType::ModernBERT => TokenDataType::U32,
                _ => TokenDataType::I32,
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
        let mode = match self.config.model_type {
            ModelType::ModernBERT => TokenizationMode::ModernBertBatch,
            ModelType::LoRA => TokenizationMode::LoRA,
            _ => TokenizationMode::Single,
        };

        match mode {
            TokenizationMode::ModernBertBatch => {
                // ModernBERT uses batch processing even for single text
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
        let mode = match self.config.model_type {
            ModelType::ModernBERT => TokenizationMode::ModernBertBatch,
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
        Ok(self.encoding_to_result(&encoding))
    }

    fn tokenize_batch_smart(
        &self,
        texts: &[&str],
        prefer_lora: bool,
    ) -> Result<BatchTokenizationResult> {
        if prefer_lora && self.config.model_type == ModelType::LoRA {
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
        matches!(self.config.model_type, ModelType::LoRA)
    }

    fn create_tensors(&self, result: &TokenizationResult) -> Result<(Tensor, Tensor)> {
        self.create_tensors(result)
    }

    fn create_batch_tensors(&self, result: &BatchTokenizationResult) -> Result<(Tensor, Tensor)> {
        self.create_batch_tensors(result)
    }
}

/// Create tokenizer for specific model type
///
/// ## Arguments
/// * `tokenizer_path` - Path to tokenizer.json file
/// * `model_type` - Model type (BERT, ModernBERT, LoRA)
/// * `device` - Computing device
///
/// ## Returns
/// * `Result<Box<dyn DualPathTokenizer>>` - Boxed tokenizer implementing dual-path interface
pub fn create_tokenizer(
    tokenizer_path: &str,
    model_type: ModelType,
    device: Device,
) -> Result<Box<dyn DualPathTokenizer>> {
    let tokenizer = UnifiedTokenizer::from_file(tokenizer_path, model_type, device)?;
    Ok(Box::new(tokenizer))
}

/// Utility function to detect model type from tokenizer configuration
pub fn detect_model_type(tokenizer_path: &str) -> Result<ModelType> {
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    // Try to detect model type from tokenizer properties
    // This is a heuristic approach - in practice, you'd pass model type explicitly
    let vocab_size = tokenizer.get_vocab_size(false);

    if vocab_size > 50000 {
        Ok(ModelType::ModernBERT)
    } else {
        Ok(ModelType::BERT)
    }
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
        model_type: ModelType::BERT,
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
        model_type: ModelType::ModernBERT,
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
        model_type: ModelType::LoRA,
        token_data_type: TokenDataType::U32, // LoRA typically uses u32
        ..Default::default()
    };

    let unified_tokenizer = UnifiedTokenizer::new(tokenizer, config, device)?;
    Ok(Box::new(unified_tokenizer))
}
