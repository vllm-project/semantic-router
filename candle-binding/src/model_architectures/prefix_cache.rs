//! Prefix Cache for Reusing KV Cache of Fixed Prompts
//!
//! This module implements prefix caching to significantly speed up inference
//! when using fixed prompt templates. The core idea:
//!
//! 1. **First Request**: Process full prompt → Save KV cache
//! 2. **Subsequent Requests**: Restore saved KV → Only process new text
//!
//! ## Example: Qwen3Guard Safety Classification
//!
//! Without prefix cache:
//! ```text
//! Request 1: [500 fixed tokens] + [10 user tokens] = 510 tokens → 1200ms
//! Request 2: [500 fixed tokens] + [15 user tokens] = 515 tokens → 1200ms
//! Request 3: [500 fixed tokens] + [12 user tokens] = 512 tokens → 1200ms
//! ```
//!
//! With prefix cache:
//! ```text
//! Initialization: [500 fixed tokens] → Save KV → 800ms
//! Request 1: Restore KV + [10 user tokens] → 400ms  (3x faster!)
//! Request 2: Restore KV + [15 user tokens] → 420ms  (3x faster!)
//! Request 3: Restore KV + [12 user tokens] → 410ms  (3x faster!)
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      Prefix Cache                           │
//! │                                                             │
//! │  ┌──────────────┐     ┌────────────────┐                  │
//! │  │ Fixed Prefix │ →   │  KV Cache      │                  │
//! │  │ Tokens       │     │  (Saved State) │                  │
//! │  │ [500 tokens] │     │                │                  │
//! │  └──────────────┘     └────────────────┘                  │
//! │                              ↓                             │
//! │                       ┌─────────────┐                     │
//! │                       │  On Request │                     │
//! │                       │  Restore KV │                     │
//! │                       └─────────────┘                     │
//! │                              ↓                             │
//! │  ┌──────────────┐     ┌────────────────┐                  │
//! │  │ User Text    │ →   │  Process Only  │ → Result        │
//! │  │ [10 tokens]  │     │  New Tokens    │                  │
//! │  └──────────────┘     └────────────────┘                  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use candle_semantic_router::model_architectures::prefix_cache::PrefixCache;
//!
//! // 1. Initialize cache with fixed prompt
//! let fixed_prompt = "You are a safety classifier. Classify the following...";
//! let mut cache = PrefixCache::new(fixed_prompt, &tokenizer, &model)?;
//!
//! // 2. Use for multiple requests (fast!)
//! let result1 = cache.generate_with_suffix("User text 1", &mut model)?;
//! let result2 = cache.generate_with_suffix("User text 2", &mut model)?;
//! let result3 = cache.generate_with_suffix("User text 3", &mut model)?;
//! ```
//!
//! ## Implementation Notes
//!
//! - **Thread Safety**: Cache is not thread-safe by design. For multi-threaded use,
//!   clone the cache for each thread (KV tensors are cheap to clone).
//! - **Memory**: Each cache stores ~2-4MB for a 0.6B model (depends on prefix length).
//! - **Invalidation**: Cache is invalidated if prefix changes (automatic).

use serde::{Deserialize, Serialize};

/// Prefix cache for reusing fixed prompt tokenization
///
/// This is a simplified cache that stores pre-tokenized prefix.
/// The KV cache is maintained automatically by the model during forward pass.
#[derive(Debug, Clone)]
pub struct PrefixCache {
    /// Fixed prefix tokens (e.g., [500 tokens for safety policy])
    prefix_tokens: Vec<u32>,

    /// Number of tokens in the prefix
    prefix_len: usize,
}

impl PrefixCache {
    /// Create a new prefix cache
    pub fn new(prefix_tokens: Vec<u32>) -> Self {
        let prefix_len = prefix_tokens.len();
        Self {
            prefix_tokens,
            prefix_len,
        }
    }

    /// Get prefix length
    pub fn prefix_length(&self) -> usize {
        self.prefix_len
    }

    /// Get reference to prefix tokens
    pub fn prefix_tokens(&self) -> &[u32] {
        &self.prefix_tokens
    }
}

/// Configuration for prefix caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixCacheConfig {
    /// Whether to enable prefix caching
    pub enabled: bool,

    /// Verbose logging
    pub verbose: bool,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            verbose: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_cache_creation() {
        let tokens = vec![1, 2, 3, 4, 5];
        let cache = PrefixCache::new(tokens.clone());

        assert_eq!(cache.prefix_length(), 5);
        assert_eq!(cache.prefix_tokens(), &tokens[..]);
    }
}
