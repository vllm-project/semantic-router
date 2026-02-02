//! FFI type definitions for C/Go interop

use std::ffi::c_char;

/// Embedding result structure for FFI
#[repr(C)]
pub struct EmbeddingResult {
    /// Pointer to embedding data (array of f32)
    pub data: *mut f32,
    /// Length of the embedding vector
    pub length: i32,
    /// Whether an error occurred
    pub error: bool,
    /// Model type: 0=mmbert, -1=error
    pub model_type: i32,
    /// Sequence length in tokens
    pub sequence_length: i32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
}

impl Default for EmbeddingResult {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            length: 0,
            error: true,
            model_type: -1,
            sequence_length: 0,
            processing_time_ms: 0.0,
        }
    }
}

/// Embedding similarity result structure
#[repr(C)]
pub struct EmbeddingSimilarityResult {
    /// Cosine similarity score (-1.0 to 1.0)
    pub similarity: f32,
    /// Model type: 0=mmbert, -1=error
    pub model_type: i32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Whether an error occurred
    pub error: bool,
}

impl Default for EmbeddingSimilarityResult {
    fn default() -> Self {
        Self {
            similarity: 0.0,
            model_type: -1,
            processing_time_ms: 0.0,
            error: true,
        }
    }
}

/// Batch similarity match structure
#[repr(C)]
pub struct SimilarityMatch {
    /// Index of the candidate in the input array
    pub index: i32,
    /// Cosine similarity score
    pub similarity: f32,
}

/// Batch similarity result structure
#[repr(C)]
pub struct BatchSimilarityResult {
    /// Array of top-k matches, sorted by similarity (descending)
    pub matches: *mut SimilarityMatch,
    /// Number of matches returned
    pub num_matches: i32,
    /// Model type: 0=mmbert, -1=error
    pub model_type: i32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Whether an error occurred
    pub error: bool,
}

impl Default for BatchSimilarityResult {
    fn default() -> Self {
        Self {
            matches: std::ptr::null_mut(),
            num_matches: 0,
            model_type: -1,
            processing_time_ms: 0.0,
            error: true,
        }
    }
}

/// Model information structure
#[repr(C)]
pub struct EmbeddingModelInfo {
    /// Model name ("mmbert")
    pub model_name: *mut c_char,
    /// Whether the model is loaded
    pub is_loaded: bool,
    /// Maximum sequence length
    pub max_sequence_length: i32,
    /// Default embedding dimension
    pub default_dimension: i32,
    /// Model path (can be null if not loaded)
    pub model_path: *mut c_char,
    /// Whether layer early exit is supported
    pub supports_layer_exit: bool,
    /// Available exit layers (comma-separated, e.g., "3,6,11,22")
    pub available_layers: *mut c_char,
}

/// Models information result structure
#[repr(C)]
pub struct EmbeddingModelsInfoResult {
    /// Array of model info
    pub models: *mut EmbeddingModelInfo,
    /// Number of models
    pub num_models: i32,
    /// Whether an error occurred
    pub error: bool,
}

impl Default for EmbeddingModelsInfoResult {
    fn default() -> Self {
        Self {
            models: std::ptr::null_mut(),
            num_models: 0,
            error: true,
        }
    }
}

/// Matryoshka configuration info for FFI
#[repr(C)]
pub struct MatryoshkaInfo {
    /// Supported dimensions (comma-separated, e.g., "768,512,256,128,64")
    pub dimensions: *mut c_char,
    /// Supported layers (comma-separated, e.g., "3,6,11,22")
    pub layers: *mut c_char,
    /// Whether 2D Matryoshka is fully supported
    pub supports_2d: bool,
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_result_default() {
        let result = EmbeddingResult::default();
        assert!(result.data.is_null());
        assert_eq!(result.length, 0);
        assert!(result.error);
        assert_eq!(result.model_type, -1);
        assert_eq!(result.sequence_length, 0);
        assert_eq!(result.processing_time_ms, 0.0);
    }

    #[test]
    fn test_embedding_similarity_result_default() {
        let result = EmbeddingSimilarityResult::default();
        assert_eq!(result.similarity, 0.0);
        assert_eq!(result.model_type, -1);
        assert_eq!(result.processing_time_ms, 0.0);
        assert!(result.error);
    }

    #[test]
    fn test_batch_similarity_result_default() {
        let result = BatchSimilarityResult::default();
        assert!(result.matches.is_null());
        assert_eq!(result.num_matches, 0);
        assert_eq!(result.model_type, -1);
        assert!(result.error);
    }

    #[test]
    fn test_embedding_models_info_result_default() {
        let result = EmbeddingModelsInfoResult::default();
        assert!(result.models.is_null());
        assert_eq!(result.num_models, 0);
        assert!(result.error);
    }

    #[test]
    fn test_similarity_match_struct() {
        let match_result = SimilarityMatch {
            index: 5,
            similarity: 0.95,
        };
        assert_eq!(match_result.index, 5);
        assert!((match_result.similarity - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_embedding_result_struct_size() {
        // Ensure struct has expected layout for FFI
        assert!(std::mem::size_of::<EmbeddingResult>() > 0);
        assert!(std::mem::size_of::<EmbeddingSimilarityResult>() > 0);
        assert!(std::mem::size_of::<BatchSimilarityResult>() > 0);
        assert!(std::mem::size_of::<EmbeddingModelInfo>() > 0);
        assert!(std::mem::size_of::<MatryoshkaInfo>() > 0);
    }
}
