package memory

import (
	"fmt"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// EmbeddingModelType represents supported embedding model types
type EmbeddingModelType string

const (
	EmbeddingModelBERT   EmbeddingModelType = "bert"
	EmbeddingModelMMBERT EmbeddingModelType = "mmbert"
	EmbeddingModelQwen3  EmbeddingModelType = "qwen3"
	EmbeddingModelGemma  EmbeddingModelType = "gemma"
)

// EmbeddingConfig holds the embedding model configuration
type EmbeddingConfig struct {
	Model     EmbeddingModelType
	Dimension int // Target dimension for Matryoshka models (default: 256 for mmbert)
}

// GenerateEmbedding generates an embedding using the configured model
func GenerateEmbedding(text string, cfg EmbeddingConfig) ([]float32, error) {
	modelName := strings.ToLower(strings.TrimSpace(string(cfg.Model)))

	switch modelName {
	case "qwen3":
		// Use GetEmbeddingBatched for Qwen3 with continuous batching
		output, err := candle_binding.GetEmbeddingBatched(text, modelName, 0)
		if err != nil {
			return nil, fmt.Errorf("qwen3 embedding failed: %w", err)
		}
		return output.Embedding, nil

	case "gemma":
		// Use GetEmbeddingWithModelType for Gemma
		output, err := candle_binding.GetEmbeddingWithModelType(text, modelName, 0)
		if err != nil {
			return nil, fmt.Errorf("gemma embedding failed: %w", err)
		}
		return output.Embedding, nil

	case "mmbert":
		// Use GetEmbedding2DMatryoshka for mmBERT with configurable dimension
		// Default to 256 if not specified (99% quality retention)
		targetDim := cfg.Dimension
		if targetDim <= 0 {
			targetDim = 256
		}
		// Use layer 6 for early exit (good balance of speed/quality)
		output, err := candle_binding.GetEmbedding2DMatryoshka(text, modelName, 6, targetDim)
		if err != nil {
			return nil, fmt.Errorf("mmbert embedding failed: %w", err)
		}
		return output.Embedding, nil

	case "bert", "":
		// Use traditional GetEmbedding for BERT (default)
		embedding, err := candle_binding.GetEmbedding(text, 0)
		if err != nil {
			return nil, fmt.Errorf("bert embedding failed: %w", err)
		}
		return embedding, nil

	default:
		return nil, fmt.Errorf("unsupported embedding model: %s (must be 'bert', 'qwen3', 'gemma', or 'mmbert')", modelName)
	}
}
