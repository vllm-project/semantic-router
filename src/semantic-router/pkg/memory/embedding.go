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
	Model EmbeddingModelType
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
		// Use GetEmbedding2DMatryoshka for mmBERT (layer 6, 384-dim)
		output, err := candle_binding.GetEmbedding2DMatryoshka(text, modelName, 6, 384)
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
