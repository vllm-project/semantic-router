package memory

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// EmbeddingModelType represents supported embedding model types
type EmbeddingModelType string

const (
	EmbeddingModelBERT   EmbeddingModelType = "bert"
	EmbeddingModelMMBERT EmbeddingModelType = "mmbert"
	EmbeddingModelMulti  EmbeddingModelType = "multimodal"
	EmbeddingModelQwen3  EmbeddingModelType = "qwen3"
	EmbeddingModelGemma  EmbeddingModelType = "gemma"
)

// EmbeddingConfig holds the embedding model configuration
type EmbeddingConfig struct {
	Provider  embedding.Provider // Prepared by the generation owner; never closed by the store.
	Model     EmbeddingModelType
	Dimension int // Target dimension for Matryoshka models (default: 256 for mmbert)
	Layer     int // Target layer for 2D Matryoshka early exit (0 = full model, recommended for search/RAG)
}

// GenerateEmbedding generates an embedding for callers without a request context.
func GenerateEmbedding(text string, cfg EmbeddingConfig) ([]float32, error) {
	return GenerateEmbeddingWithContext(context.Background(), text, cfg)
}

// GenerateEmbeddingWithContext uses the provider prepared for this generation.
// The owner retains the native resource until inference and the generation drain.
func GenerateEmbeddingWithContext(ctx context.Context, text string, cfg EmbeddingConfig) ([]float32, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.Provider == nil && deterministicEmbeddingsEnabled() {
		return generateDeterministicEmbedding(text, cfg), nil
	}
	modelName := strings.ToLower(strings.TrimSpace(string(cfg.Model)))
	options := embedding.Options{}
	switch modelName {
	case "qwen3", "gemma", "bert", "":
		// These paths have always requested the provider's full output.
	case "mmbert":
		options = embedding.Options{Dimension: cfg.Dimension, Layer: cfg.Layer}
		if options.Dimension <= 0 {
			options.Dimension = 256
		}
	case "multimodal":
		options.Dimension = cfg.Dimension
		if options.Dimension <= 0 {
			options.Dimension = 384
		}
	default:
		return nil, fmt.Errorf("unsupported embedding model: %s (must be 'bert', 'qwen3', 'gemma', 'mmbert', or 'multimodal')", modelName)
	}
	vector, err := embedding.Embed(ctx, cfg.Provider, text, options)
	if err != nil {
		return nil, fmt.Errorf("%s embedding failed: %w", modelName, err)
	}
	return vector, nil
}
