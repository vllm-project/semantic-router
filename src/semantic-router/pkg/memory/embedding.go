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
		if strings.EqualFold(strings.TrimSpace(string(cfg.Model)), "multimodal") && cfg.Dimension <= 0 {
			return nil, fmt.Errorf("simulated multimodal memory requires an explicit dimension")
		}
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
		dimension, err := embedding.ResolveDimension(cfg.Provider, cfg.Dimension)
		if err != nil {
			return nil, err
		}
		options.Dimension = dimension
	default:
		return nil, fmt.Errorf("unsupported embedding model: %s (must be 'bert', 'qwen3', 'gemma', 'mmbert', or 'multimodal')", modelName)
	}
	vector, err := embedding.Embed(ctx, cfg.Provider, text, options)
	if err != nil {
		return nil, fmt.Errorf("%s embedding failed: %w", modelName, err)
	}
	if modelName == "multimodal" && len(vector) != options.Dimension {
		return nil, fmt.Errorf("multimodal embedding returned %d values, expected %d", len(vector), options.Dimension)
	}
	return vector, nil
}

// embedForWrite discards a vector produced after cancellation: a write path must
// not persist it. Retrieval keeps a result the caller has already paid for, since
// the native ABI cannot be interrupted and the work is done either way.
func embedForWrite(ctx context.Context, text string, cfg EmbeddingConfig) ([]float32, error) {
	vector, err := GenerateEmbeddingWithContext(ctx, text, cfg)
	if err != nil {
		return nil, err
	}
	if cause := ctx.Err(); cause != nil {
		return nil, cause
	}
	return vector, nil
}

// StorageDimension binds a fixed-width index to its prepared representation.
// Store-only callers without a provider must supply an explicit schema width.
func StorageDimension(configured int, cfg EmbeddingConfig) (int, error) {
	if configured == 0 {
		configured = cfg.Dimension
	}
	if configured == 0 && cfg.Model == EmbeddingModelMMBERT {
		configured = 256
	}
	if cfg.Provider != nil {
		return embedding.ResolveDimension(cfg.Provider, configured)
	}
	if configured > 0 {
		return configured, nil
	}
	return 0, fmt.Errorf("memory vector storage needs an explicit dimension or prepared embedding provider")
}
