package memory

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// EmbeddingConfig holds the embedding deployment memories are embedded with
// and the view they read.
type EmbeddingConfig struct {
	Provider   embedding.Provider // Prepared by the generation owner; never closed by the store.
	Deployment string
	Dimension  int // Width of the vectors (0: the model's full output)
	Layer      int // Target layer for 2D Matryoshka early exit (0 = full model, recommended for search/RAG)
}

// defaultEmbeddingConfig embeds with Vela Embedding at its memory view, for
// stores built without an embedding configuration.
func defaultEmbeddingConfig() EmbeddingConfig {
	return EmbeddingConfig{Deployment: config.EmbeddingDefaultDeployment, Dimension: config.VelaEmbeddingMemoryDimension}
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
	if cfg.Dimension > 0 {
		if _, err := embedding.ResolveDimension(cfg.Provider, cfg.Dimension); err != nil {
			return nil, err
		}
	}
	vector, err := embedding.Embed(ctx, cfg.Provider, text, embedding.Options{Dimension: cfg.Dimension, Layer: cfg.Layer})
	if err != nil {
		return nil, fmt.Errorf("%s embedding failed: %w", cfg.Deployment, err)
	}
	if cfg.Dimension > 0 && len(vector) != cfg.Dimension {
		return nil, fmt.Errorf("%s embedding returned %d values, expected %d", cfg.Deployment, len(vector), cfg.Dimension)
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

// memoryUpdateEmbedding keeps an existing vector only when its content still
// describes the memory being written. An explicitly supplied vector is
// authoritative; otherwise changed content must be embedded before persistence.
func memoryUpdateEmbedding(ctx context.Context, memory, existing *Memory, cfg EmbeddingConfig) ([]float32, error) {
	if len(memory.Embedding) > 0 {
		return memory.Embedding, nil
	}
	if existing != nil && memory.Content == existing.Content && len(existing.Embedding) > 0 {
		return existing.Embedding, nil
	}

	vector, err := embedForWrite(ctx, memory.Content, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}
	return vector, nil
}

// StorageDimension binds a fixed-width index to its prepared representation.
// Store-only callers without a provider must supply an explicit schema width.
func StorageDimension(configured int, cfg EmbeddingConfig) (int, error) {
	if configured == 0 {
		configured = cfg.Dimension
	}
	if cfg.Provider != nil {
		return embedding.ResolveDimension(cfg.Provider, configured)
	}
	if configured > 0 {
		return configured, nil
	}
	return 0, fmt.Errorf("memory vector storage needs an explicit dimension or prepared embedding provider")
}
