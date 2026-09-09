package memory

import "context"

// GenerateEmbeddingContext checks cancellation on both sides of native work.
// The native ABI cannot interrupt an in-flight call: the calling worker must
// retain its slot and resources until it returns. A canceled result must never
// advance to backend persistence.
func GenerateEmbeddingContext(ctx context.Context, text string, cfg EmbeddingConfig) ([]float32, error) {
	return generateEmbeddingContext(ctx, text, cfg, GenerateEmbedding)
}

func generateEmbeddingContext(
	ctx context.Context, text string, cfg EmbeddingConfig,
	generate func(string, EmbeddingConfig) ([]float32, error),
) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	embedding, err := generate(text, cfg)
	if cause := ctx.Err(); cause != nil {
		return nil, cause
	}
	return embedding, err
}
