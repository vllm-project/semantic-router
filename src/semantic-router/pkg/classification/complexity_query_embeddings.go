package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type complexityQueryEmbeddings struct {
	text   []float32
	mmText []float32
	image  []float32
}

func (c *ComplexityClassifier) loadQueryEmbeddingsCached(ctx context.Context, query string, imageURL string, cache *requestImageEmbeddingCache) (complexityQueryEmbeddings, error) {
	if err := ctx.Err(); err != nil {
		return complexityQueryEmbeddings{}, err
	}
	var embeddings complexityQueryEmbeddings
	var err error
	if c.provider != nil {
		embeddings.text, err = c.provider.Embed(ctx, query)
	} else {
		queryOutput, queryErr := getEmbeddingWithModelType(query, c.modelType, 0)
		err = queryErr
		if err == nil {
			embeddings.text = queryOutput.Embedding
		}
	}
	if err != nil {
		return complexityQueryEmbeddings{}, fmt.Errorf("failed to compute query embedding: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return complexityQueryEmbeddings{}, err
	}
	if !c.hasImageCandidates {
		return embeddings, nil
	}

	embeddings.mmText = c.loadOptionalMultiModalTextEmbedding(ctx, query)
	if imageURL != "" {
		embeddings.image = c.loadOptionalMultiModalImageEmbeddingCached(ctx, imageURL, cache)
	}
	// Optional modality failures may fall back to text, but a canceled request
	// must not publish a partial result or launch another native modality.
	return embeddings, ctx.Err()
}

func (c *ComplexityClassifier) loadOptionalMultiModalTextEmbedding(ctx context.Context, query string) []float32 {
	if ctx.Err() != nil {
		return nil
	}
	var vector []float32
	var err error
	if c.multiModalProvider != nil {
		vector, err = c.multiModalProvider.Embed(ctx, query)
	} else {
		vector, err = getMultiModalTextEmbedding(query, 0)
	}
	if err != nil {
		logging.Warnf("[Complexity Signal] Failed to compute multimodal text embedding: %v", err)
		return nil
	}
	return vector
}

func (c *ComplexityClassifier) loadOptionalMultiModalImageEmbeddingCached(ctx context.Context, imageURL string, cache *requestImageEmbeddingCache) []float32 {
	if cache == nil {
		embedding, err := c.embedMultiModalImage(ctx, imageURL)
		if err != nil {
			logging.Warnf("[Complexity Signal] Failed to compute request image embedding: %v", err)
			return nil
		}
		return embedding
	}

	embedding, err := cache.resolveFor(c.multiModalProvider, imageURL, 0, func() ([]float32, error) {
		return c.embedMultiModalImage(ctx, imageURL)
	})
	if err != nil {
		logging.Warnf("[Complexity Signal] Failed to compute request image embedding: %v", err)
		return nil
	}
	return embedding
}

func (c *ComplexityClassifier) embedMultiModalImage(ctx context.Context, ref string) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c.multiModalProvider != nil {
		return embedding.Image(ctx, c.multiModalProvider, ref, 0)
	}
	return getMultiModalImageEmbedding(ref, 0)
}
