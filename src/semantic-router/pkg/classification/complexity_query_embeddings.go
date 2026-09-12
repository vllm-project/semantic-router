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

func (c *ComplexityClassifier) loadQueryEmbeddingsCached(query string, imageURL string, cache *requestImageEmbeddingCache) (complexityQueryEmbeddings, error) {
	if c.provider != nil {
		embedding, err := c.provider.Embed(context.Background(), query)
		if err != nil {
			return complexityQueryEmbeddings{}, fmt.Errorf("failed to compute query embedding: %w", err)
		}
		embeddings := complexityQueryEmbeddings{text: embedding}
		if !c.hasImageCandidates {
			return embeddings, nil
		}

		embeddings.mmText = c.loadOptionalMultiModalTextEmbedding(query)
		if imageURL != "" {
			embeddings.image = c.loadOptionalMultiModalImageEmbeddingCached(imageURL, cache)
		}
		return embeddings, nil
	}

	queryOutput, err := getEmbeddingWithModelType(query, c.modelType, 0)
	if err != nil {
		return complexityQueryEmbeddings{}, fmt.Errorf("failed to compute query embedding: %w", err)
	}

	embeddings := complexityQueryEmbeddings{text: queryOutput.Embedding}
	if !c.hasImageCandidates {
		return embeddings, nil
	}

	embeddings.mmText = c.loadOptionalMultiModalTextEmbedding(query)
	if imageURL != "" {
		embeddings.image = c.loadOptionalMultiModalImageEmbeddingCached(imageURL, cache)
	}
	return embeddings, nil
}

func (c *ComplexityClassifier) loadOptionalMultiModalTextEmbedding(query string) []float32 {
	var vector []float32
	var err error
	if c.multiModalProvider != nil {
		vector, err = c.multiModalProvider.Embed(context.Background(), query)
	} else {
		vector, err = getMultiModalTextEmbedding(query, 0)
	}
	if err != nil {
		logging.Warnf("[Complexity Signal] Failed to compute multimodal text embedding: %v", err)
		return nil
	}
	return vector
}

func (c *ComplexityClassifier) loadOptionalMultiModalImageEmbeddingCached(imageURL string, cache *requestImageEmbeddingCache) []float32 {
	if cache == nil {
		embedding, err := c.embedMultiModalImage(imageURL)
		if err != nil {
			logging.Warnf("[Complexity Signal] Failed to compute request image embedding: %v", err)
			return nil
		}
		return embedding
	}

	embedding, err := cache.resolveFor(c.multiModalProvider, imageURL, 0, func() ([]float32, error) {
		return c.embedMultiModalImage(imageURL)
	})
	if err != nil {
		logging.Warnf("[Complexity Signal] Failed to compute request image embedding: %v", err)
		return nil
	}
	return embedding
}

func (c *ComplexityClassifier) embedMultiModalImage(ref string) ([]float32, error) {
	if c.multiModalProvider != nil {
		return embedding.Image(context.Background(), c.multiModalProvider, ref, 0)
	}
	return getMultiModalImageEmbedding(ref, 0)
}
