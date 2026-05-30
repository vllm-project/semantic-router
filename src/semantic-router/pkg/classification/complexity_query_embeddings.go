package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type complexityQueryEmbeddings struct {
	text   []float32
	mmText []float32
	image  []float32
}

func (c *ComplexityClassifier) loadQueryEmbeddingsCached(query string, imageURL string, cache *requestImageEmbeddingCache) (complexityQueryEmbeddings, error) {
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
	embedding, err := getMultiModalTextEmbedding(query, 0)
	if err != nil {
		logging.Warnf("[Complexity Signal] Failed to compute multimodal text embedding: %v", err)
		return nil
	}
	return embedding
}

func (c *ComplexityClassifier) loadOptionalMultiModalImageEmbeddingCached(imageURL string, cache *requestImageEmbeddingCache) []float32 {
	if cache == nil {
		embedding, err := getMultiModalImageEmbedding(imageURL, 0)
		if err != nil {
			logging.Warnf("[Complexity Signal] Failed to compute request image embedding: %v", err)
			return nil
		}
		return embedding
	}

	embedding, err := cache.resolve(imageURL, 0, func() ([]float32, error) {
		return getMultiModalImageEmbedding(imageURL, 0)
	})
	if err != nil {
		logging.Warnf("[Complexity Signal] Failed to compute request image embedding: %v", err)
		return nil
	}
	return embedding
}
