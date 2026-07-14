package classification

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type complexityQueryEmbeddings struct {
	text   []float32
	mmText []float32
	image  []float32
}

func (c *ComplexityClassifier) loadQueryEmbeddingsCached(query string, imageURL string, cache *requestImageEmbeddingCache) (complexityQueryEmbeddings, error) {
	query = strings.TrimSpace(query)
	imageURL = strings.TrimSpace(imageURL)
	if query == "" && imageURL == "" {
		return complexityQueryEmbeddings{}, fmt.Errorf("complexity query or image must be provided")
	}

	var embeddings complexityQueryEmbeddings
	if query != "" {
		textEmbedding, _, err := executeTextEmbedding(context.Background(), c.backend, c.provider, query, c.modelType, 0)
		if err != nil {
			return complexityQueryEmbeddings{}, fmt.Errorf("failed to compute query embedding: %w", err)
		}
		embeddings.text = textEmbedding
	}
	if !c.hasImageCandidates {
		return embeddings, nil
	}
	if query != "" {
		embeddings.mmText = c.loadOptionalMultiModalTextEmbedding(query)
	}
	if imageURL != "" {
		imageEmbedding, err := c.loadMultiModalImageEmbeddingCached(imageURL, cache)
		if err != nil {
			return complexityQueryEmbeddings{}, newImageSignalEvaluationError(err)
		}
		embeddings.image = imageEmbedding
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

func (c *ComplexityClassifier) loadMultiModalImageEmbeddingCached(imageURL string, cache *requestImageEmbeddingCache) ([]float32, error) {
	if cache == nil {
		embedding, err := getMultiModalImageEmbedding(imageURL, 0)
		if err != nil {
			return nil, err
		}
		return embedding, nil
	}

	embedding, err := cache.resolve(imageURL, 0, func() ([]float32, error) {
		return getMultiModalImageEmbedding(imageURL, 0)
	})
	if err != nil {
		return nil, err
	}
	return embedding, nil
}
