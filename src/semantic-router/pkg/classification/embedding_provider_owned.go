package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
)

func (b *classifierOptionBuilder) prepareEmbeddingSet() error {
	b.providerInitOnce.Do(func() {
		if b.embeddingSet == nil {
			if b.models == nil {
				b.models = standaloneModelRuntime()
				b.models.cfg = b.cfg
			}
			b.embeddingSet, b.providerErr = modelruntime.PrepareOwnedRecipeEmbeddings(context.Background(), b.cfg, b.models.runtime)
			b.ownsEmbeddingSet = b.providerErr == nil
		}
		if b.providerErr == nil {
			b.provider, _ = b.embeddingSet.Get("", 0, 0)
		}
	})
	return b.providerErr
}

func (b *classifierOptionBuilder) embeddingProviderForModel(model string, dimension, layer int) (embedding.Provider, error) {
	if err := b.prepareEmbeddingSet(); err != nil {
		return nil, err
	}
	return b.embeddingSet.Get(model, dimension, layer)
}

// EmbeddingForModel reads only this recipe's prepared snapshot.
func (c *Classifier) EmbeddingForModel(model string, dimension, layer int) (embedding.Provider, error) {
	if c == nil {
		return nil, fmt.Errorf("recipe classifier is unavailable")
	}
	if c.embeddingSet != nil {
		return c.embeddingSet.Get(model, dimension, layer)
	}
	if model == "" && c.embeddingProvider != nil {
		return embedding.WithOptions(c.embeddingProvider, embedding.Options{Dimension: dimension, Layer: layer}), nil
	}
	return nil, fmt.Errorf("embedding provider was not prepared for this recipe")
}

func (c *Classifier) PreparedEmbeddings() *embedding.Set {
	if c == nil {
		return nil
	}
	return c.embeddingSet
}
