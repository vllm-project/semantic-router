package classification

import (
	"context"

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
			b.embeddingSet, b.providerErr = modelruntime.PrepareOwnedEmbeddings(context.Background(), b.cfg, b.models.runtime)
			b.ownsEmbeddingSet = b.providerErr == nil
		}
	})
	return b.providerErr
}

// PreparedEmbeddings is this recipe's prepared embedding snapshot.
func (c *Classifier) PreparedEmbeddings() *embedding.Set {
	if c == nil {
		return nil
	}
	return c.embeddingSet
}
