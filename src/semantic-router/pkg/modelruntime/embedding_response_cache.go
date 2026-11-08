package modelruntime

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
)

// PrepareOwnedResponseCacheEmbeddings gives the shared cache its own typed
// consumer handle over its embedding deployment, prepared at the view the
// cache reads. Its resource reference uses the same pool as recipe handles,
// without borrowing any recipe's override.
func PrepareOwnedResponseCacheEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *serving.Runtime) (*embedding.Set, error) {
	if cfg == nil {
		return nil, fmt.Errorf("response cache requires model configuration")
	}
	if !cfg.NeedsSemanticResponseCache() {
		return embedding.NewSet(nil, ""), nil
	}
	// An empty routing profile carries service settings without routing
	// signals, selectors or recipe overrides: only the cache's demand.
	scoped := cfg.ConfigForGlobalModelServices()
	scoped.API.Embeddings.Enabled = false
	scoped.Memory.Enabled = false
	scoped.VectorStore = nil
	_, dimension, layer := scoped.ResponseCacheEmbedding()
	return prepareEmbeddings(ctx, scoped, runtime, true, embedding.Options{Dimension: dimension, Layer: layer})
}
