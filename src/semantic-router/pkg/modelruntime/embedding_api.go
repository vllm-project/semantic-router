package modelruntime

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// PrepareOwnedEmbeddingAPI supports standalone management servers. Router
// generations borrow the same API view from their global service embeddings.
func PrepareOwnedEmbeddingAPI(ctx context.Context, cfg *config.RouterConfig, runtime *native.Runtime) (*embedding.Set, error) {
	if cfg == nil || !cfg.API.Embeddings.Enabled {
		return nil, nil
	}
	scoped := cfg.ConfigForGlobalModelServices()
	scoped.SemanticCache.Enabled = false
	scoped.Tools.Enabled = false
	scoped.Memory.Enabled = false
	scoped.VectorStore = nil
	return prepareEmbeddings(ctx, scoped, runtime, true, embedding.Options{})
}
