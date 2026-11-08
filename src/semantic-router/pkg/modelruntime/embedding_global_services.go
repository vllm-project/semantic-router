package modelruntime

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
)

// PrepareOwnedGlobalServiceEmbeddings gives the embeddings API, memory and
// ingestion independent global handles. Cache owns its separate
// layer/dimension view; compatible handles all acquire the same physical
// resource from the pool.
func PrepareOwnedGlobalServiceEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *serving.Runtime) (*embedding.Set, error) {
	if cfg == nil {
		return nil, fmt.Errorf("global model services require configuration")
	}
	scoped := cfg.ConfigForGlobalModelServices()
	scoped.SemanticCache.InUse = false
	return prepareEmbeddings(ctx, scoped, runtime, true, embedding.Options{})
}
