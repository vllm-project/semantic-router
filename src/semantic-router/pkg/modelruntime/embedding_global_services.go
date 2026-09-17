package modelruntime

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// PrepareOwnedGlobalServiceEmbeddings gives tools, memory, and ingestion
// independent global handles. Cache owns its separate layer/dimension view;
// compatible handles all acquire the same physical resource from the pool.
func PrepareOwnedGlobalServiceEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *native.Runtime) (*embedding.Set, error) {
	if cfg == nil {
		return nil, fmt.Errorf("global model services require configuration")
	}
	scoped := cfg.ConfigForGlobalModelServices()
	scoped.SemanticCache.Enabled = false
	return prepareEmbeddings(ctx, scoped, runtime, true, embedding.Options{})
}

func globalEmbeddingConsumerName(cfg *config.RouterConfig, model, primary string) string {
	var consumers []string
	if cfg.SemanticCache.Enabled && config.SemanticCacheEmbeddingModel(cfg) == model {
		consumers = append(consumers, "response_cache")
	}
	if cfg.Tools.Enabled && model == primary {
		consumers = append(consumers, "tools")
	}
	if cfg.Memory.Enabled && config.MemoryEmbeddingModel(cfg) == model {
		consumers = append(consumers, "memory")
	}
	if cfg.VectorStore != nil && cfg.VectorStore.Enabled {
		selected := cfg.VectorStore.EmbeddingModel
		if selected == "" {
			selected = "bert"
		}
		if selected == model {
			consumers = append(consumers, "vector_store")
		}
	}
	return strings.Join(consumers, "+") + ".embedding"
}
