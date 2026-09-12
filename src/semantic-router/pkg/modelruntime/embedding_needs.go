package modelruntime

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func embeddingNeedsForScope(cfg *config.RouterConfig, primary string, sharedServices bool) map[string]bool {
	return config.EmbeddingModelsNeeded(cfg, primary, sharedServices)
}
