package modellifecycle

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func ResolveSemanticCacheEmbeddingModel(cfg *config.RouterConfig) string {
	if cfg == nil {
		return "bert"
	}

	embeddingModel := strings.ToLower(strings.TrimSpace(cfg.EmbeddingModel))
	if embeddingModel != "" {
		return embeddingModel
	}
	return "bert"
}

func ResolveMemoryEmbeddingModel(cfg *config.RouterConfig) string {
	if cfg == nil {
		return "bert"
	}

	embeddingModel := strings.ToLower(strings.TrimSpace(cfg.Memory.EmbeddingModel))
	if embeddingModel != "" {
		return embeddingModel
	}
	return "bert"
}
