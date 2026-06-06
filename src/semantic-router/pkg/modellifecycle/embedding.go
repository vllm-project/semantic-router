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

	switch {
	case cfg.MmBertModelPath != "":
		return "mmbert"
	case cfg.MultiModalModelPath != "":
		return "multimodal"
	case cfg.Qwen3ModelPath != "":
		return "qwen3"
	case cfg.GemmaModelPath != "":
		return "gemma"
	default:
		return "bert"
	}
}

func ResolveMemoryEmbeddingModel(cfg *config.RouterConfig) string {
	if cfg == nil {
		return "bert"
	}

	embeddingModel := strings.ToLower(strings.TrimSpace(cfg.Memory.EmbeddingModel))
	if embeddingModel != "" {
		return embeddingModel
	}

	embeddingModels := cfg.EmbeddingModels
	switch {
	case embeddingModels.BertModelPath != "":
		return "bert"
	case embeddingModels.MmBertModelPath != "":
		return "mmbert"
	case embeddingModels.MultiModalModelPath != "":
		return "multimodal"
	case embeddingModels.Qwen3ModelPath != "":
		return "qwen3"
	case embeddingModels.GemmaModelPath != "":
		return "gemma"
	default:
		return "bert"
	}
}
