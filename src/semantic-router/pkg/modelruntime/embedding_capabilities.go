package modelruntime

import (
	"fmt"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func batchedEmbeddingNeeds(cfg *config.RouterConfig, paths embeddingPaths) (bool, bool, error) {
	semanticCacheNeedsBatched := false
	semanticCacheModelType := resolveSemanticCacheEmbeddingModel(cfg)
	if cfg.Enabled && unifiedEmbeddingModelConfigured(paths, semanticCacheModelType) {
		capabilities, err := candle_binding.EmbeddingCapabilitiesFor(semanticCacheModelType)
		if err != nil {
			return false, false, fmt.Errorf("semantic cache embedding capabilities: %w", err)
		}
		semanticCacheNeedsBatched = capabilities.SupportsBatching
	}

	mlSelectionNeedsBatched := false
	if cfg.ModelSelection.Enabled &&
		cfg.ModelSelection.ML.ModelsPath != "" {
		mlModelType := strings.TrimSpace(cfg.ModelSelection.ML.ModelType)
		if mlModelType == "" {
			mlModelType = string(candle_binding.DefaultEmbeddingModelType)
		}
		if !unifiedEmbeddingModelConfigured(paths, mlModelType) {
			return semanticCacheNeedsBatched, false, nil
		}
		capabilities, err := candle_binding.EmbeddingCapabilitiesFor(mlModelType)
		if err != nil {
			return false, false, fmt.Errorf("ML selection embedding capabilities: %w", err)
		}
		mlSelectionNeedsBatched = capabilities.SupportsBatching
	}
	return semanticCacheNeedsBatched, mlSelectionNeedsBatched, nil
}

func unifiedEmbeddingModelConfigured(paths embeddingPaths, modelType string) bool {
	switch strings.ToLower(strings.TrimSpace(modelType)) {
	case "qwen3":
		return paths.qwen3 != ""
	case "gemma":
		return paths.gemma != ""
	case "mmbert":
		return paths.mmBert != ""
	default:
		return false
	}
}
