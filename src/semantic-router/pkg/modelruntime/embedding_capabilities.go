package modelruntime

import (
	"fmt"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func batchedEmbeddingNeeds(cfg *config.RouterConfig, qwen3Path string) (bool, bool, error) {
	semanticCacheNeedsBatched := false
	if cfg.Enabled && strings.TrimSpace(cfg.EmbeddingModel) != "" && qwen3Path != "" {
		capabilities, err := candle_binding.EmbeddingCapabilitiesFor(cfg.EmbeddingModel)
		if err != nil {
			return false, false, fmt.Errorf("semantic cache embedding capabilities: %w", err)
		}
		semanticCacheNeedsBatched = capabilities.SupportsBatching
	}

	mlSelectionNeedsBatched := false
	if cfg.ModelSelection.Enabled &&
		cfg.ModelSelection.ML.ModelsPath != "" &&
		cfg.Qwen3ModelPath != "" {
		capabilities, err := candle_binding.EmbeddingCapabilitiesFor(config.EmbeddingModelTypeQwen3)
		if err != nil {
			return false, false, fmt.Errorf("ML selection embedding capabilities: %w", err)
		}
		mlSelectionNeedsBatched = capabilities.SupportsBatching
	}
	return semanticCacheNeedsBatched, mlSelectionNeedsBatched, nil
}
