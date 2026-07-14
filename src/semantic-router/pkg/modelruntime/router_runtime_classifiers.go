package modelruntime

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// classifierEmbeddingPaths returns every local text model that classifier
// construction can request. Contrastive preference routing may intentionally
// use a different model from the default routing families, while Candle owns a
// single process-global factory that must receive the complete model union on
// its first initialization.
func classifierEmbeddingPaths(
	cfg *config.RouterConfig,
	configured embeddingPaths,
	backend string,
) (embeddingPaths, error) {
	requestedModels := classifierEmbeddingModelRequests(cfg)
	if len(requestedModels) == 0 {
		return embeddingPaths{}, nil
	}
	plans, err := embedding.ResolveRuntimePlans(
		cfg.EmbeddingModels,
		backend,
		embedding.ModelTypeOverrideFromEnv(),
		requestedModels...,
	)
	if err != nil {
		return embeddingPaths{}, fmt.Errorf("resolve classifier embedding models: %w", err)
	}
	result := embeddingPaths{}
	for _, plan := range plans {
		if plan.Backend != config.EmbeddingBackendCandle {
			continue
		}
		result = result.union(configured.forRuntimePlan(plan))
	}
	return result, nil
}

func classifierEmbeddingModelRequests(cfg *config.RouterConfig) []string {
	if cfg == nil {
		return nil
	}
	requests := make([]string, 0, 2)
	if classifierUsesDefaultTextEmbedding(cfg) {
		requests = append(requests, cfg.EmbeddingConfig.ModelType)
	}
	if len(cfg.PreferenceRules) > 0 && cfg.PreferenceModel.ContrastiveEnabled() {
		requests = append(requests, cfg.PreferenceModel.WithDefaults().EmbeddingModel)
	}
	return requests
}

func classifierUsesDefaultTextEmbedding(cfg *config.RouterConfig) bool {
	if len(cfg.EmbeddingRules) > 0 || len(cfg.ReaskRules) > 0 ||
		len(cfg.ComplexityRules) > 0 || len(cfg.KnowledgeBases) > 0 {
		return true
	}
	for _, rule := range cfg.JailbreakRules {
		if rule.Method == "contrastive" {
			return true
		}
	}
	return false
}
