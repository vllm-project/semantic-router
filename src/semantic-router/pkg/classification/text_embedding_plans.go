package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// configuredTextEmbeddingRuntimePlans expands a requested Candle plan to the
// complete classifier model union. This must happen before the first native
// initialization because the Candle factory is process-global and immutable.
func configuredTextEmbeddingRuntimePlans(
	cfg *config.RouterConfig,
	requested embedding.RuntimePlan,
) ([]embedding.RuntimePlan, error) {
	plans := []embedding.RuntimePlan{requested}
	if cfg == nil || requested.Backend != config.EmbeddingBackendCandle {
		return plans, nil
	}

	modelRequests := make([]string, 0, 2)
	if hasNonPreferenceTextEmbeddingFamily(cfg) {
		modelRequests = append(modelRequests, cfg.EmbeddingConfig.ModelType)
	}
	if len(cfg.PreferenceRules) > 0 && cfg.PreferenceModel.ContrastiveEnabled() {
		modelRequests = append(modelRequests, cfg.PreferenceModel.WithDefaults().EmbeddingModel)
	}
	resolved, err := embedding.ResolveRuntimePlans(
		cfg.EmbeddingModels,
		requested.Backend,
		embedding.ModelTypeOverrideFromEnv(),
		modelRequests...,
	)
	if err != nil {
		return nil, fmt.Errorf("resolve configured classifier embedding models: %w", err)
	}

	seen := map[embedding.RuntimePlan]struct{}{requested: {}}
	for _, plan := range resolved {
		if _, ok := seen[plan]; ok {
			continue
		}
		seen[plan] = struct{}{}
		plans = append(plans, plan)
	}
	return plans, nil
}

func initializeResolvedTextEmbeddingBackend(
	cfg *config.RouterConfig,
	plans []embedding.RuntimePlan,
) error {
	if cfg == nil {
		return fmt.Errorf("text embedding runtime config is nil")
	}
	if len(plans) == 0 {
		return fmt.Errorf("text embedding runtime has no resolved plans")
	}
	primary := plans[0]
	switch primary.Backend {
	case config.EmbeddingBackendOpenVINO:
		mmBertPath := ""
		fallbackPath := config.ResolveModelPath(primary.ModelPath)
		if primary.ModelType == "mmbert" {
			mmBertPath = fallbackPath
			fallbackPath = ""
		}
		return initializeOpenVINOTextEmbedding(primary.ModelType, mmBertPath, fallbackPath, cfg.UseCPU)
	case config.EmbeddingBackendCandle:
		return initializeCandleRuntimePlans(cfg, plans)
	default:
		return nil
	}
}

func initializeCandleRuntimePlans(cfg *config.RouterConfig, plans []embedding.RuntimePlan) error {
	qwen3Path, gemmaPath, mmBertPath, multiModalPath, err := candleRuntimePlanPaths(plans)
	if err != nil {
		return err
	}

	if qwen3Path != "" || gemmaPath != "" || mmBertPath != "" {
		if err := initializeCandleTextEmbedding(qwen3Path, gemmaPath, mmBertPath, cfg.UseCPU); err != nil {
			return fmt.Errorf("initialize Candle text model union: %w", err)
		}
	}
	if multiModalPath != "" {
		if err := initMultiModalModel(multiModalPath, cfg.UseCPU); err != nil {
			return fmt.Errorf("initialize Candle multimodal model: %w", err)
		}
	}
	return nil
}

func candleRuntimePlanPaths(plans []embedding.RuntimePlan) (string, string, string, string, error) {
	qwen3Path, gemmaPath, mmBertPath, multiModalPath := "", "", "", ""
	for _, plan := range plans {
		if plan.Backend != config.EmbeddingBackendCandle {
			return "", "", "", "", fmt.Errorf("mixed text embedding backends %q and %q", config.EmbeddingBackendCandle, plan.Backend)
		}
		path := config.ResolveModelPath(plan.ModelPath)
		switch plan.ModelType {
		case config.EmbeddingModelTypeQwen3:
			qwen3Path = path
		case "gemma":
			gemmaPath = path
		case "mmbert":
			mmBertPath = path
		case "multimodal":
			multiModalPath = path
		default:
			return "", "", "", "", fmt.Errorf("candle embedding runtime does not support model type %q", plan.ModelType)
		}
	}
	return qwen3Path, gemmaPath, mmBertPath, multiModalPath, nil
}
