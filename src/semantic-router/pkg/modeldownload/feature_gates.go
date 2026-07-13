package modeldownload

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type modelFeatureGate struct {
	enabled func(*config.RouterConfig, embedding.RuntimePlan) bool
	paths   func(*config.RouterConfig) []string
}

var optionalModelFeatureGates = []modelFeatureGate{
	{
		enabled: func(_ *config.RouterConfig, plan embedding.RuntimePlan) bool {
			return plan.Backend != config.EmbeddingBackendOpenAICompatible
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{
				cfg.Qwen3ModelPath,
				cfg.GemmaModelPath,
				cfg.MmBertModelPath,
				cfg.MultiModalModelPath,
				cfg.BertModelPath,
			}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig, _ embedding.RuntimePlan) bool {
			return cfg.NeedsCategoryMappingForRouting()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.CategoryModel.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig, _ embedding.RuntimePlan) bool {
			return cfg.NeedsPIIMappingForRouting()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.PIIModel.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig, _ embedding.RuntimePlan) bool {
			return cfg.NeedsJailbreakMappingForRouting()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.PromptGuard.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig, _ embedding.RuntimePlan) bool {
			return cfg.IsFactCheckClassifierEnabled()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.HallucinationMitigation.FactCheckModel.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig, _ embedding.RuntimePlan) bool {
			return cfg.IsHallucinationModelEnabled()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{
				cfg.HallucinationMitigation.HallucinationModel.ModelID,
				cfg.HallucinationMitigation.NLIModel.ModelID,
			}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig, _ embedding.RuntimePlan) bool {
			return cfg.IsFeedbackDetectorEnabled()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.FeedbackDetector.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig, _ embedding.RuntimePlan) bool {
			return isModalityClassifierEnabled(cfg)
		},
		paths: func(cfg *config.RouterConfig) []string {
			if cfg.ModalityDetector.Classifier == nil {
				return nil
			}
			return []string{cfg.ModalityDetector.Classifier.ModelPath}
		},
	},
}

func filterDisabledOptionalModelPaths(
	cfg *config.RouterConfig,
	paths []string,
	plan embedding.RuntimePlan,
) []string {
	disabled := make(map[string]struct{})
	for _, gate := range optionalModelFeatureGates {
		if gate.enabled(cfg, plan) {
			continue
		}
		for _, path := range gate.paths(cfg) {
			if path != "" {
				disabled[path] = struct{}{}
			}
		}
	}
	if plan.LocalOverride {
		for _, path := range []string{
			cfg.Qwen3ModelPath,
			cfg.GemmaModelPath,
			cfg.MmBertModelPath,
			cfg.MultiModalModelPath,
		} {
			if path != "" && path != plan.ModelPath {
				disabled[path] = struct{}{}
			}
		}
	}

	filtered := make([]string, 0, len(paths))
	for _, path := range paths {
		if _, skip := disabled[path]; skip {
			continue
		}
		filtered = append(filtered, path)
	}
	return filtered
}

func isModalityClassifierEnabled(cfg *config.RouterConfig) bool {
	md := cfg.ModalityDetector
	if !md.Enabled || md.Classifier == nil || md.Classifier.ModelPath == "" {
		return false
	}

	method := md.GetMethod()
	return method == config.ModalityDetectionClassifier || method == config.ModalityDetectionHybrid
}
