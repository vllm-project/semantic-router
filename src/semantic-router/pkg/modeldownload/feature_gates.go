package modeldownload

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type modelFeatureGate struct {
	enabled func(*config.RouterConfig) bool
	paths   func(*config.RouterConfig) []string
}

var optionalModelFeatureGates = []modelFeatureGate{
	{
		enabled: func(cfg *config.RouterConfig) bool {
			return !cfg.EmbeddingModels.UsesRemoteEmbeddingBackend()
		},
		paths: func(cfg *config.RouterConfig) []string {
			// Canonical form, so these match the paths resolveAliasForField collected
			// for the same fields. Gates over ModelID stay literal for the same reason.
			return []string{
				config.ResolveModelPath(cfg.Qwen3ModelPath),
				config.ResolveModelPath(cfg.GemmaModelPath),
				config.ResolveModelPath(cfg.MmBertModelPath),
				config.ResolveModelPath(cfg.MultiModalModelPath),
				config.ResolveModelPath(cfg.BertModelPath),
			}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig) bool {
			return cfg.NeedsCategoryMappingForRouting()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.CategoryModel.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig) bool {
			return cfg.NeedsPIIMappingForRouting()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.PIIModel.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig) bool {
			return cfg.NeedsJailbreakMappingForRouting()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.PromptGuard.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig) bool {
			return cfg.NeedsFactCheckModelForAPI() ||
				cfg.NeedsFactCheckModelForRouting()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.HallucinationMitigation.FactCheckModel.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig) bool {
			return cfg.NeedsLocalHallucinationModelsForRouting() ||
				(cfg.NeedsHallucinationDetectorForDefaultRuntime() &&
					cfg.HallucinationMitigation.HallucinationModel.NormalizedBackend() == config.HallucinationBackendCandle)
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.HallucinationMitigation.HallucinationModel.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig) bool {
			return cfg.NeedsLocalHallucinationNLIForAPI() ||
				cfg.NeedsLocalHallucinationNLIForRouting()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.HallucinationMitigation.NLIModel.ModelID}
		},
	},
	{
		enabled: func(cfg *config.RouterConfig) bool {
			return cfg.NeedsFeedbackModelForAPI() ||
				cfg.NeedsFeedbackModelForRouting()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{cfg.FeedbackDetector.ModelID}
		},
	},
	{
		enabled: isModalityClassifierEnabled,
		paths: func(cfg *config.RouterConfig) []string {
			if cfg.ModalityDetector.Classifier == nil {
				return nil
			}
			return []string{config.ResolveModelPath(cfg.ModalityDetector.Classifier.ModelPath)}
		},
	},
}

func filterDisabledOptionalModelPaths(cfg *config.RouterConfig, paths []string) []string {
	disabled := make(map[string]struct{})
	for _, gate := range optionalModelFeatureGates {
		if gate.enabled(cfg) {
			continue
		}
		for _, path := range gate.paths(cfg) {
			if path != "" {
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
