package modeldownload

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type modelFeatureGate struct {
	enabled func(*config.RouterConfig) bool
	paths   func(*config.RouterConfig) []string
}

var optionalModelFeatureGates = []modelFeatureGate{
	{
		enabled: func(cfg *config.RouterConfig) bool {
			return !cfg.UsesRemoteEmbeddings()
		},
		paths: func(cfg *config.RouterConfig) []string {
			return []string{
				cfg.Qwen3ModelPath,
				cfg.MmBertModelPath,
				cfg.MultiModalModelPath,
			}
		},
	},
}

func filterDisabledOptionalModelPaths(cfg *config.RouterConfig, paths []string) []string {
	disabled := make(map[string]struct{})
	enabled := make(map[string]bool)
	for _, gate := range optionalModelFeatureGates {
		if gate.enabled(cfg) {
			for _, path := range gate.paths(cfg) {
				enabled[config.ResolveModelPath(path)] = true
			}
		}
	}
	for _, gate := range optionalModelFeatureGates {
		if gate.enabled(cfg) {
			continue
		}
		for _, path := range gate.paths(cfg) {
			path = config.ResolveModelPath(path)
			if path != "" && !enabled[path] {
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
