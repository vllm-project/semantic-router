package classification

import (
	"path/filepath"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// moduleConfigured reports whether a built-in module's signal resolves to a
// deployment in the classifier's scope: a binding or the decision model.
func moduleConfigured(cfg *config.RouterConfig, consumer string) bool {
	return cfg != nil && cfg.ModuleModelConfigured(consumer)
}

// remoteModule reports whether a built-in module's signal resolves to a
// remote classify or chat deployment, which has no local model lifecycle
// and reads its labels from a mapping file.
func remoteModule(cfg *config.RouterConfig, consumer string) bool {
	if cfg == nil {
		return false
	}
	switch cfg.ResolvedProvider(consumer) {
	case config.ClassifyProvider, config.OpenAIChatProvider, config.LegacyHTTPProvider:
		return true
	}
	return false
}

// moduleDeploymentName is the deployment a module's signal resolves to, for
// diagnostics.
func moduleDeploymentName(cfg *config.RouterConfig, consumer string) string {
	if cfg == nil {
		return ""
	}
	resolved, ok, err := cfg.ResolveTask(consumer)
	if !ok || err != nil {
		return ""
	}
	return resolved.Name
}

// deploymentModelDir is the local directory of a deployment's model, where
// its package configuration lives: an absolute package path, or a built-in
// model's local path; empty for any other deployment.
func deploymentModelDir(deployment config.ModelDeployment) string {
	if filepath.IsAbs(deployment.Artifact) {
		return deployment.Artifact
	}
	if spec := config.RegisteredModel(deployment.Artifact); spec != nil {
		return config.ResolveModelPath(spec.LocalPath)
	}
	return ""
}
