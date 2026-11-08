package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ImplicitDeploymentPrefix starts names the Router reserves; configured
// deployment names cannot start with it.
const ImplicitDeploymentPrefix = "@"

// ImplicitModelRuntimeDeployment resolves a model reference to the
// model_runtime deployment that serves it: a built-in model (a registry path
// or alias) at its pinned revision, or a local package directory. It runs on
// CPU when useCPU, else on the best available device, with the runtime's
// default engine arguments. A model that requires a GPU runs on the best
// available device whatever useCPU says; the model runtime manager refuses it
// on a host without a GPU.
func ImplicitModelRuntimeDeployment(model string, useCPU bool) (ModelDeployment, error) {
	deployment, err := modelRuntimeDeploymentOf(model, useCPU)
	if err != nil || GetModelByPath(strings.TrimSpace(model)) != nil {
		return deployment, err
	}
	if _, err := os.Stat(deployment.Artifact); err != nil {
		return ModelDeployment{}, fmt.Errorf("model %q is neither a built-in model nor a local package directory; declare a model_runtime deployment with its Hub repository and revision", strings.TrimSpace(model))
	}
	return deployment, nil
}

// modelRuntimeDeploymentOf is ImplicitModelRuntimeDeployment without
// checking that a local package directory exists yet: the Router downloads
// a registered one before the runtime serves it.
func modelRuntimeDeploymentOf(model string, useCPU bool) (ModelDeployment, error) {
	deployment := ModelDeployment{Provider: ModelRuntimeProvider, Device: "auto"}
	reference := strings.TrimSpace(model)
	spec := GetModelByPath(reference)
	if spec != nil && spec.RequiresGPU {
		useCPU = false
	}
	if useCPU {
		deployment.Device = "cpu"
	}
	if spec != nil {
		if !servedBuiltIn(spec) {
			return ModelDeployment{}, fmt.Errorf("model %q has no model_runtime family; run `vllm-sr config migrate` to move to its Vela 1.0 replacement", reference)
		}
		deployment.Artifact, deployment.Revision = spec.RepoID, spec.Revision
		return deployment, nil
	}
	if reference == "" {
		return ModelDeployment{}, fmt.Errorf("no model is configured")
	}
	path, err := filepath.Abs(reference)
	if err != nil {
		return ModelDeployment{}, fmt.Errorf("model %q is neither a built-in model nor a local package directory; declare a model_runtime deployment with its Hub repository and revision", reference)
	}
	deployment.Artifact = path
	return deployment, nil
}

// servedBuiltIn reports whether the runtime's built-in table serves a
// registry model; earlier aliases have Vela 1.0 replacements.
func servedBuiltIn(spec *ModelSpec) bool {
	return strings.HasPrefix(spec.RepoID, "vllm-sr/Vela-1.0-") || strings.HasPrefix(spec.RepoID, "vllm-sr/Vela-2.0-") ||
		strings.HasPrefix(spec.RepoID, "vllm-sr/vela3-") ||
		spec.RepoID == "Qwen/Qwen3-Embedding-0.6B"
}

// RuntimeServedModelPaths lists the resolved local paths of the built-in
// models that the scope's consumers, and the embedding consumers, run on
// model_runtime deployments: the runtime downloads them at their pinned
// revisions, so the router does not.
func (c *RouterConfig) RuntimeServedModelPaths() map[string]bool {
	paths := make(map[string]bool)
	served := func(deployment ModelDeployment) {
		if spec := registeredModel(deployment.Artifact); deployment.IsModelRuntime() && spec != nil && servedBuiltIn(spec) {
			paths[ResolveModelPath(spec.LocalPath)] = true
		}
	}
	for _, consumer := range c.scopeConsumers() {
		if task, ok, err := c.resolveTask(consumer); ok && err == nil {
			served(task.Deployment)
		}
	}
	for name := range c.embeddingDeploymentsInUse() {
		served(c.ModelDeployments[name])
	}
	return paths
}

// implicitConsumers lists the application signal types whose model the
// Router prepares for a scope: the guards and the span signals.
func (c *RouterConfig) implicitConsumers() []string {
	return []string{SignalTypeJailbreak, SignalTypePII, SignalTypeHallucination}
}
