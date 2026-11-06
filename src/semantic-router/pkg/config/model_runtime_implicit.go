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

// ImplicitTaskDeployment resolves the deployment that answers consumer in
// this scope without a binding the configuration authors: the decision
// model, or the deployment a deprecated module path declared. ok is false
// when nothing answers it or an authored binding does; err reports a
// deprecated model that cannot be resolved.
func (c *RouterConfig) ImplicitTaskDeployment(consumer string) (name string, deployment ModelDeployment, ok bool, err error) {
	resolved, ok, err := c.ResolveTask(consumer)
	if IsDeferredModelError(err) {
		return "", ModelDeployment{}, false, err
	}
	if !ok || err != nil || (resolved.Explicit() && !resolved.Deployment.Translated()) {
		return "", ModelDeployment{}, false, nil
	}
	return resolved.Name, resolved.Deployment, true, nil
}

// ImplicitModelRuntimeDeployment resolves a module's model reference to the
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

// MissingLocalModel reports, for a deployment a deprecated model path
// declared, a local package directory that does not exist (yet): the
// consumer that runs it fails, as such a model did before, and the runtime
// does not start it.
func (d ModelDeployment) MissingLocalModel() error {
	if !d.translated || !d.IsModelRuntime() || d.legacyPath == "" || GetModelByPath(d.legacyPath) != nil {
		return nil
	}
	if _, err := os.Stat(d.Artifact); err != nil {
		return fmt.Errorf("model %q is neither a built-in model nor a local package directory; declare a model_runtime deployment with its Hub repository and revision", d.legacyPath)
	}
	return nil
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

// resolvedTaskDeploymentsInUse lists the model_runtime deployments that the
// active consumers of every scope resolve to. A consumer whose deprecated
// model cannot be resolved contributes nothing here; preparation reports its
// error.
func resolvedTaskDeploymentsInUse(cfg *RouterConfig) map[string]ModelDeployment {
	used := make(map[string]ModelDeployment)
	for recipe, scoped := range cfg.consumerScopes() {
		for _, consumer := range scoped.scopeConsumers() {
			if consumer == "embedding" || !TaskConsumerInUse(scoped, recipe, consumer) {
				continue
			}
			resolved, ok, err := scoped.resolveTask(consumer)
			if ok && err == nil && resolved.Deployment.IsModelRuntime() && resolved.Deployment.MissingLocalModel() == nil {
				used[resolved.Name] = resolved.Deployment.WithDefaults()
			}
		}
	}
	return used
}

// RuntimeServedModelPaths lists the resolved local paths of the built-in
// models that the scope's consumers run on model_runtime deployments: the
// runtime downloads them at their pinned revisions, so the router does not.
func (c *RouterConfig) RuntimeServedModelPaths() map[string]bool {
	resolved := c.withModelResources()
	paths := make(map[string]bool)
	for _, consumer := range resolved.scopeConsumers() {
		task, ok, err := resolved.resolveTask(consumer)
		if !ok || err != nil || !task.Deployment.IsModelRuntime() {
			continue
		}
		if spec := registeredModel(task.Deployment.Artifact); spec != nil && servedBuiltIn(spec) {
			paths[ResolveModelPath(spec.LocalPath)] = true
		}
	}
	return paths
}

// implicitConsumers lists the task consumers a scope's modules and rules may run.
func (c *RouterConfig) implicitConsumers() []string {
	return []string{"prompt_guard", "pii_classifier", "hallucination_detector"}
}
