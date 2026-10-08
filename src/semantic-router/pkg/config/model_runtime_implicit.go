package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ImplicitDeploymentPrefix starts the names of the model_runtime deployments
// that module defaults resolve to when no binding names a deployment.
// Configured deployment names cannot start with it.
const ImplicitDeploymentPrefix = "@"

// moduleModel is the local model a module or rule names for one consumer.
type moduleModel struct {
	module string
	model  string
	useCPU bool
}

// implicitModule returns the module model a consumer runs when no binding
// declares one. Module defaults are global, so consumers of one module share
// a deployment across recipes; a generic classifier rule is recipe-local.
func (c *RouterConfig) implicitModule(consumer string) (moduleModel, bool) {
	local := func(module, model string, useCPU bool) (moduleModel, bool) {
		return moduleModel{module: module, model: strings.TrimSpace(model), useCPU: useCPU}, strings.TrimSpace(model) != ""
	}
	switch consumer {
	case "preference", "reask", "complexity":
		return local(consumer, c.DecisionModelSpec().Model, true)
	case "domain_classifier":
		if c.CategoryModel.Backend == nil {
			return local(consumer, c.CategoryModel.ModelID, c.CategoryModel.UseCPU)
		}
	case "prompt_guard":
		if c.PromptGuard.Backend == nil {
			return local(consumer, c.PromptGuard.ModelID, c.PromptGuard.UseCPU)
		}
	case "pii_classifier":
		if c.PIIModel.Backend == nil {
			return local(consumer, c.PIIModel.ModelID, c.PIIModel.UseCPU)
		}
	case "fact_check_classifier":
		return local(consumer, c.HallucinationMitigation.FactCheckModel.ModelID, c.HallucinationMitigation.FactCheckModel.UseCPU)
	case "feedback_detector":
		return local(consumer, c.FeedbackDetector.ModelID, c.FeedbackDetector.UseCPU)
	case "modality_detector":
		if model, useCPU, ok := c.ModalityClassifierModel(); ok {
			return local(consumer, model, useCPU)
		}
	case "hallucination_detector":
		if c.HallucinationMitigation.HallucinationModel.NormalizedBackend() == HallucinationBackendLocal {
			return local(consumer, c.HallucinationMitigation.HallucinationModel.ModelID, c.HallucinationMitigation.HallucinationModel.UseCPU)
		}
	}
	switch {
	case strings.HasPrefix(consumer, "safety.") && strings.HasSuffix(consumer, ".hazard"):
		return local("hazard", c.SafetyModels.Hazard.ModelID, c.SafetyModels.Hazard.UseCPU)
	case strings.HasPrefix(consumer, "safety."):
		return local("safety", c.SafetyModels.Safety.ModelID, c.SafetyModels.Safety.UseCPU)
	case strings.HasPrefix(consumer, "classifier."):
		if rule := classifierSignalRuleByName(c.ClassifierRules, strings.TrimPrefix(consumer, "classifier.")); rule != nil && rule.Model == "" {
			if rule.ModelPath == "" {
				return local(consumer, c.DecisionModelSpec().Model, true)
			}
			return local(string(c.recipeScope())+"/"+consumer, rule.ModelPath, rule.UseCPU)
		}
	}
	return moduleModel{}, false
}

func (c *RouterConfig) recipeScope() RecipeName {
	if c.RoutingScope != "" {
		return c.RoutingScope
	}
	return DefaultRecipeName
}

// ImplicitTaskDeployment resolves a consumer that no binding declares to the
// model_runtime deployment of its module's model: the deployment's name and
// its definition. ok is false when the module names no local model. Modules
// that name one shared built-in model (Vela 2.0) on one device share its
// deployment, named after the model.
func (c *RouterConfig) ImplicitTaskDeployment(consumer string) (name string, deployment ModelDeployment, ok bool, err error) {
	module, ok := c.implicitModule(consumer)
	if !ok {
		return "", ModelDeployment{}, false, nil
	}
	// A default task uses the declared resource, including its device/profile.
	// Sharing is by resource identity, not by a newly manufactured module alias.
	if selected, resource, found, resolveErr := c.DecisionModelDeployment(); found && resolveErr == nil && module.model == c.DecisionModelSpec().Model {
		return selected, resource, true, nil
	}
	deployment, err = ImplicitModelRuntimeDeployment(module.model, module.useCPU)
	if spec := GetModelByPath(module.model); spec != nil && spec.SharedDeployment {
		return sharedDeploymentName(spec, deployment.Device), deployment, true, err
	}
	return ImplicitDeploymentPrefix + module.module, deployment, true, err
}

// sharedDeploymentName names the implicit deployment of a shared built-in
// model on a device: "@Vela-2.0-0.3B" on CPU, "@Vela-2.0-0.3B/auto" elsewhere.
func sharedDeploymentName(spec *ModelSpec, device string) string {
	name := ImplicitDeploymentPrefix + strings.TrimPrefix(spec.LocalPath, "models/")
	if device != "cpu" {
		name += "/" + device
	}
	return name
}

// ImplicitModelRuntimeDeployment resolves a module's model reference to the
// model_runtime deployment that serves it: a built-in model (a registry path
// or alias) at its pinned revision, or a local package directory. It runs on
// CPU when useCPU, else on the best available device; a built-in model runs
// its registered CPU profile on CPU, every other deployment exact. A model
// that requires a GPU runs on the best available device whatever useCPU says;
// the model runtime manager refuses it on a host without a GPU.
func ImplicitModelRuntimeDeployment(model string, useCPU bool) (ModelDeployment, error) {
	deployment := ModelDeployment{Provider: ModelRuntimeProvider, Device: "auto", Profile: "exact"}
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
		if useCPU && spec.CPUProfile != "" {
			deployment.Profile = spec.CPUProfile
		}
		return deployment, nil
	}
	if reference == "" {
		return ModelDeployment{}, fmt.Errorf("no model is configured")
	}
	path, err := filepath.Abs(reference)
	if err == nil {
		_, err = os.Stat(path)
	}
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
		spec.RepoID == "Qwen/Qwen3-Embedding-0.6B"
}

// implicitTaskDeploymentsInUse lists the implicit deployments of the active
// consumers that no binding declares. A module whose model cannot be resolved
// contributes nothing here; preparation reports its error.
func implicitTaskDeploymentsInUse(cfg *RouterConfig, plan *ModelBindingPlan) map[string]ModelDeployment {
	used := make(map[string]ModelDeployment)
	for recipe, scoped := range cfg.consumerScopes() {
		for _, consumer := range scoped.implicitConsumers() {
			if _, declared := plan.Lookup(recipe, consumer); declared || !TaskConsumerInUse(scoped, recipe, consumer) {
				continue
			}
			name, deployment, ok, err := scoped.ImplicitTaskDeployment(consumer)
			if ok && err == nil {
				used[name] = deployment.WithDefaults()
			}
		}
	}
	return used
}

// RuntimeServedModelPaths lists the resolved local paths of the built-in
// models that the scope's module consumers run through implicit model_runtime
// deployments: the runtime downloads them at their pinned revisions, so the
// router does not. Any other module model is a local package directory.
func (c *RouterConfig) RuntimeServedModelPaths() map[string]bool {
	paths := make(map[string]bool)
	for _, consumer := range c.implicitConsumers() {
		module, ok := c.implicitModule(consumer)
		if !ok {
			continue
		}
		if spec := GetModelByPath(strings.TrimSpace(module.model)); spec != nil && servedBuiltIn(spec) {
			paths[ResolveModelPath(module.model)] = true
		}
	}
	return paths
}

// implicitConsumers lists the task consumers a scope's modules and rules may run.
func (c *RouterConfig) implicitConsumers() []string {
	consumers := []string{"domain_classifier", "prompt_guard", "pii_classifier", "fact_check_classifier", "feedback_detector", "modality_detector", "hallucination_detector", "preference", "reask", "complexity"}
	for _, rule := range c.ClassifierRules {
		consumers = append(consumers, "classifier."+rule.Name)
	}
	for _, rule := range c.SafetyRules {
		if rule.Model == "" {
			consumers = append(consumers, "safety."+rule.Name)
		}
		if rule.Hazard != nil && rule.Hazard.Model == "" {
			consumers = append(consumers, "safety."+rule.Name+".hazard")
		}
	}
	return consumers
}
