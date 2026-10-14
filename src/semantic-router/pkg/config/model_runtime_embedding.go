package config

import "fmt"

// ImplicitEmbeddingBinding resolves the module-default model used by both
// startup planning and embedding preparation.
func (c *RouterConfig) ImplicitEmbeddingBinding(recipe RecipeName, model string) (ResolvedModelBinding, error) {
	var path string
	switch model {
	case "mmbert":
		path = c.MmBertModelPath
	case "qwen3":
		path = c.Qwen3ModelPath
	case "multimodal":
		path = c.MultiModalModelPath
	default:
		return ResolvedModelBinding{}, fmt.Errorf("embedding model %q is not served by the model runtime; use mmbert, qwen3 or multimodal, or an OpenAI-compatible endpoint (vllm-sr config migrate rewrites legacy settings)", model)
	}
	deployment, err := ImplicitModelRuntimeDeployment(path, c.EmbeddingModels.UseCPU)
	if err != nil {
		return ResolvedModelBinding{}, fmt.Errorf("embedding model %s: %w", model, err)
	}
	deployment.Input.Overflow = "truncate"
	return ResolvedModelBinding{
		Recipe: recipe, Name: "embedding",
		Binding:    ModelBinding{Deployment: "@embedding." + model, Contract: "embedding.v1"},
		Deployment: deployment, Admission: c.ModelAdmission["embedding:"+model],
	}, nil
}

func implicitEmbeddingDeploymentsInUse(cfg *RouterConfig, plan *ModelBindingPlan) map[string]ModelDeployment {
	used := make(map[string]ModelDeployment)
	collect := func(scoped *RouterConfig, recipe RecipeName, sharedServices bool) {
		_, declared := plan.Lookup(recipe, "embedding")
		if recipe == GlobalModelScope {
			_, declared = plan.LookupGlobal("embedding")
		}
		if scoped.EmbeddingModels.UsesRemoteEmbeddingBackend() && !declared {
			return
		}
		primary := scoped.primaryEmbeddingModel()
		for model := range EmbeddingModelsNeeded(scoped, primary, sharedServices) {
			if declared && model == primary {
				continue
			}
			if spec, err := scoped.ImplicitEmbeddingBinding(recipe, model); err == nil {
				used[spec.Binding.Deployment] = spec.Deployment
			}
		}
	}
	for recipe, scoped := range cfg.consumerScopes() {
		collect(scoped, recipe, false)
	}
	collect(cfg.ConfigForGlobalModelServices(), GlobalModelScope, true)
	return used
}
