package modelruntime

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// PrepareOwnedResponseCacheNLI gives the shared cache its own typed handle.
// Global module defaults remain supported, but recipe overrides never choose
// this service's artifact or execution. Equal physical specs share the pool.
func PrepareOwnedResponseCacheNLI(ctx context.Context, cfg *config.RouterConfig, runtime *native.Runtime) (*binding.Resolved[tasks.TextPairRequest, tasks.LabelDistribution], error) {
	if cfg == nil || !cfg.NeedsLocalNLIForSemanticCache() {
		return nil, nil
	}
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	spec, ok := plan.LookupGlobal("hallucination_explainer")
	if !ok {
		model := cfg.HallucinationMitigation.NLIModel
		provider, device := config.DefaultModelExecution(model.UseCPU)
		spec = config.ResolvedModelBinding{
			Recipe:     config.GlobalModelScope,
			Binding:    config.ModelBinding{Deployment: "hallucination_explainer", Adapter: "modernbert", Contract: "text_pair_distribution.v1"},
			Deployment: config.ModelDeployment{Artifact: model.ModelID, Provider: provider, Device: device, Precision: "native", Input: config.ModelInputBudget{Overflow: "truncate"}},
			Admission:  cfg.ModelAdmission["hallucination_explainer"],
		}
	}
	spec.Name = "response_cache.hallucination_explainer"
	spec.Deployment.Artifact = config.ResolveModelPath(spec.Deployment.Artifact)
	if runtime == nil {
		runtime = native.New(nil)
	}
	handle, err := runtime.TextPair(ctx, spec)
	if err != nil {
		return nil, fmt.Errorf("prepare global NLI for semantic cache polarity guard: %w", err)
	}
	return handle, nil
}
