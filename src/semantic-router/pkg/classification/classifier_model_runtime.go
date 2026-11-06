package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// classifierModelRuntime is preparation-only state. Requests use typed handles;
// they never resolve catalog paths or scan another recipe's configuration.
type classifierModelRuntime struct {
	runtime *serving.Runtime
	// systemOne asks the systemone deployments' remote APIs.
	systemOne modelservice.Remote
	plan      *config.ModelBindingPlan
	cfg       *config.RouterConfig
	recipe    config.RecipeName
}

func newClassifierModelRuntime(cfg *config.RouterConfig, options RecipeRuntimeOptions) (*classifierModelRuntime, error) {
	cfg = config.WithModelResources(cfg)
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	runtime := options.Runtime
	if runtime == nil {
		runtime = serving.New(nil, nil)
	}
	recipe := cfg.RoutingScope
	if recipe == "" {
		recipe = config.DefaultRecipeName
	}
	return &classifierModelRuntime{runtime: runtime, systemOne: options.SystemOne, plan: plan, cfg: cfg, recipe: recipe}, nil
}

// decider answers systemone signals through the generation's deployments.
func (m *classifierModelRuntime) decider() (modelservice.Decider, bool) {
	if m == nil || m.runtime == nil {
		return nil, false
	}
	decider, ok := m.runtime.Services().(modelservice.Decider)
	return decider, ok
}

func standaloneModelRuntime() *classifierModelRuntime {
	return &classifierModelRuntime{runtime: serving.New(nil, nil), cfg: &config.RouterConfig{}, recipe: config.DefaultRecipeName}
}
