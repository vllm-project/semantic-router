package routerruntime

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestRegistryPublishRouterRuntime(t *testing.T) {
	initial := &config.RouterConfig{}
	next := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "next"},
	}

	registry := NewRegistry(initial)
	if registry.CurrentConfig() != initial {
		t.Fatalf("CurrentConfig() = %p, want %p", registry.CurrentConfig(), initial)
	}

	registry.PublishRouterRuntime(next, nil, nil)

	if registry.CurrentConfig() != next {
		t.Fatalf("CurrentConfig() = %p, want %p", registry.CurrentConfig(), next)
	}
}

func TestRegistryPublishesModelSelector(t *testing.T) {
	registry := NewRegistry(&config.RouterConfig{})
	selectorRegistry := selection.NewRegistry()

	registry.SetModelSelector(selectorRegistry)

	if got := registry.ModelSelector(); got != selectorRegistry {
		t.Fatalf("ModelSelector() = %p, want %p", got, selectorRegistry)
	}

	registry.SetModelSelector(nil)
	if got := registry.ModelSelector(); got != nil {
		t.Fatalf("ModelSelector() = %p, want nil after clear", got)
	}
}
