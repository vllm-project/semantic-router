package routerruntime

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
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

func TestRegistryRejectsPartialClassifierRefresh(t *testing.T) {
	initial := &config.RouterConfig{}
	service := services.NewClassificationService(
		&classification.Classifier{Config: initial},
		initial,
	)
	registry := NewRegistry(initial)
	registry.PublishRouterRuntime(initial, service, nil)

	registry.RefreshRuntimeConfig(nil)

	if registry.CurrentConfig() != initial {
		t.Fatal("registry published config after classifier rebuild failure")
	}
	if service.GetConfig() != initial {
		t.Fatal("classification service published partial config")
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
