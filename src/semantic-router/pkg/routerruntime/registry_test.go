package routerruntime

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
