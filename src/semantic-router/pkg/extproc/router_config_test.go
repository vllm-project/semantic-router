package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestRouterConfigUsesRuntimeRegistryBeforeGlobal(t *testing.T) {
	globalCfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "global-decision"}},
		},
	}
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(globalCfg)
	defer restoreGlobalConfig()

	runtimeCfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "runtime-decision"}},
		},
	}
	router := &OpenAIRouter{
		RuntimeRegistry: routerruntime.NewRegistry(runtimeCfg),
	}

	if got := router.routerConfig(); got != runtimeCfg {
		t.Fatalf("routerConfig() = %p, want runtime config %p", got, runtimeCfg)
	}
	if decision := router.decisionByName("runtime-decision"); decision == nil {
		t.Fatal("decisionByName() did not resolve runtime-owned decision")
	}
	if decision := router.decisionByName("global-decision"); decision != nil {
		t.Fatalf("decisionByName() resolved global decision with runtime registry: %+v", decision)
	}
}

func TestRouterConfigWithEmptyRuntimeRegistryDoesNotUseGlobal(t *testing.T) {
	globalCfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "global-decision"}},
		},
	}
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(globalCfg)
	defer restoreGlobalConfig()

	router := &OpenAIRouter{
		RuntimeRegistry: routerruntime.NewRegistry(nil),
	}

	if got := router.routerConfig(); got != nil {
		t.Fatalf("routerConfig() = %p, want nil for empty runtime registry", got)
	}
	if decision := router.decisionByName("global-decision"); decision != nil {
		t.Fatalf("decisionByName() resolved global decision with empty runtime registry: %+v", decision)
	}
}
