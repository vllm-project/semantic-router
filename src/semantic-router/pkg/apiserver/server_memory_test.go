//go:build !windows && cgo

package apiserver

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestShouldInitMemoryStore(t *testing.T) {
	tests := []struct {
		name string
		cfg  *config.RouterConfig
		want bool
	}{
		{
			name: "nil config",
			cfg:  nil,
			want: false,
		},
		{
			name: "global memory enabled",
			cfg:  &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true}},
			want: true,
		},
		{
			name: "memory plugin present",
			cfg: &config.RouterConfig{
				Memory: config.MemoryConfig{Enabled: false},
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{memoryPluginDecision("with-memory-plugin", "memory")},
				},
			},
			want: true,
		},
		{
			name: "memory disabled and no plugin",
			cfg: &config.RouterConfig{
				Memory: config.MemoryConfig{Enabled: false},
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{memoryPluginDecision("no-memory-plugin", "pii")},
				},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldInitMemoryStore(tt.cfg)
			if got != tt.want {
				t.Fatalf("shouldInitMemoryStore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestResolveMemoryStoreUsesRuntimeRegistryOnly(t *testing.T) {
	cfg := &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true}}
	globalStore := newMockMemoryStore()
	runtimeStore := newMockMemoryStore()

	memory.SetGlobalMemoryStore(globalStore)
	t.Cleanup(func() {
		memory.SetGlobalMemoryStore(nil)
	})

	registry := routerruntime.NewRegistry(cfg)
	if got := resolveMemoryStore(cfg, registry); got != nil {
		t.Fatalf("resolveMemoryStore() = %v, want nil before runtime store is published", got)
	}

	registry.SetMemoryStore(runtimeStore)
	if got := resolveMemoryStore(cfg, registry); got != runtimeStore {
		t.Fatalf("resolveMemoryStore() = %v, want runtime store %v", got, runtimeStore)
	}
}

func TestResolveAPIServerConfigUsesRuntimeRegistryOnly(t *testing.T) {
	globalCfg := &config.RouterConfig{}
	runtimeCfg := &config.RouterConfig{}

	restoreGlobalConfig := replaceGlobalConfigForTest(globalCfg)
	t.Cleanup(restoreGlobalConfig)

	registry := routerruntime.NewRegistry(nil)
	if got := resolveAPIServerConfig(registry); got != nil {
		t.Fatalf("resolveAPIServerConfig() = %v, want nil before runtime config is published", got)
	}

	registry.UpdateConfig(runtimeCfg)
	if got := resolveAPIServerConfig(registry); got != runtimeCfg {
		t.Fatalf("resolveAPIServerConfig() = %v, want runtime config %v", got, runtimeCfg)
	}
}

func TestResolveAPIServerConfigUsesFileProcessConfig(t *testing.T) {
	globalCfg := &config.RouterConfig{}

	restoreGlobalConfig := replaceGlobalConfigForTest(globalCfg)
	t.Cleanup(restoreGlobalConfig)

	if got := resolveAPIServerConfig(nil); got != globalCfg {
		t.Fatalf("resolveAPIServerConfig() = %v, want legacy global config %v", got, globalCfg)
	}
}

func TestResolveClassificationServiceUsesRuntimeRegistryOnly(t *testing.T) {
	cfg := &config.RouterConfig{}
	runtimeSvc := services.NewClassificationService(nil, cfg)

	registry := routerruntime.NewRegistry(cfg)
	if got := resolveClassificationService(cfg, registry); got != nil {
		t.Fatalf("resolveClassificationService() = %v, want nil before runtime service is published", got)
	}

	registry.SetClassificationService(runtimeSvc)
	if got := resolveClassificationService(cfg, registry); got != runtimeSvc {
		t.Fatalf("resolveClassificationService() = %v, want runtime service %v", got, runtimeSvc)
	}
}

func TestResolveClassificationServiceBuildsCanonicalFileComposition(t *testing.T) {
	cfg := &config.RouterConfig{Recipes: []config.RoutingRecipe{{Name: config.DefaultRecipeName}}}

	got := resolveClassificationService(cfg, nil)
	if got == nil {
		t.Fatal("resolveClassificationService() returned nil for canonical file config")
	}
	if got.GetConfig() != cfg {
		t.Fatalf("resolveClassificationService() config = %p, want %p", got.GetConfig(), cfg)
	}
}

func TestEnsureClassificationServiceWaitsForRuntimeRegistry(t *testing.T) {
	cfg := &config.RouterConfig{}
	registry := routerruntime.NewRegistry(cfg)

	svc := ensureClassificationService(registry, nil)
	if svc == nil {
		t.Fatal("ensureClassificationService() returned nil, want placeholder service")
	}
	if svc.HasClassifier() {
		t.Fatal("placeholder service unexpectedly has a classifier before runtime registry publication")
	}
	if got := registry.ClassificationService(); got != nil {
		t.Fatalf("runtime registry classification service = %v, want nil before router runtime publication", got)
	}
}

func TestResolveMemoryStorePreservesLegacyGlobalFallback(t *testing.T) {
	cfg := &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true}}
	globalStore := newMockMemoryStore()

	memory.SetGlobalMemoryStore(globalStore)
	t.Cleanup(func() {
		memory.SetGlobalMemoryStore(nil)
	})

	if got := resolveMemoryStore(cfg, nil); got != globalStore {
		t.Fatalf("resolveMemoryStore() = %v, want legacy global store %v", got, globalStore)
	}
}

func replaceGlobalConfigForTest(newCfg *config.RouterConfig) func() {
	previous := config.Get()
	config.Replace(newCfg)
	return func() {
		if previous != nil {
			config.Replace(previous)
			return
		}
		config.Replace(&config.RouterConfig{})
	}
}

func memoryPluginDecision(name, pluginType string) config.Decision {
	return config.Decision{
		Name: name,
		Plugins: []config.DecisionPlugin{
			{
				Type: pluginType,
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled": true,
				}),
			},
		},
	}
}
