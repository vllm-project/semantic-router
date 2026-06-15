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
	globalCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceFile}
	runtimeCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}

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

func TestResolveAPIServerConfigPreservesLegacyGlobalFallback(t *testing.T) {
	globalCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceFile}

	restoreGlobalConfig := replaceGlobalConfigForTest(globalCfg)
	t.Cleanup(restoreGlobalConfig)

	if got := resolveAPIServerConfig(nil); got != globalCfg {
		t.Fatalf("resolveAPIServerConfig() = %v, want legacy global config %v", got, globalCfg)
	}
}

func TestResolveClassificationServiceUsesRuntimeRegistryOnly(t *testing.T) {
	cfg := &config.RouterConfig{}
	globalSvc := services.NewPlaceholderClassificationService()
	runtimeSvc := services.NewClassificationService(nil, cfg)

	services.SetGlobalClassificationService(globalSvc)
	t.Cleanup(func() {
		services.SetGlobalClassificationService(nil)
	})

	registry := routerruntime.NewRegistry(cfg)
	if got := resolveClassificationService(cfg, registry); got != nil {
		t.Fatalf("resolveClassificationService() = %v, want nil before runtime service is published", got)
	}

	registry.SetClassificationService(runtimeSvc)
	if got := resolveClassificationService(cfg, registry); got != runtimeSvc {
		t.Fatalf("resolveClassificationService() = %v, want runtime service %v", got, runtimeSvc)
	}
}

func TestResolveClassificationServicePreservesLegacyGlobalFallback(t *testing.T) {
	cfg := &config.RouterConfig{}
	globalSvc := services.NewPlaceholderClassificationService()

	services.SetGlobalClassificationService(globalSvc)
	t.Cleanup(func() {
		services.SetGlobalClassificationService(nil)
	})

	if got := resolveClassificationService(cfg, nil); got != globalSvc {
		t.Fatalf("resolveClassificationService() = %v, want legacy global service %v", got, globalSvc)
	}
}

func TestEnsureClassificationServiceWaitsForRuntimeRegistry(t *testing.T) {
	cfg := &config.RouterConfig{}
	registry := routerruntime.NewRegistry(cfg)

	svc := ensureClassificationService(cfg, registry, nil)
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

func TestBuildConfigUpdaterPreservesLegacyGlobalFallback(t *testing.T) {
	globalCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceFile}
	nextCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	classificationSvc := &fakeResolvedClassificationService{}

	restoreGlobalConfig := replaceGlobalConfigForTest(globalCfg)
	t.Cleanup(restoreGlobalConfig)

	updater := buildConfigUpdater(nil, classificationSvc)
	updater(nextCfg)

	if classificationSvc.updatedConfig != nextCfg {
		t.Fatalf("classification service updated config = %p, want %p", classificationSvc.updatedConfig, nextCfg)
	}
	if got := config.Get(); got != nextCfg {
		t.Fatalf("config.Get() = %p, want legacy global config update %p", got, nextCfg)
	}
}

func TestBuildConfigUpdaterUsesRuntimeRegistryWithoutReplacingGlobalConfig(t *testing.T) {
	globalCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceFile}
	initialRuntimeCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceFile}
	nextCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	classificationSvc := services.NewClassificationService(nil, initialRuntimeCfg)
	registry := routerruntime.NewRegistry(initialRuntimeCfg)
	registry.SetClassificationService(classificationSvc)

	restoreGlobalConfig := replaceGlobalConfigForTest(globalCfg)
	t.Cleanup(restoreGlobalConfig)

	updater := buildConfigUpdater(registry, nil)
	updater(nextCfg)

	if got := registry.CurrentConfig(); got != nextCfg {
		t.Fatalf("registry.CurrentConfig() = %p, want %p", got, nextCfg)
	}
	if got := classificationSvc.GetConfig(); got != nextCfg {
		t.Fatalf("classificationSvc.GetConfig() = %p, want %p", got, nextCfg)
	}
	if got := config.Get(); got != globalCfg {
		t.Fatalf("config.Get() = %p, want unchanged global config %p", got, globalCfg)
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
