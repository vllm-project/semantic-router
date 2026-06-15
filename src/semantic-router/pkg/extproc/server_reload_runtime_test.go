package extproc

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type reloadMemoryStore struct{}

func (reloadMemoryStore) Store(_ context.Context, _ *memory.Memory) error { return nil }
func (reloadMemoryStore) Retrieve(_ context.Context, _ memory.RetrieveOptions) ([]*memory.RetrieveResult, error) {
	return nil, nil
}
func (reloadMemoryStore) Get(_ context.Context, _ string) (*memory.Memory, error)    { return nil, nil }
func (reloadMemoryStore) Update(_ context.Context, _ string, _ *memory.Memory) error { return nil }
func (reloadMemoryStore) List(_ context.Context, _ memory.ListOptions) (*memory.ListResult, error) {
	return nil, nil
}
func (reloadMemoryStore) Forget(_ context.Context, _ string) error                    { return nil }
func (reloadMemoryStore) ForgetByScope(_ context.Context, _ memory.MemoryScope) error { return nil }
func (reloadMemoryStore) IsEnabled() bool                                             { return true }
func (reloadMemoryStore) CheckConnection(_ context.Context) error                     { return nil }
func (reloadMemoryStore) Close() error                                                { return nil }

func TestReloadRouterFromConfigSkipsReplaceForKubernetesSource(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()

	candidateCfg := &config.RouterConfig{
		ConfigSource:  config.ConfigSourceKubernetes,
		BackendModels: config.BackendModels{DefaultModel: "new-model"},
	}
	oldRouter := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "old-model"},
	}}
	server := &Server{
		configPath: "/unused/config.yaml",
		service:    NewRouterService(oldRouter),
	}

	ensureReloadConfigModels = func(cfg *config.RouterConfig) error {
		t.Fatalf("ensureReloadConfigModels() should not run during kubernetes watcher reload")
		return nil
	}
	prepareReloadRuntime = func(cfg *config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		if cfg != candidateCfg {
			t.Fatalf("prepareReloadRuntime() cfg = %p, want %p", cfg, candidateCfg)
		}
		return modelruntime.EmbeddingRuntimeState{AnyReady: true}, nil
	}

	buildCalls := 0
	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		buildCalls++
		if cfg != candidateCfg {
			t.Fatalf("buildReloadRouter() cfg = %p, want %p", cfg, candidateCfg)
		}
		return &OpenAIRouter{Config: cfg}, nil
	}
	warmupCalls := 0
	warmupReloadRouter = func(router *OpenAIRouter, state modelruntime.EmbeddingRuntimeState) error {
		warmupCalls++
		if router == nil || router.Config != candidateCfg {
			t.Fatalf("warmupReloadRouter() router config mismatch")
		}
		if !state.AnyReady {
			t.Fatalf("warmupReloadRouter() state = %+v, want AnyReady=true", state)
		}
		return nil
	}

	replaceCalls := 0
	replaceReloadConfig = func(cfg *config.RouterConfig) {
		replaceCalls++
	}

	if err := server.reloadRouterFromConfig("kubernetes", server.configPath, candidateCfg); err != nil {
		t.Fatalf("reloadRouterFromConfig() error = %v", err)
	}

	if buildCalls != 1 {
		t.Fatalf("buildReloadRouter() calls = %d, want 1", buildCalls)
	}
	if warmupCalls != 1 {
		t.Fatalf("warmupReloadRouter() calls = %d, want 1", warmupCalls)
	}
	if replaceCalls != 0 {
		t.Fatalf("replaceReloadConfig() calls = %d, want 0", replaceCalls)
	}
	if got := server.service.GetRouter(); got == oldRouter || got.Config != candidateCfg {
		t.Fatalf("router swap did not install kubernetes config")
	}
}

func TestReloadRouterFromConfigDoesNotSwapWhenRuntimePreparationFails(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()

	candidateCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "candidate"},
	}
	oldRouter := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "old"},
	}}
	server := &Server{
		configPath: "/tmp/router-config.yaml",
		service:    NewRouterService(oldRouter),
	}

	ensureReloadConfigModels = func(cfg *config.RouterConfig) error { return nil }
	prepareReloadRuntime = func(cfg *config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		return modelruntime.EmbeddingRuntimeState{}, errors.New("modality init failed")
	}
	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		t.Fatalf("buildReloadRouter() should not be called when runtime prep fails")
		return nil, nil
	}
	warmupReloadRouter = func(router *OpenAIRouter, state modelruntime.EmbeddingRuntimeState) error {
		t.Fatalf("warmupReloadRouter() should not be called when runtime prep fails")
		return nil
	}
	replaceReloadConfig = func(cfg *config.RouterConfig) {
		t.Fatalf("replaceReloadConfig() should not be called when runtime prep fails")
	}

	err := server.reloadRouterFromConfig("file", server.configPath, candidateCfg)
	if err == nil {
		t.Fatal("reloadRouterFromConfig() error = nil, want failure")
	}
	if got := err.Error(); got != "runtime dependency init failed: modality init failed" {
		t.Fatalf("reloadRouterFromConfig() error = %q", got)
	}
	if got := server.service.GetRouter(); got != oldRouter {
		t.Fatalf("router changed on runtime prep failure")
	}
}

func TestReloadRouterFromConfigPublishesRuntimeRegistryAfterSwap(t *testing.T) {
	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()

	globalCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "global"},
	}
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(globalCfg)
	defer restoreGlobalConfig()

	oldCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "old"},
	}
	newCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "new"},
	}
	oldService := &services.ClassificationService{}
	newService := &services.ClassificationService{}
	newModelSelector := selection.NewRegistry()
	registry := routerruntime.NewRegistry(oldCfg)
	registry.PublishRouterRuntime(oldCfg, oldService, nil)

	server := &Server{
		configPath: "/tmp/router-config.yaml",
		service: NewRouterService(&OpenAIRouter{
			Config:                oldCfg,
			ClassificationService: oldService,
		}),
		runtime: registry,
	}

	ensureReloadConfigModels = func(cfg *config.RouterConfig) error { return nil }
	prepareReloadRuntime = func(cfg *config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		return modelruntime.EmbeddingRuntimeState{AnyReady: true, ToolsReady: true}, nil
	}
	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		return &OpenAIRouter{
			Config:                newCfg,
			ClassificationService: newService,
			MemoryStore:           reloadMemoryStore{},
			ModelSelector:         newModelSelector,
		}, nil
	}
	warmupReloadRouter = func(router *OpenAIRouter, state modelruntime.EmbeddingRuntimeState) error { return nil }
	replaceReloadConfig = func(cfg *config.RouterConfig) {
		t.Fatalf("replaceReloadConfig() should not run for registry-backed reload")
	}

	if err := server.reloadRouterFromConfig("file", server.configPath, newCfg); err != nil {
		t.Fatalf("reloadRouterFromConfig() error = %v", err)
	}

	if got := registry.CurrentConfig(); got != newCfg {
		t.Fatalf("registry.CurrentConfig() = %p, want %p", got, newCfg)
	}
	if got := config.Get(); got != globalCfg {
		t.Fatalf("config.Get() = %p, want unchanged global cfg %p", got, globalCfg)
	}
	if got := registry.ClassificationService(); got != newService {
		t.Fatalf("registry.ClassificationService() = %p, want %p", got, newService)
	}
	if got := registry.MemoryStore(); got == nil {
		t.Fatal("registry.MemoryStore() = nil, want populated store")
	}
	if got := registry.ModelSelector(); got != newModelSelector {
		t.Fatalf("registry.ModelSelector() = %p, want %p", got, newModelSelector)
	}
	if got := server.service.GetRouter(); got == nil || got.RuntimeRegistry != registry {
		t.Fatalf("reloaded router did not retain runtime registry")
	}
}
