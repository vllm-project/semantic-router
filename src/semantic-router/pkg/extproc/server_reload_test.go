package extproc

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

var expectedAMDModelPaths = []string{
	"models/mom-embedding-ultra",
	"models/mmbert32k-intent-classifier-merged",
	"models/mmbert32k-pii-detector-merged",
	"models/mmbert32k-jailbreak-detector-merged",
	"models/mmbert32k-factcheck-classifier-merged",
	"models/mmbert32k-feedback-detector-merged",
}

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

func TestReloadRouterFromFileEnsuresAMDModelsBeforeSwap(t *testing.T) {
	configPath := amdDeployConfigPath(t)
	candidateCfg := loadRouterConfigFixture(t, configPath)

	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()

	oldRouter := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "old-model"},
	}}
	server := &Server{
		configPath: configPath,
		service:    NewRouterService(oldRouter),
	}

	order := make([]string, 0, 6)
	stubSuccessfulReloadSequence(t, configPath, candidateCfg, &order)

	if err := server.reloadRouterFromFile(configPath); err != nil {
		t.Fatalf("reloadRouterFromFile() error = %v", err)
	}

	if got := server.service.GetRouter(); got == oldRouter || got.Config != candidateCfg {
		t.Fatalf("router swap did not install candidate config")
	}

	wantOrder := []string{"parse", "ensure", "prepare", "build", "warmup", "replace"}
	if !reflect.DeepEqual(order, wantOrder) {
		t.Fatalf("reload order = %v, want %v", order, wantOrder)
	}
}

func TestReloadRouterFromFileDoesNotSwapWhenModelEnsureFails(t *testing.T) {
	const configPath = "/tmp/router-config.yaml"

	restoreReloadSeams := stubReloadSeams(t)
	defer restoreReloadSeams()

	candidateCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "candidate"},
	}
	oldRouter := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "old"},
	}}
	server := &Server{
		configPath: configPath,
		service:    NewRouterService(oldRouter),
	}

	order := make([]string, 0, 3)
	parseReloadConfig = func(path string) (*config.RouterConfig, error) {
		order = append(order, "parse")
		return candidateCfg, nil
	}
	ensureReloadConfigModels = func(cfg *config.RouterConfig) error {
		order = append(order, "ensure")
		return errors.New("download unavailable")
	}
	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		t.Fatalf("buildReloadRouter() should not be called on ensure failure")
		return nil, nil
	}
	prepareReloadRuntime = func(cfg *config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		t.Fatalf("prepareReloadRuntime() should not be called on ensure failure")
		return modelruntime.EmbeddingRuntimeState{}, nil
	}
	warmupReloadRouter = func(router *OpenAIRouter, state modelruntime.EmbeddingRuntimeState) error {
		t.Fatalf("warmupReloadRouter() should not be called on ensure failure")
		return nil
	}
	replaceReloadConfig = func(cfg *config.RouterConfig) {
		t.Fatalf("replaceReloadConfig() should not be called on ensure failure")
	}

	err := server.reloadRouterFromFile(configPath)
	if err == nil {
		t.Fatal("reloadRouterFromFile() error = nil, want failure")
	}
	if got := err.Error(); got != "model download preflight failed: download unavailable" {
		t.Fatalf("reloadRouterFromFile() error = %q", got)
	}
	if got := server.service.GetRouter(); got != oldRouter {
		t.Fatalf("router changed on ensure failure")
	}

	wantOrder := []string{"parse", "ensure"}
	if !reflect.DeepEqual(order, wantOrder) {
		t.Fatalf("reload order = %v, want %v", order, wantOrder)
	}
}

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

	oldCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "old"},
	}
	newCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "new"},
	}
	oldService := &services.ClassificationService{}
	newService := &services.ClassificationService{}
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
		}, nil
	}
	warmupReloadRouter = func(router *OpenAIRouter, state modelruntime.EmbeddingRuntimeState) error { return nil }
	replaceReloadConfig = func(cfg *config.RouterConfig) {}

	if err := server.reloadRouterFromConfig("file", server.configPath, newCfg); err != nil {
		t.Fatalf("reloadRouterFromConfig() error = %v", err)
	}

	if got := registry.CurrentConfig(); got != newCfg {
		t.Fatalf("registry.CurrentConfig() = %p, want %p", got, newCfg)
	}
	if got := registry.ClassificationService(); got != newService {
		t.Fatalf("registry.ClassificationService() = %p, want %p", got, newService)
	}
	if got := registry.MemoryStore(); got == nil {
		t.Fatal("registry.MemoryStore() = nil, want populated store")
	}
	if got := server.service.GetRouter(); got == nil || got.RuntimeRegistry != registry {
		t.Fatalf("reloaded router did not retain runtime registry")
	}
}

func stubSuccessfulReloadSequence(
	t *testing.T,
	configPath string,
	candidateCfg *config.RouterConfig,
	order *[]string,
) {
	t.Helper()

	stubReloadParse(t, configPath, candidateCfg, order)
	stubReloadEnsure(t, candidateCfg, order)
	stubReloadPrepare(t, candidateCfg, order)
	stubReloadBuild(t, candidateCfg, order)
	stubReloadWarmup(t, candidateCfg, order)
	stubReloadReplace(t, candidateCfg, order)
}

func stubReloadParse(
	t *testing.T,
	configPath string,
	candidateCfg *config.RouterConfig,
	order *[]string,
) {
	t.Helper()

	parseReloadConfig = func(path string) (*config.RouterConfig, error) {
		appendReloadStep(order, "parse")
		if path != configPath {
			t.Fatalf("parseReloadConfig() path = %q, want %q", path, configPath)
		}
		return candidateCfg, nil
	}
}

func stubReloadEnsure(t *testing.T, candidateCfg *config.RouterConfig, order *[]string) {
	t.Helper()

	ensureReloadConfigModels = func(cfg *config.RouterConfig) error {
		appendReloadStep(order, "ensure")
		if cfg != candidateCfg {
			t.Fatalf("ensureReloadConfigModels() cfg = %p, want %p", cfg, candidateCfg)
		}

		specs, err := modeldownload.BuildModelSpecs(cfg)
		if err != nil {
			t.Fatalf("BuildModelSpecs() error = %v", err)
		}

		assertModelSpecPaths(t, specs, expectedAMDModelPaths)
		if len(specs) != len(expectedAMDModelPaths) {
			t.Fatalf("BuildModelSpecs() returned %d specs, want %d", len(specs), len(expectedAMDModelPaths))
		}

		return nil
	}
}

func stubReloadPrepare(t *testing.T, candidateCfg *config.RouterConfig, order *[]string) {
	t.Helper()

	prepareReloadRuntime = func(cfg *config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		appendReloadStep(order, "prepare")
		if cfg != candidateCfg {
			t.Fatalf("prepareReloadRuntime() cfg = %p, want %p", cfg, candidateCfg)
		}
		return modelruntime.EmbeddingRuntimeState{AnyReady: true, ToolsReady: true}, nil
	}
}

func stubReloadBuild(t *testing.T, candidateCfg *config.RouterConfig, order *[]string) {
	t.Helper()

	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		appendReloadStep(order, "build")
		if cfg != candidateCfg {
			t.Fatalf("buildReloadRouter() cfg = %p, want %p", cfg, candidateCfg)
		}
		return &OpenAIRouter{Config: cfg}, nil
	}
}

func stubReloadWarmup(t *testing.T, candidateCfg *config.RouterConfig, order *[]string) {
	t.Helper()

	warmupReloadRouter = func(router *OpenAIRouter, state modelruntime.EmbeddingRuntimeState) error {
		appendReloadStep(order, "warmup")
		if router == nil || router.Config != candidateCfg {
			t.Fatalf("warmupReloadRouter() router config mismatch")
		}
		if !state.AnyReady || !state.ToolsReady {
			t.Fatalf("warmupReloadRouter() state = %+v, want ready", state)
		}
		return nil
	}
}

func stubReloadReplace(t *testing.T, candidateCfg *config.RouterConfig, order *[]string) {
	t.Helper()

	replaceReloadConfig = func(cfg *config.RouterConfig) {
		appendReloadStep(order, "replace")
		if cfg != candidateCfg {
			t.Fatalf("replaceReloadConfig() cfg = %p, want %p", cfg, candidateCfg)
		}
	}
}

func appendReloadStep(order *[]string, step string) {
	*order = append(*order, step)
}

func stubReloadSeams(t *testing.T) func() {
	t.Helper()

	originalParse := parseReloadConfig
	originalEnsure := ensureReloadConfigModels
	originalPrepare := prepareReloadRuntime
	originalBuild := buildReloadRouter
	originalWarmup := warmupReloadRouter
	originalReplace := replaceReloadConfig

	return func() {
		parseReloadConfig = originalParse
		ensureReloadConfigModels = originalEnsure
		prepareReloadRuntime = originalPrepare
		buildReloadRouter = originalBuild
		warmupReloadRouter = originalWarmup
		replaceReloadConfig = originalReplace
	}
}

func amdDeployConfigPath(t *testing.T) string {
	t.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve test file path")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "../../../../deploy/recipes/balance.yaml"))
}

func loadRouterConfigFixture(t *testing.T, path string) *config.RouterConfig {
	t.Helper()

	cfg, err := config.Parse(path)
	if err != nil {
		t.Fatalf("config.Parse(%q) error = %v", path, err)
	}
	return cfg
}

func assertModelSpecPaths(t *testing.T, specs []modeldownload.ModelSpec, wantPaths []string) {
	t.Helper()

	gotPaths := make([]string, 0, len(specs))
	for _, spec := range specs {
		gotPaths = append(gotPaths, spec.LocalPath)
	}

	for _, wantPath := range wantPaths {
		found := false
		for _, gotPath := range gotPaths {
			if gotPath == wantPath {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("BuildModelSpecs() missing %q; got %v", wantPath, gotPaths)
		}
	}
}
