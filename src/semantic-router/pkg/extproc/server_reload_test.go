package extproc

import (
	"errors"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
)

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

	order := make([]string, 0, 4)
	parseReloadConfig = func(path string) (*config.RouterConfig, error) {
		order = append(order, "parse")
		if path != configPath {
			t.Fatalf("parseReloadConfig() path = %q, want %q", path, configPath)
		}
		return candidateCfg, nil
	}
	ensureReloadConfigModels = func(cfg *config.RouterConfig) error {
		order = append(order, "ensure")
		if cfg != candidateCfg {
			t.Fatalf("ensureReloadConfigModels() cfg = %p, want %p", cfg, candidateCfg)
		}

		specs, err := modeldownload.BuildModelSpecs(cfg)
		if err != nil {
			t.Fatalf("BuildModelSpecs() error = %v", err)
		}

		wantPaths := []string{
			"models/mom-embedding-ultra",
			"models/mmbert32k-intent-classifier-merged",
			"models/mmbert32k-pii-detector-merged",
			"models/mmbert32k-jailbreak-detector-merged",
			"models/mmbert32k-factcheck-classifier-merged",
			"models/mmbert32k-feedback-detector-merged",
		}
		assertModelSpecPaths(t, specs, wantPaths)
		if len(specs) != len(wantPaths) {
			t.Fatalf("BuildModelSpecs() returned %d specs, want %d", len(specs), len(wantPaths))
		}

		return nil
	}
	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		order = append(order, "build")
		if cfg != candidateCfg {
			t.Fatalf("buildReloadRouter() cfg = %p, want %p", cfg, candidateCfg)
		}
		return &OpenAIRouter{Config: cfg}, nil
	}
	replaceReloadConfig = func(cfg *config.RouterConfig) {
		order = append(order, "replace")
		if cfg != candidateCfg {
			t.Fatalf("replaceReloadConfig() cfg = %p, want %p", cfg, candidateCfg)
		}
	}

	if err := server.reloadRouterFromFile(configPath); err != nil {
		t.Fatalf("reloadRouterFromFile() error = %v", err)
	}

	if got := server.service.GetRouter(); got == oldRouter || got.Config != candidateCfg {
		t.Fatalf("router swap did not install candidate config")
	}

	wantOrder := []string{"parse", "ensure", "build", "replace"}
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

	order := make([]string, 0, 2)
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

	buildCalls := 0
	buildReloadRouter = func(cfg *config.RouterConfig) (*OpenAIRouter, error) {
		buildCalls++
		if cfg != candidateCfg {
			t.Fatalf("buildReloadRouter() cfg = %p, want %p", cfg, candidateCfg)
		}
		return &OpenAIRouter{Config: cfg}, nil
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
	if replaceCalls != 0 {
		t.Fatalf("replaceReloadConfig() calls = %d, want 0", replaceCalls)
	}
	if got := server.service.GetRouter(); got == oldRouter || got.Config != candidateCfg {
		t.Fatalf("router swap did not install kubernetes config")
	}
}

func stubReloadSeams(t *testing.T) func() {
	t.Helper()

	originalParse := parseReloadConfig
	originalEnsure := ensureReloadConfigModels
	originalBuild := buildReloadRouter
	originalReplace := replaceReloadConfig

	return func() {
		parseReloadConfig = originalParse
		ensureReloadConfigModels = originalEnsure
		buildReloadRouter = originalBuild
		replaceReloadConfig = originalReplace
	}
}

func amdDeployConfigPath(t *testing.T) string {
	t.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve test file path")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "../../../../deploy/amd/config.yaml"))
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
