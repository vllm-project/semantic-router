package extproc

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

var expectedAMDModelPaths = []string{
	"models/mmbert-embed-32k-2d-matryoshka",
	"models/mmbert32k-intent-classifier-merged",
	"models/mmbert32k-factcheck-classifier-merged",
	"models/mmbert32k-feedback-detector-merged",
}

func TestReloadRejectsLiveArtifactMutationBeforeDownload(t *testing.T) {
	restore := stubReloadSeams(t)
	defer restore()
	artifact := t.TempDir()
	weights := filepath.Join(artifact, "model.safetensors")
	if err := os.WriteFile(weights, []byte("live weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	makeConfig := func(revision string) *config.RouterConfig {
		cfg := &config.RouterConfig{MoMRegistry: map[string]string{artifact: "test/model"}}
		cfg.CategoryModel.ModelID = artifact
		cfg.CategoryMappingPath = filepath.Join(artifact, "labels.json")
		cfg.Decisions = []config.Decision{{Name: "route", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"}}}
		cfg.ModelDeployments = map[string]config.ModelDeployment{"intent": {Provider: "candle", Artifact: artifact, Revision: revision}}
		cfg.ModelBindings = map[string]config.ModelBinding{"domain_classifier": {Deployment: "intent", Contract: "label_distribution.v1", Adapter: "mmbert32k"}}
		return cfg
	}
	previous := &OpenAIRouter{Config: makeConfig(strings.Repeat("a", 40))}
	server := &Server{service: NewRouterService(previous)}
	ensureReloadConfigModels = func(*config.RouterConfig) error {
		t.Fatal("download may mutate live weights and must not run")
		return nil
	}
	err := server.reloadRouterFromConfig("file", "config.yaml", makeConfig(strings.Repeat("b", 40)))
	if err == nil || !strings.Contains(err.Error(), "in use") {
		t.Fatalf("reload error = %v, want live artifact rejection", err)
	}
	if server.service.GetRouter() != previous || server.CurrentConfig() != previous.Config {
		t.Fatal("rejected artifact candidate replaced the live generation")
	}
	data, readErr := os.ReadFile(weights)
	if readErr != nil || string(data) != "live weights" {
		t.Fatalf("live artifact changed: %q %v", data, readErr)
	}
}

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
	buildReloadRouter = func(cfg *config.RouterConfig, _ ...*binding.Pool) (*OpenAIRouter, error) {
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

	buildReloadRouter = func(cfg *config.RouterConfig, _ ...*binding.Pool) (*OpenAIRouter, error) {
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
	return filepath.Clean(filepath.Join(filepath.Dir(file), "../../../../config/recipes/balance/config.yaml"))
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
