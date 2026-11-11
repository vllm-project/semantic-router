//go:build !windows

package apiserver

import (
	"context"
	"io"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func preparedInventoryService(t *testing.T, runtime *serving.Runtime, configs ...*config.RouterConfig) (*config.RouterConfig, *services.ClassificationService) {
	t.Helper()
	return preparedInventoryServiceWith(t, classification.RecipeRuntimeOptions{Runtime: runtime}, configs...)
}

func preparedInventoryServiceWith(t *testing.T, options classification.RecipeRuntimeOptions, configs ...*config.RouterConfig) (*config.RouterConfig, *services.ClassificationService) {
	t.Helper()
	cfg := &config.RouterConfig{}
	if len(configs) > 0 {
		cfg = configs[0]
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	return cfg, services.NewRecipeClassificationService(classifiers, cfg)
}

// These controlled typed handles exercise inventory lifecycle and API wiring;
// provider inference correctness is covered by the native integration tests.
func prepareInventoryTask(t *testing.T, runtime *serving.Runtime, recipe, name, contract, device string) *binding.Resolved[string, string] {
	t.Helper()
	task, err := binding.RegisterTask(binding.NewRegistry(runtime.ObserveBinding), contract, contract, func(string) error { return nil }, func(string, string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	resource, err := runtime.Pool.Acquire(context.Background(), binding.ResourceIdentity{Artifact: "models/" + name, Provider: "test", Device: device, Precision: "fp32"}, "", nil, func(context.Context) (io.Closer, error) { return io.NopCloser(strings.NewReader("")), nil })
	if err != nil {
		t.Fatal(err)
	}
	handle, err := task.Resolve(binding.Identity{Recipe: recipe, Name: name, Contract: contract, Deployment: "test-deployment", Adapter: "test"}, binding.Capability{Contract: contract, Provider: "test", Device: device, Precision: "fp32", Limits: binding.Limits{ModelTokens: 32768, TaskTokens: 512, Overflow: "truncate"}}, resource, func(_ context.Context, _ io.Closer, text string) (string, error) { return text, nil })
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handle.Close() })
	handle.Ready()
	return handle
}

func TestPreparedInventoryIncludesEveryTaskAndRecipe(t *testing.T) {
	runtime := serving.New(nil, nil)
	cfg, service := preparedInventoryService(t, runtime)
	for _, task := range []struct{ name, contract string }{
		{"jailbreak", "decision.v1"},
		{"pii", "decision.v1"},
		{"embedding", "embedding.v1"},
	} {
		prepareInventoryTask(t, runtime, "vela", task.name, task.contract, "rocm:0")
	}
	api := &ClassificationAPIServer{config: cfg, classificationSvc: service}
	response := api.buildModelsInfoResponse()
	if len(response.Models) != 3 || response.Summary.LoadedModels != 3 || response.Summary.TotalModels != 3 || !response.System.GPUAvailable {
		t.Fatalf("incomplete prepared task inventory: %+v", response)
	}
	for _, model := range response.Models {
		if len(model.Metadata["resource_id"]) != 64 {
			t.Fatalf("missing opaque physical resource identity: %+v", model)
		}
		if !model.Loaded || model.State != "ready" || model.Recipe != "vela" || model.Metadata["device"] != "rocm:0" || model.Metadata["max_sequence_length"] != "512" {
			t.Fatalf("lost runtime evidence: %+v", model)
		}
	}
	_ = requireModelInfo(t, response.Models, "jailbreak")
	_ = requireModelInfo(t, response.Models, "pii")
	_ = requireModelInfo(t, response.Models, "test_embedding_model")
}

func TestModelsUseGPUReadsTheRuntimeDeviceNames(t *testing.T) {
	for device, want := range map[string]bool{"cuda:0": true, "rocm:1": true, "xpu": true, "mps": true, "cpu": false, "": false} {
		models := []ModelInfo{{Loaded: true, Metadata: map[string]string{"device": device}}, {Loaded: false, Metadata: map[string]string{"device": "cuda:0"}}}
		if got := modelsUseGPU(models); got != want {
			t.Fatalf("device %q: modelsUseGPU = %v, want %v", device, got, want)
		}
	}
}

func TestPreparedInventoryUsesPublishedGenerationAndPreservesStartupReadiness(t *testing.T) {
	pool := binding.NewPool()
	oldRuntime, nextRuntime := serving.New(nil, pool), serving.New(nil, pool)
	oldCfg, oldService := preparedInventoryService(t, oldRuntime)
	nextCfg, nextService := preparedInventoryService(t, nextRuntime)
	prepareInventoryTask(t, oldRuntime, "old", "domain_classifier", "label_distribution.v1", "rocm:0")
	prepareInventoryTask(t, nextRuntime, "next", "domain_classifier", "label_distribution.v1", "cpu")
	registry := routerruntime.NewRegistry(oldCfg)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: oldCfg, ClassificationService: oldService})
	api := &ClassificationAPIServer{runtimeRegistry: registry}
	if !api.buildModelsInfoResponse().System.GPUAvailable {
		t.Fatal("published GPU binding missing")
	}
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: nextCfg, ClassificationService: nextService})
	response := api.buildModelsInfoResponse()
	if response.System.GPUAvailable || len(response.Models) != 1 || response.Models[0].Recipe != "next" {
		t.Fatalf("retiring generation leaked into current inventory: %+v", response)
	}
	api.configPath = filepath.Join(t.TempDir(), "config.yaml")
	writer := registry.StartupStatusWriter(startupstatus.NewFileWriter(api.configPath))
	if err := writer.Write(startupstatus.State{Phase: "initializing_models", Ready: false, TotalModels: 2, ReadyModels: 1}); err != nil {
		t.Fatal(err)
	}
	response = api.buildModelsInfoResponse()
	if response.Summary.Ready || response.Summary.TotalModels != 2 || response.Summary.LoadedModels != 1 {
		t.Fatalf("partial preparation incorrectly marked router ready: %+v", response.Summary)
	}
}
