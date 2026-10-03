//go:build !windows && cgo

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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func preparedInventoryService(t *testing.T, runtime *native.Runtime, configs ...*config.RouterConfig) (*config.RouterConfig, *services.ClassificationService) {
	t.Helper()
	cfg := &config.RouterConfig{}
	if len(configs) > 0 {
		cfg = configs[0]
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil, classification.RecipeRuntimeOptions{Runtime: runtime})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	return cfg, services.NewRecipeClassificationService(classifiers, cfg)
}

// These controlled typed handles exercise inventory lifecycle and API wiring;
// provider inference correctness is covered by the native integration tests.
func prepareInventoryTask(t *testing.T, runtime *native.Runtime, recipe, name, contract, device string) *binding.Resolved[string, string] {
	t.Helper()
	task, err := binding.Register(binding.NewRegistry(runtime.ObserveBinding), contract, func(string) error { return nil }, func(string, string) error { return nil })
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
	runtime := native.New(nil)
	cfg, service := preparedInventoryService(t, runtime)
	for _, task := range []struct{ name, contract string }{
		{"domain_classifier", "label_distribution.v1"},
		{"prompt_guard", "label_distribution.v1"},
		{"pii_classifier", "token_spans.v1"},
		{"fact_check_classifier", "label_distribution.v1"},
		{"feedback_detector", "label_distribution.v1"},
		{"modality_detector", "label_distribution.v1"},
		{"safety.safe", "label_distribution.v1"},
		{"safety.safe.hazard", "label_scores.v1"},
		{"embedding", "embedding.v1"},
		{"rag.reranker", "relevance_scores.v1"},
	} {
		prepareInventoryTask(t, runtime, "vela", task.name, task.contract, "migraphx:0")
	}
	api := &ClassificationAPIServer{config: cfg, classificationSvc: service}
	response := api.buildModelsInfoResponse()
	if len(response.Models) != 10 || response.Summary.LoadedModels != 10 || response.Summary.TotalModels != 10 || !response.System.GPUAvailable {
		t.Fatalf("incomplete prepared task inventory: %+v", response)
	}
	for _, model := range response.Models {
		if len(model.Metadata["resource_id"]) != 64 {
			t.Fatalf("missing opaque physical resource identity: %+v", model)
		}
		if !model.Loaded || model.State != "ready" || model.Recipe != "vela" || model.Metadata["device"] != "migraphx:0" || model.Metadata["max_sequence_length"] != "512" {
			t.Fatalf("lost runtime evidence: %+v", model)
		}
	}
	_ = requireModelInfo(t, response.Models, "category_classifier")
	_ = requireModelInfo(t, response.Models, "jailbreak_classifier")
	_ = requireModelInfo(t, response.Models, "safety.safe.hazard")
}

func TestPreparedInventoryDoesNotInferReadinessFromConstructedClassifier(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.CategoryModel = config.CategoryModel{ModelID: "models/custom-domain", CategoryMappingPath: "unused-domain-mapping.json"}
	cfg.PIIModel = config.PIIModel{ModelID: "models/custom-pii", PIIMappingPath: "unused-pii-mapping.json"}
	cfg.PromptGuard = config.PromptGuardConfig{Enabled: true, ModelID: "models/custom-guard", JailbreakMappingPath: "unused-guard-mapping.json"}
	cfg.BertModelPath = "models/custom-embedding"
	cfg, service := preparedInventoryService(t, native.New(nil), cfg)
	if !service.HasClassifier() {
		t.Fatal("fixture must contain a constructed classifier")
	}
	if previous := appendConfiguredModels(nil, cfg, classifierModelAvailability{core: true}); len(previous) != 4 {
		t.Fatalf("fixture must reproduce the four configured, unused core models: %+v", previous)
	}
	api := &ClassificationAPIServer{config: cfg, classificationSvc: service}
	response := api.buildModelsInfoResponse()
	if len(response.Models) != 0 || response.Summary.LoadedModels != 0 || response.System.GPUAvailable {
		t.Fatalf("unused models were reported ready: %+v", response)
	}
}

func TestPreparedInventoryUsesPublishedGenerationAndPreservesStartupReadiness(t *testing.T) {
	pool := binding.NewPool()
	oldRuntime, nextRuntime := native.New(pool), native.New(pool)
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
