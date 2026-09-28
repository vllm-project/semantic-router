//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func embeddingInventoryResponse(t *testing.T, api *ClassificationAPIServer) embeddingModelsResponse {
	t.Helper()
	response := httptest.NewRecorder()
	api.handleEmbeddingModelsInfo(response, httptest.NewRequest(http.MethodGet, apiInventoryEmbeddingModels, nil))
	if response.Code != http.StatusOK {
		t.Fatalf("embedding inventory status: %d", response.Code)
	}
	var body embeddingModelsResponse
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	return body
}

func TestEmbeddingInventoryUsesActualGlobalAPIAndRefresh(t *testing.T) {
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		data := make([]map[string]any, len(request.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": []float32{1, 0}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	t.Cleanup(endpoint.Close)
	cfg := &config.RouterConfig{}
	cfg.API.Embeddings.Enabled = true
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "bert", TargetDimension: 2}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"global": {Provider: "http", ExternalModel: "global"}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "openai_compatible", Contract: "embedding.v1"}}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "global", ModelName: "global", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: endpoint.URL}}}
	service, err := services.NewClassificationServiceFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = service.Close() })
	if service.GetClassifier().PreparedEmbeddings().Ready() {
		t.Fatal("fixture must have no default-recipe embedding")
	}
	api := &ClassificationAPIServer{config: cfg, classificationSvc: service}
	response := embeddingInventoryResponse(t, api)
	if response.Count != 1 || len(response.Models) != 1 {
		t.Fatalf("global API owner missing: %+v", response)
	}
	model := response.Models[0]
	if model.Recipe != "@global" || model.Type != "embedding" || model.Metadata["binding"] != "api.embedding" || model.Metadata["default_dimension"] != "2" || model.Metadata["modalities"] != "text" {
		t.Fatalf("actual global embedding metadata differs: %+v", model)
	}
	disabled := *cfg
	disabled.API.Embeddings.Enabled = false
	refreshed := make(chan error, 1)
	go func() { refreshed <- service.TryRefreshRuntimeConfig(&disabled) }()
	select {
	case refreshErr := <-refreshed:
		if refreshErr != nil {
			t.Fatal(refreshErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("inventory retained a lease and blocked refresh")
	}
	if response := embeddingInventoryResponse(t, api); response.Count != 0 || len(response.Models) != 0 {
		t.Fatalf("retired global API leaked into inventory: %+v", response)
	}
}

func prepareEmbeddingInventoryOwner(t *testing.T, runtime *native.Runtime, recipe, name string) {
	t.Helper()
	task, err := binding.Register(binding.NewRegistry(runtime.ObserveBinding), "embedding.v1", func(string) error { return nil }, func(string, string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	resource, err := runtime.Pool.Acquire(context.Background(), binding.ResourceIdentity{Artifact: "models/shared-embedding", Provider: "test", Device: "cpu", Precision: "fp32"}, "", nil, func(context.Context) (io.Closer, error) { return io.NopCloser(strings.NewReader("")), nil })
	if err != nil {
		t.Fatal(err)
	}
	handle, err := task.Resolve(binding.Identity{Recipe: recipe, Name: name, Contract: "embedding.v1", Deployment: "shared", Adapter: "test"}, binding.Capability{Contract: "embedding.v1", Provider: "test", Device: "cpu", Precision: "fp32", Embedding: &binding.EmbeddingCapability{Dimension: 384, Modalities: []string{"text", "image", "audio"}}}, resource, func(_ context.Context, _ io.Closer, value string) (string, error) { return value, nil })
	if err != nil {
		_ = resource.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handle.Close() })
	handle.Ready()
}

func TestEmbeddingInventoryPreservesOwnersAndReleasesEveryGeneration(t *testing.T) {
	pool := binding.NewPool()
	runtime := native.New(pool)
	_, service := preparedInventoryService(t, runtime)
	prepareEmbeddingInventoryOwner(t, runtime, "@global", "api+tools.embedding")
	prepareEmbeddingInventoryOwner(t, runtime, "@global", "response_cache.embedding")
	prepareEmbeddingInventoryOwner(t, runtime, "knowledge", "embedding")
	prepareInventoryTask(t, runtime, "knowledge", "domain_classifier", "label_distribution.v1", "cpu")
	var current classificationService = service
	leases := 0
	api := &ClassificationAPIServer{classificationSvc: newLiveClassificationService(nil, nil, func() (classificationService, func(), bool) {
		leases++
		return current, func() { leases-- }, true
	})}
	response := embeddingInventoryResponse(t, api)
	if response.Count != 3 || len(response.Models) != 3 || leases != 0 {
		t.Fatalf("incomplete owner inventory or retained lease: %+v, leases=%d", response, leases)
	}
	owners := map[string]bool{}
	resourceID := response.Models[0].Metadata["resource_id"]
	for _, model := range response.Models {
		if model.Type != "embedding" || model.Metadata["resource_id"] != resourceID || len(resourceID) != 64 || model.Metadata["default_dimension"] != "384" || model.Metadata["modalities"] != "text,image,audio" {
			t.Fatalf("lost shared resource or typed capability: %+v", model)
		}
		owners[model.Recipe+"/"+model.Metadata["binding"]] = true
	}
	for _, owner := range []string{"@global/api+tools.embedding", "@global/response_cache.embedding", "knowledge/embedding"} {
		if !owners[owner] {
			t.Fatalf("shared physical instance erased owner %q", owner)
		}
	}
	_, next := preparedInventoryService(t, native.New(pool))
	for name, replacement := range map[string]classificationService{"empty prepared generation": next, "unavailable placeholder": services.NewPlaceholderClassificationService()} {
		t.Run(name, func(t *testing.T) {
			current = replacement
			response := embeddingInventoryResponse(t, api)
			if response.Count != 0 || len(response.Models) != 0 || leases != 0 {
				t.Fatalf("stale owner or retained unavailable-generation lease: %+v, leases=%d", response, leases)
			}
		})
	}
}
