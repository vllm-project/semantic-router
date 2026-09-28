package services

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func embeddingAPIConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		data := make([]map[string]any, len(request.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": []float32{1, 0}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	t.Cleanup(server.Close)
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "bert", TargetDimension: 2}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"global": {Provider: "http", ExternalModel: "global"}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "openai_compatible", Contract: "embedding.v1"}}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "global", ModelName: "global", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: server.URL}}}
	return cfg
}

func TestEmbeddingAPIStandaloneLifecycleWithoutRoutingDemand(t *testing.T) {
	cfg := embeddingAPIConfig(t)
	service, err := NewClassificationServiceFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = service.Close() })
	if bindings, _ := service.PreparedBindings(); len(bindings) != 0 {
		t.Fatalf("declaration alone loaded models: %+v", bindings)
	}
	enabled := *cfg
	enabled.API.Embeddings.Enabled = true
	if refreshErr := service.TryRefreshRuntimeConfig(&enabled); refreshErr != nil {
		t.Fatal(refreshErr)
	}
	before, _ := service.PreparedBindings()
	if len(before) != 1 || before[0].Identity.Recipe != "@global" || before[0].Identity.Name != "api.embedding" {
		t.Fatalf("global API missing from inventory: %+v", before)
	}
	scope, prepared, release, err := service.AcquireEmbeddingAPISnapshot()
	if err != nil || scope.RoutingScope != config.GlobalModelScope {
		release()
		t.Fatalf("API did not acquire global snapshot: %v", err)
	}
	provider, err := prepared.Default()
	if err != nil {
		release()
		t.Fatal(err)
	}
	refreshed := make(chan error, 1)
	go func() { refreshed <- service.TryRefreshRuntimeConfig(&enabled) }()
	select {
	case refreshErr := <-refreshed:
		release()
		t.Fatalf("reload retired leased API view: %v", refreshErr)
	case <-time.After(30 * time.Millisecond):
	}
	if vector, callErr := provider.Embed(context.Background(), "still leased"); callErr != nil || len(vector) != 2 {
		release()
		t.Fatalf("leased provider failed: %v / %v", vector, callErr)
	}
	release()
	select {
	case refreshErr := <-refreshed:
		if refreshErr != nil {
			t.Fatal(refreshErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("reload did not drain after release")
	}
	after, _ := service.PreparedBindings()
	if len(after) != 1 || before[0].ResourceID != after[0].ResourceID {
		t.Fatalf("unchanged reload recreated physical model: %+v", after)
	}
	if _, callErr := provider.Embed(context.Background(), "retired"); !errors.Is(callErr, binding.ErrClosed) {
		t.Fatalf("retired handle remains callable: %v", callErr)
	}
	invalid := enabled
	invalid.ExternalModels = nil
	if refreshErr := service.TryRefreshRuntimeConfig(&invalid); refreshErr == nil {
		t.Fatal("invalid candidate was published")
	}
	_, retained, release, err := service.AcquireEmbeddingAPISnapshot()
	defer release()
	if err != nil {
		t.Fatal(err)
	}
	current, err := retained.Default()
	if err != nil {
		release()
		t.Fatal(err)
	}
	_, err = current.Embed(context.Background(), "retained after failed reload")
	release()
	if err != nil {
		t.Fatal(err)
	}
	if refreshErr := service.TryRefreshRuntimeConfig(cfg); refreshErr != nil {
		t.Fatal(refreshErr)
	}
	if bindings, _ := service.PreparedBindings(); len(bindings) != 0 {
		t.Fatalf("disabled API retained demand: %+v", bindings)
	}
	if _, callErr := current.Embed(context.Background(), "disabled"); !errors.Is(callErr, binding.ErrClosed) {
		t.Fatalf("disabled API leaked a model handle: %v", callErr)
	}
}
