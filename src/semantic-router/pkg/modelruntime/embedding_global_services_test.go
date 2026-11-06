package modelruntime

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
)

func TestGlobalModuleDefaultDoesNotBorrowRecipeEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		if request.Model != "global-default" {
			t.Errorf("global service borrowed %s", request.Model)
		}
		data := make([]map[string]any, len(request.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": []float32{1, 0}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer server.Close()
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "bert", Backend: config.EmbeddingBackendOpenAICompatible, TargetDimension: 2}
	cfg.EmbeddingModels.Endpoint = config.EmbeddingEndpointConfig{BaseURL: server.URL, Model: "global-default", Dimensions: 2}
	cfg.Tools.Enabled = true
	cfg.ModelDeployments = map[string]config.ModelDeployment{"recipe-only": {Provider: config.ModelRuntimeProvider, Device: "rocm:7", Artifact: "/not-installed/recipe-only"}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "recipe-only", Adapter: "mmbert", Contract: "embedding.v1"}}
	runtime := serving.New(nil, binding.NewPool())
	services, err := PrepareOwnedGlobalServiceEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer services.Close()
	provider, err := services.Default()
	if err != nil {
		t.Fatal(err)
	}
	if vector, err := provider.Embed(context.Background(), "hello"); err != nil || len(vector) != 2 {
		t.Fatalf("global module default: %v / %v", vector, err)
	}
	entries := runtime.PreparedBindings()
	if len(entries) != 1 || entries[0].Identity.Recipe != string(config.GlobalModelScope) || entries[0].Capability.Provider != "http" {
		t.Fatalf("wrong physical provider: %+v", entries)
	}
	if cfg.ModelBindings["embedding"].Deployment != "recipe-only" {
		t.Fatal("global preparation mutated recipe")
	}
}
