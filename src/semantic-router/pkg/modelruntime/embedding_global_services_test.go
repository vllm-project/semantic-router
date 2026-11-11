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

// The global services embed with embeddings.default; a recipe's RAG source
// that names its own deployment does not lend it to them.
func TestGlobalServicesEmbedWithTheDefaultDeployment(t *testing.T) {
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
	cfg.EmbeddingDeployment = "global-default"
	cfg.Memory.InUse, cfg.Memory.Milvus.Dimension = true, 2
	cfg.GlobalPlugins = []config.DecisionPlugin{{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]any{
		"enabled": true, "sources": []any{map[string]any{"type": config.ContextSourceMemory}},
	})}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"global-default": {Provider: config.OpenAIEmbeddingsProvider, Endpoint: server.URL, Model: "global-default", Capabilities: &config.DeploymentCapabilities{Dimensions: 2}},
		"recipe-only":    {Provider: config.ModelRuntimeProvider, Device: "rocm:7", Artifact: "/not-installed/recipe-only"},
	}
	cfg.Decisions = []config.Decision{{Name: "docs", Plugins: []config.DecisionPlugin{{
		Type: config.DecisionPluginContext,
		Configuration: config.MustStructuredPayload(map[string]any{"enabled": true, "sources": []any{map[string]any{
			"type": config.ContextSourceRAG, "backend": "milvus",
			"backend_config": map[string]any{"collection": "docs", "embedding": map[string]any{"deployment": "recipe-only"}},
		}}}),
	}}}}
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
		t.Fatalf("global default: %v / %v", vector, err)
	}
	entries := runtime.PreparedBindings()
	if len(entries) != 1 || entries[0].Identity.Recipe != string(config.GlobalModelScope) || entries[0].Capability.Provider != "http" {
		t.Fatalf("wrong physical provider: %+v", entries)
	}
	if services.Has("recipe-only") {
		t.Fatal("the global services prepared a recipe's RAG deployment")
	}
}
