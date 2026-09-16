package modelruntime

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

func TestGlobalServicesSharePeerResourceWithoutBorrowingDefault(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		vector := []float32{1, 0}
		if request.Model == "local" {
			vector = []float32{0, 1}
		}
		data := make([]map[string]any, len(request.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": vector}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer server.Close()
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "bert", TargetDimension: 2}
	cfg.Tools.Enabled = true
	cfg.Memory.Enabled, cfg.Memory.EmbeddingModel, cfg.Memory.Milvus.Dimension = true, "bert", 2
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"global": {Provider: "http", ExternalModel: "global"},
		"local":  {Provider: "http", ExternalModel: "local"},
	}
	for _, name := range []string{"global", "local"} {
		cfg.ExternalModels = append(cfg.ExternalModels, config.ExternalModelConfig{Name: name, ModelName: name, ModelEndpoint: config.ClassifierVLLMEndpoint{Address: server.URL}})
	}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "openai_compatible", Contract: "embedding.v1"}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "local", Adapter: "openai_compatible", Contract: "embedding.v1"}}
	cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "route", Candidates: []string{"hello"}}}
	runtime := native.New(binding.NewPool())
	services, err := PrepareOwnedGlobalServiceEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = services.Close() })
	local, err := PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = local.Close() })
	peerConfig := cfg.ConfigForRecipe(&config.RoutingRecipe{Name: "peer", Profile: config.RoutingProfile{Signals: config.Signals{EmbeddingRules: cfg.EmbeddingRules}}})
	peer, err := PrepareOwnedRecipeEmbeddings(context.Background(), peerConfig, runtime)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = peer.Close() })
	ids := map[string]string{}
	for _, entry := range runtime.PreparedBindings() {
		ids[entry.Identity.Recipe] = entry.ResourceID
		if entry.Identity.Recipe == string(config.GlobalModelScope) && entry.Identity.Name != "tools+memory.embedding" {
			t.Fatalf("shared consumers not recorded: %+v", entry.Identity)
		}
	}
	if len(ids) != 3 || ids["@global"] == "" || ids["@global"] != ids["peer"] || ids["@global"] == ids["default"] {
		t.Fatalf("resource sharing crossed override boundary: %v", ids)
	}
	if closeErr := services.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	for name, set := range map[string]*embedding.Set{"local": local, "peer": peer} {
		provider, getErr := set.Default()
		if getErr != nil {
			t.Fatal(getErr)
		}
		vector, embedErr := provider.Embed(context.Background(), "hello")
		if embedErr != nil || len(vector) != 2 || (name == "local" && vector[1] != 1) || (name == "peer" && vector[0] != 1) {
			t.Fatalf("service retirement affected %s: %v / %v", name, vector, embedErr)
		}
	}
}
