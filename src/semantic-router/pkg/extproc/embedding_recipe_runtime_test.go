package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
)

// ragEmbeddingDecisions are decisions whose RAG source embeds its queries
// with deployment, or embeddings.default without one.
func ragEmbeddingDecisions(deployment string) []config.Decision {
	backend := map[string]any{"collection": "docs"}
	if deployment != "" {
		backend["embedding"] = map[string]any{"deployment": deployment}
	}
	return []config.Decision{{Name: "nearest", Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]any{
		"enabled": true, "sources": []any{map[string]any{"type": config.ContextSourceRAG, "backend": "milvus", "backend_config": backend}},
	})}}}}
}

func TestRAGQueryEmbeddingUsesTheSelectedRecipesDeployment(t *testing.T) {
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			w.WriteHeader(400)
			return
		}
		vector := []float32{1, 0}
		if request.Model == "named" {
			vector = []float32{0, 1}
		}
		data := make([]map[string]any, len(request.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": vector}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer endpoint.Close()
	remote := func(model string) config.ModelDeployment {
		return config.ModelDeployment{Provider: config.OpenAIEmbeddingsProvider, Endpoint: endpoint.URL, Model: model, Capabilities: &config.DeploymentCapabilities{Dimensions: 2}}
	}
	cfg := &config.RouterConfig{}
	cfg.EmbeddingDeployment = "default-embedding"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"default-embedding": remote("default"), "named-embedding": remote("named")}
	cfg.Decisions = ragEmbeddingDecisions("")
	cfg.Recipes = []config.RoutingRecipe{
		{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Decisions: ragEmbeddingDecisions("")}},
		{Name: "named", Profile: config.RoutingProfile{Decisions: ragEmbeddingDecisions("named-embedding")}},
	}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"named-entry"}, Recipe: "named"}}
	runtime := serving.New(nil, nil)
	defaultSet, err := modelruntime.PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer defaultSet.Close()
	classifiers, err := classification.BuildRecipeClassifiers(cfg, classification.RecipeRuntimeOptions{Runtime: runtime, Embeddings: defaultSet})
	if err != nil {
		t.Fatal(err)
	}
	defer classifiers.Close()
	router := &OpenAIRouter{Config: cfg, Embeddings: defaultSet, Classifier: classifiers.Default(), RecipeClassifiers: classifiers}
	for _, tc := range []struct {
		recipe     int
		binding    *config.EmbeddingBinding
		want       int
		unprepared string
	}{
		{0, nil, 0, "named-embedding"},
		{1, &config.EmbeddingBinding{Deployment: "named-embedding"}, 1, "default-embedding"},
	} {
		request := &RequestContext{}
		request.Routing.SelectRecipe(&cfg.Recipes[tc.recipe])
		vector, err := router.ragQueryEmbedding(context.Background(), "query", request, tc.binding)
		if err != nil {
			t.Fatal(err)
		}
		if len(vector) != 2 || vector[tc.want] != 1 {
			t.Fatalf("recipe %s got %v", cfg.Recipes[tc.recipe].Name, vector)
		}
		if router.embeddingsForRequest(request).Has(tc.unprepared) {
			t.Fatalf("recipe %s prepared %s, which none of its sources names", cfg.Recipes[tc.recipe].Name, tc.unprepared)
		}
	}
	unknown := &RequestContext{}
	unknown.Routing.SelectRecipe(&config.RoutingRecipe{Name: "foreign"})
	if _, err := router.embeddingsForRequest(unknown).Default(); err == nil {
		t.Fatal("foreign recipe borrowed the default embedding")
	}
}
