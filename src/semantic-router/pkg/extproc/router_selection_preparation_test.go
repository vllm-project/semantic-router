package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelselection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestPreparedMLSelectionUsesExplicitEmbedding(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
	for _, recipe := range []config.RecipeName{config.DefaultRecipeName, "ml-recipe"} {
		t.Run(string(recipe), func(t *testing.T) {
			const query = "select the second candidate using its embedding"
			var queryCalls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Input []string `json:"input"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("decode embedding request: %v", err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				for _, input := range request.Input {
					if input == query {
						queryCalls.Add(1)
					}
				}
				w.Header().Set("Content-Type", "application/json")
				if err := json.NewEncoder(w).Encode(map[string]interface{}{
					"data": []map[string]interface{}{{"index": 0, "embedding": []float64{0, 1}}},
				}); err != nil {
					t.Errorf("encode embedding response: %v", err)
				}
			}))
			defer server.Close()

			cfg := preparedMLSelectionConfig(t, server.URL, recipe)
			prepared, err := modelruntime.PrepareOwnedRecipeEmbeddings(context.Background(), cfg, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer prepared.Close()
			embed, defaultEmbedding := resolveSelectionEmbeddingFunc(cfg, prepared)
			registry := createModelSelectorRegistry(cfg, nil, embed, defaultEmbedding)
			defer registry.Close()
			selector, ok := registry.Get(selection.MethodKNN)
			if !ok {
				t.Fatal("KNN selector was not registered")
			}
			result, err := selector.Select(context.Background(), &selection.SelectionContext{
				Query: query, CategoryName: "math", DecisionName: "ml", RecipeName: recipe,
				CandidateModels: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if result.SelectedModel != "model-b" || queryCalls.Load() != 1 {
				t.Fatalf("selection = %+v, embedding query calls = %d; want model-b and one real embedding request", result, queryCalls.Load())
			}
			if !prepared.Has("mmbert") || defaultEmbedding.ModelType != "mmbert" {
				t.Fatal("explicit ML request displaced the primary embedding consumer")
			}
		})
	}
}

func preparedMLSelectionConfig(t *testing.T, endpoint string, recipe config.RecipeName) *config.RouterConfig {
	t.Helper()
	artifact, err := json.Marshal(map[string]interface{}{
		"algorithm": "knn", "format_version": 2, "trained": true, "k": 1,
		"embeddings": [][]float64{
			modelselection.CombineEmbeddingWithCategory([]float64{1, 0}, "math"),
			modelselection.CombineEmbeddingWithCategory([]float64{0, 1}, "math"),
		},
		"labels": []string{"model-a", "model-b"},
	})
	if err != nil {
		t.Fatal(err)
	}
	artifactPath := filepath.Join(t.TempDir(), "knn.json")
	if err := os.WriteFile(artifactPath, artifact, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{RouterOptions: config.RouterOptions{AutoModelNames: []string{"auto"}}}
	cfg.EmbeddingConfig = config.HNSWConfig{Backend: config.EmbeddingBackendOpenAICompatible, ModelType: "mmbert", TargetDimension: 2}
	cfg.EmbeddingModels.Endpoint = config.EmbeddingEndpointConfig{BaseURL: endpoint + "/v1", Model: "test-embedding"}
	cfg.ModelSelection.ML = config.MLSelectionConfig{
		ModelType: "  Qwen3  ", EmbeddingDim: 2,
		KNN: config.MLKNNConfig{K: 1, PretrainedPath: artifactPath},
	}
	cfg.Decisions = []config.Decision{
		{Name: "ml", Algorithm: &config.AlgorithmConfig{Type: "knn"}},
		{Name: "primary", Algorithm: &config.AlgorithmConfig{Type: "router_dc"}},
	}
	return cfg.ConfigForRecipe(&config.RoutingRecipe{
		Name: recipe, Profile: config.RoutingProfile{Decisions: cfg.Decisions},
	})
}
