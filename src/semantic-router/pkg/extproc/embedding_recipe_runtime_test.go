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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

func TestEmbeddingConsumerUsesSelectedRecipeBinding(t *testing.T) {
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
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "remote"
	cfg.EmbeddingConfig.TargetDimension = 2
	cfg.Tools.Enabled = true
	cfg.ModelSelection.Enabled = true
	cfg.ModelSelection.ML.ModelsPath = "test-selection"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"default-embedding": {Provider: "http", ExternalModel: "default"}, "named-embedding": {Provider: "http", ExternalModel: "named"}}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "default", ModelName: "default", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: endpoint.URL}}, {Name: "named", ModelName: "named", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: endpoint.URL}}}
	binding := func(deployment string) map[string]config.ModelBinding {
		return map[string]config.ModelBinding{"embedding": {Deployment: deployment, Contract: "embedding.v1", Adapter: "openai_compatible"}}
	}
	cfg.ModelBindings = binding("default-embedding")
	cfg.Recipes = []config.RoutingRecipe{{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{ModelBindings: binding("default-embedding")}}, {Name: "named", Profile: config.RoutingProfile{ModelBindings: binding("named-embedding")}}}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"named-entry"}, Recipe: "named"}}
	runtime := native.New(nil)
	defaultSet, err := modelruntime.PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer defaultSet.Close()
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil, classification.RecipeRuntimeOptions{Runtime: runtime, Embeddings: defaultSet})
	if err != nil {
		t.Fatal(err)
	}
	defer classifiers.Close()
	router := &OpenAIRouter{Config: cfg, Embeddings: defaultSet, Classifier: classifiers.Default(), RecipeClassifiers: classifiers}
	for _, tc := range []struct {
		recipe int
		want   int
	}{{0, 0}, {1, 1}} {
		request := &RequestContext{}
		request.Routing.SelectRecipe(&cfg.Recipes[tc.recipe])
		provider, err := router.embeddingsForRequest(request).Get("", 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		vector, err := provider.Embed(context.Background(), "query")
		if err != nil {
			t.Fatal(err)
		}
		if len(vector) != 2 || vector[tc.want] != 1 {
			t.Fatalf("recipe %s got %v", cfg.Recipes[tc.recipe].Name, vector)
		}
	}
	unknown := &RequestContext{}
	unknown.Routing.SelectRecipe(&config.RoutingRecipe{Name: "foreign"})
	if _, err := router.embeddingsForRequest(unknown).Default(); err == nil {
		t.Fatal("foreign recipe borrowed default embedding binding")
	}
}
