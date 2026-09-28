//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestConvenienceDiagnosticsHTTPRejectsForeignRecipe(t *testing.T) {
	cfg, service := preparedInventoryService(t, native.New(nil), &config.RouterConfig{Recipes: []config.RoutingRecipe{{Name: config.DefaultRecipeName}, {Name: "private"}}})
	api := &ClassificationAPIServer{config: cfg, classificationSvc: service}
	for _, test := range []struct {
		name, body string
		handler    func(http.ResponseWriter, *http.Request)
	}{
		{"embeddings", `{"recipe":"foreign","texts":["hello"]}`, api.handleEmbeddings},
		{"similarity", `{"recipe":"foreign","text1":"hello","text2":"world"}`, api.handleSimilarity},
		{"batch-similarity", `{"recipe":"foreign","query":"hello","candidates":["world"]}`, api.handleBatchSimilarity},
		{"batch", `{"recipe":"foreign","texts":["hello"]}`, api.handleBatchClassification},
		{"combined", `{"recipe":"foreign","text":"hello"}`, api.handleCombinedClassification},
		{"nli", `{"recipe":"foreign","premise":"hello","hypothesis":"greeting"}`, api.handleNLIClassification},
	} {
		t.Run(test.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			test.handler(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(test.body)))
			if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "INVALID_RECIPE") {
				t.Fatalf("requested recipe silently fell back: %d %s", w.Code, w.Body.String())
			}
		})
	}
}

func TestConvenienceEmbeddingUsesSelectedRecipePreparedProvider(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Input []string `json:"input"`
			Model string   `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		vector := make([]float32, 64)
		vector[0] = 1
		if request.Model == "private-embedder" {
			vector[0], vector[1] = 0, 1
		}
		data := make([]map[string]any, len(request.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": vector}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer upstream.Close()
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "remote"
	cfg.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible
	cfg.EmbeddingConfig.TargetDimension = 64
	cfg.ModelDeployments = map[string]config.ModelDeployment{}
	for _, name := range []string{"default", "private"} {
		cfg.ModelDeployments[name] = config.ModelDeployment{Provider: "http", ExternalModel: name + "-embedder"}
		cfg.ExternalModels = append(cfg.ExternalModels, config.ExternalModelConfig{Name: name + "-embedder", ModelName: name + "-embedder", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: upstream.URL}})
		cfg.Recipes = append(cfg.Recipes, config.RoutingRecipe{Name: config.RecipeName(name), Profile: config.RoutingProfile{
			ModelBindings: map[string]config.ModelBinding{"embedding": {Deployment: name, Contract: "embedding.v1", Adapter: "openai_compatible"}},
			Signals:       config.Signals{EmbeddingRules: []config.EmbeddingRule{{Name: "active", Candidates: []string{"candidate"}}}},
			Decisions:     []config.Decision{{Name: "active", Rules: config.RuleNode{Type: "embedding", Name: "active"}}},
		}})
		cfg.Entrypoints = append(cfg.Entrypoints, config.EntrypointMapping{ModelNames: []string{name + "-alias"}, Recipe: config.RecipeName(name)})
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer classifiers.Close()
	if err = classifiers.InitializeRuntime(); err != nil {
		t.Fatal(err)
	}
	service := services.NewRecipeClassificationService(classifiers, cfg)
	defer service.Close()
	api := &ClassificationAPIServer{classificationSvc: service, config: cfg}
	for _, test := range []struct {
		recipe string
		want   []float32
	}{{"", []float32{1, 0}}, {"default", []float32{1, 0}}, {"private", []float32{0, 1}}} {
		body, _ := json.Marshal(EmbeddingRequest{Recipe: test.recipe, Texts: []string{"hello"}, Model: "remote", Dimension: 64})
		w := httptest.NewRecorder()
		api.handleEmbeddings(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(string(body))))
		if w.Code != http.StatusOK {
			t.Fatalf("recipe %q: %d %s", test.recipe, w.Code, w.Body.String())
		}
		var result EmbeddingResponse
		if err = json.Unmarshal(w.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Recipe != diagnosticRecipeName(test.recipe) || len(result.Embeddings) != 1 || result.Embeddings[0].Embedding[0] != test.want[0] || result.Embeddings[0].Embedding[1] != test.want[1] {
			t.Fatalf("wrong recipe provider: %+v", result)
		}
	}
}
