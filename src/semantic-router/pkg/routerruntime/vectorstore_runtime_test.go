package routerruntime

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestVectorStorePreparesEmbeddingWithoutRecipeClassifierBindings(t *testing.T) {
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
		if request.Model == "selected" {
			vector = []float32{0, 1}
		}
		data := make([]map[string]any, len(request.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": vector}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer server.Close()

	for _, name := range []string{"implicit embedding", "explicit embedding", "global embedding"} {
		t.Run(name, func(t *testing.T) {
			cfg := &config.RouterConfig{VectorStore: &config.VectorStoreConfig{
				Enabled: true, BackendType: "memory", EmbeddingModel: "bert", EmbeddingDimension: 2,
				FileStorageDir: t.TempDir(), MetadataStore: "memory",
			}}
			cfg.EmbeddingConfig = config.HNSWConfig{Backend: config.EmbeddingBackendOpenAICompatible, ModelType: "bert", TargetDimension: 2}
			cfg.Endpoint = config.EmbeddingEndpointConfig{BaseURL: server.URL, Model: "implicit", Dimensions: 2}
			cfg.ClassifierRules = []config.ClassifierSignalRule{{Name: "risk", Type: "local", Labels: []string{"safe", "unsafe"}}}
			cfg.SafetyRules = []config.SafetyRule{{Name: "unsafe", Threshold: .5}}
			cfg.ModelDeployments = map[string]config.ModelDeployment{
				"classifier": {Provider: "candle", Device: "cpu", Artifact: "/unavailable/recipe-classifier"},
				"embedder":   {Provider: "http", ExternalModel: "embedding-service"},
			}
			cfg.ExternalModels = []config.ExternalModelConfig{{Name: "embedding-service", ModelName: "selected", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: server.URL}}}
			cfg.ModelBindings = map[string]config.ModelBinding{
				"classifier.risk": {Deployment: "classifier", Contract: "label_distribution.v1", Adapter: "modernbert"},
				"safety.unsafe":   {Deployment: "classifier", Contract: "label_distribution.v1", Adapter: "modernbert"},
			}
			if name != "implicit embedding" {
				cfg.ModelBindings["embedding"] = config.ModelBinding{Deployment: "embedder", Contract: "embedding.v1", Adapter: "openai_compatible"}
			}
			if name == "global embedding" {
				cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": cfg.ModelBindings["embedding"]}
				cfg.ModelDeployments["local-override"] = config.ModelDeployment{Provider: "ort", Artifact: "/not-installed/recipe-only"}
				cfg.ModelBindings["embedding"] = config.ModelBinding{Deployment: "local-override", Contract: "embedding.v1", Adapter: "bert"}
			}
			if _, err := config.CompileModelBindings(cfg); err != nil {
				t.Fatalf("complete source configuration is invalid: %v", err)
			}
			original, err := json.Marshal(cfg.ModelBindings)
			if err != nil {
				t.Fatal(err)
			}
			runtime, err := NewVectorStoreRuntime(cfg)
			if err != nil {
				t.Fatalf("unrelated recipe bindings blocked vector store preparation: %v", err)
			}
			t.Cleanup(func() {
				if shutdownErr := runtime.Shutdown(); shutdownErr != nil {
					t.Error(shutdownErr)
				}
			})
			vector, err := runtime.Embedder.Embed(context.Background(), "ingestion query")
			want := []float32{1, 0}
			if name == "global embedding" {
				want = []float32{0, 1}
			}
			if err != nil || !reflect.DeepEqual(vector, want) {
				t.Fatalf("embedding selection changed: %v, %v", vector, err)
			}
			current, err := json.Marshal(cfg.ModelBindings)
			if err != nil || string(original) != string(current) {
				t.Fatal("service preparation mutated the recipe bindings")
			}
		})
	}
}
