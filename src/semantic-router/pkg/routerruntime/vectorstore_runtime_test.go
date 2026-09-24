package routerruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
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

func TestVectorStoreResolvesPreparedOmniWidthBeforeSelectingView(t *testing.T) {
	for _, size := range []int{384, 768} {
		provider, _ := embedding.NewFuncProvider("owned-test", size, func(context.Context, string) ([]float32, error) {
			t.Fatal("preparation ran inference")
			return nil, nil
		})
		prepared := embedding.NewSet(map[string]embedding.Provider{"multimodal": provider}, "multimodal")
		for _, requested := range []int{0, size} {
			cfg := &config.VectorStoreConfig{EmbeddingModel: "multimodal", EmbeddingDimension: requested}
			cfg.ApplyDefaults()
			selected, err := prepareVectorStoreEmbedding(prepared, cfg)
			if err != nil || cfg.EmbeddingDimension != size || selected.Dimension() != size {
				t.Fatalf("size %d requested %d: %v", size, requested, err)
			}
		}
		for _, requested := range []int{-1, 128} {
			cfg := &config.VectorStoreConfig{EmbeddingModel: "multimodal", EmbeddingDimension: requested}
			if _, err := prepareVectorStoreEmbedding(prepared, cfg); err == nil {
				t.Fatal("unchecked option view accepted an unsupported dimension")
			}
		}
	}
}

func TestVectorStoreRuntimeUsesEndpointWidthForStorage(t *testing.T) {
	for _, size := range []int{384, 768} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Input []string `json:"input"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
					return
				}
				values := make([]float32, size)
				values[0] = 1
				data := make([]map[string]any, len(request.Input))
				for i := range data {
					data[i] = map[string]any{"index": i, "embedding": values}
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
			}))
			defer server.Close()
			makeConfig := func(dimension int) *config.RouterConfig {
				cfg := &config.RouterConfig{VectorStore: &config.VectorStoreConfig{Enabled: true, BackendType: "memory", EmbeddingModel: "bert", EmbeddingDimension: dimension, FileStorageDir: filepath.Join(t.TempDir(), "files"), MetadataStore: "memory"}}
				cfg.EmbeddingConfig = config.HNSWConfig{Backend: config.EmbeddingBackendOpenAICompatible, ModelType: "bert", TargetDimension: size}
				cfg.Endpoint = config.EmbeddingEndpointConfig{BaseURL: server.URL, Model: "prepared", Dimensions: size}
				return cfg
			}
			cfg := makeConfig(0)
			runtime, err := NewVectorStoreRuntime(cfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if shutdownErr := runtime.Shutdown(); shutdownErr != nil {
					t.Error(shutdownErr)
				}
			})
			if cfg.VectorStore.EmbeddingDimension != size || runtime.Embedder.Dimension() != size {
				t.Fatalf("default width did not resolve to %d", size)
			}
			if _, createErr := runtime.Manager.CreateStore(context.Background(), vectorstore.CreateStoreRequest{Name: "actual-width"}); createErr != nil {
				t.Fatal(createErr)
			}
			values, err := runtime.Embedder.Embed(context.Background(), "document")
			if err != nil || len(values) != size {
				t.Fatalf("full output lost: %d %v", len(values), err)
			}
			bad := makeConfig(128)
			if _, err := NewVectorStoreRuntime(bad); err == nil {
				t.Fatal("explicit mismatch accepted")
			}
			if _, err := os.Stat(bad.VectorStore.FileStorageDir); !os.IsNotExist(err) {
				t.Fatal("invalid width opened storage before rejecting")
			}
		})
	}
}
