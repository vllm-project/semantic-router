package routerruntime

import (
	"context"
	"encoding/json"
	"errors"
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

func TestCandleBERTVectorStoreRejectsHistoricalPaddingVectors(t *testing.T) {
	ctx := context.Background()
	provider, err := embedding.NewFuncProvider(config.EmbeddingBackendCandle, 3, func(context.Context, string) ([]float32, error) {
		t.Fatal("identity resolution ran inference")
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// The runtime passes an output-width view to the identity resolver.
	view := embedding.WithOptions(provider, embedding.Options{Dimension: 3})
	cfg := &config.VectorStoreConfig{EmbeddingModel: "bert", EmbeddingDimension: 3}
	identity, err := resolveVectorStoreEmbeddingIdentity(view, cfg)
	if err != nil || identity.Fingerprint == "" {
		t.Fatalf("Candle BERT has no vector-store identity: %+v %v", identity, err)
	}

	backend := vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})
	registry := vectorstore.NewMemoryMetadataRegistry()
	old := vectorstore.NewManager(backend, registry, 3, vectorstore.BackendTypeMemory)
	store, err := old.CreateStore(ctx, vectorstore.CreateStoreRequest{Name: "padded vectors"})
	if err != nil {
		t.Fatal(err)
	}
	chunk := vectorstore.EmbeddedChunk{ID: "old", Content: "historical", Embedding: []float32{1, 0, 0}}
	if err = old.InsertChunks(ctx, store.ID, []vectorstore.EmbeddedChunk{chunk}); err != nil {
		t.Fatal(err)
	}

	current := vectorstore.NewManager(backend, registry, 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity(identity.Fingerprint))
	if err = current.LoadFromRegistry(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err = current.Search(ctx, store.ID, chunk.Embedding, 1, 0, nil); !errors.Is(err, vectorstore.ErrEmbeddingIncompatible) {
		t.Fatalf("new BERT query compared with persisted padded vector: %v", err)
	}
	if err = current.InsertChunks(ctx, store.ID, []vectorstore.EmbeddedChunk{chunk}); !errors.Is(err, vectorstore.ErrEmbeddingIncompatible) {
		t.Fatalf("new BERT vector appended to persisted padded store: %v", err)
	}

	fresh, err := current.CreateStore(ctx, vectorstore.CreateStoreRequest{Name: "unpadded vectors"})
	if err != nil {
		t.Fatal(err)
	}
	if fresh.Metadata[vectorstore.EmbeddingIdentityMetadataKey] != identity.Fingerprint {
		t.Fatal("new vector store omitted the BERT encoder identity")
	}
	if err = current.InsertChunks(ctx, fresh.ID, []vectorstore.EmbeddedChunk{chunk}); err != nil {
		t.Fatal(err)
	}
	restarted := vectorstore.NewManager(backend, registry, 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity(identity.Fingerprint))
	if err = restarted.LoadFromRegistry(ctx); err != nil {
		t.Fatal(err)
	}
	results, err := restarted.Search(ctx, fresh.ID, chunk.Embedding, 1, 0, nil)
	if err != nil || len(results) != 1 {
		t.Fatalf("matching BERT identity could not reuse new store: %v %v", results, err)
	}
	if _, err = restarted.Search(ctx, store.ID, chunk.Embedding, 1, 0, nil); !errors.Is(err, vectorstore.ErrEmbeddingIncompatible) {
		t.Fatalf("restart adopted the historical vector store: %v", err)
	}
}

func TestVectorStoreIdentityLeavesOtherBERTProvidersUntouched(t *testing.T) {
	embed := func(context.Context, string) ([]float32, error) { return []float32{1, 0, 0}, nil }
	for _, backend := range []string{"ort", config.EmbeddingBackendOpenAICompatible} {
		provider, err := embedding.NewFuncProvider(backend, 3, embed)
		if err != nil {
			t.Fatal(err)
		}
		cfg := &config.VectorStoreConfig{EmbeddingModel: "bert", EmbeddingDimension: 3}
		identity, err := resolveVectorStoreEmbeddingIdentity(provider, cfg)
		if err != nil || identity.Fingerprint != "" {
			t.Fatalf("%s BERT acquired a new namespace: %+v %v", backend, identity, err)
		}
		cfg.EmbeddingModel = "mmbert"
		if _, err = resolveVectorStoreEmbeddingIdentity(provider, cfg); !errors.Is(err, embedding.ErrIdentityUnsupported) {
			t.Fatalf("%s mmbert bypassed content identity: %v", backend, err)
		}
	}
}
