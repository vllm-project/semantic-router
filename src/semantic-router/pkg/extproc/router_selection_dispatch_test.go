package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
)

func TestSelectionUsesPreparedProviderForEveryLocalBackend(t *testing.T) {
	for _, backend := range []string{"candle", "ort", "openvino"} {
		t.Run(backend, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.EmbeddingConfig = config.HNSWConfig{Backend: backend, ModelType: "multimodal"}
			calls := 0
			provider, err := embedding.NewFuncProvider(backend, 768, func(context.Context, string) ([]float32, error) {
				calls++
				return make([]float32, 768), nil
			})
			if err != nil {
				t.Fatal(err)
			}
			set := embedding.NewSet(map[string]embedding.Provider{"multimodal": provider}, "multimodal")
			embed, options := resolveSelectionEmbeddingFunc(cfg, set)
			vector, err := embed(context.Background(), "query", options)
			if err != nil || len(vector) != 768 || calls != 1 {
				t.Fatalf("selection bypassed its prepared provider: dimension=%d calls=%d err=%v", len(vector), calls, err)
			}
		})
	}
}

func TestSelectionEmbeddingRuntimeUsesRequestedRemoteConfig(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Fatalf("request path = %q, want /v1/embeddings", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"index": 0, "embedding": []float64{0.1, 0.2}},
			},
		}); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{
					Backend:         config.EmbeddingBackendOpenAICompatible,
					ModelType:       config.EmbeddingModelTypeRemote,
					TargetDimension: 2,
				},
				Endpoint: config.EmbeddingEndpointConfig{
					BaseURL: server.URL + "/v1",
					Model:   "BAAI/bge-m3",
				},
			},
		},
	}
	cfg.ModelSelection.Enabled = true
	cfg.ModelSelection.ML.ModelsPath = "test-model-selection"
	cfg.ModelSelection.ML.ModelType = config.EmbeddingModelTypeRemote
	cfg.Decisions = []config.Decision{{Name: "nearest", Algorithm: &config.AlgorithmConfig{Type: "knn"}}}
	prepared, err := modelruntime.PrepareOwnedEmbeddings(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	embed, defaultConfig := resolveSelectionEmbeddingFunc(cfg, prepared)

	embedding, err := embed(context.Background(), "hello", defaultConfig)
	if err != nil {
		t.Fatalf("selection embedding function error = %v", err)
	}
	if len(embedding) != 2 || embedding[0] != float32(0.1) {
		t.Fatalf("embedding = %#v, want two remote values", embedding)
	}
}

func TestSelectionEmbeddingPropagatesCancellationContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})

	provider, err := embedding.NewFuncProvider("test", 1, func(got context.Context, _ string) ([]float32, error) {
		if got != ctx {
			t.Errorf("embedding context = %p, want request context %p", got, ctx)
		}
		close(started)
		<-got.Done()
		return nil, got.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	set := embedding.NewSet(map[string]embedding.Provider{"test": provider}, "test")
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "test"}
	embed, options := resolveSelectionEmbeddingFunc(cfg, set)

	errCh := make(chan error, 1)
	go func() {
		_, err := embed(ctx, "cancel me", options)
		errCh <- err
	}()

	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("embedding provider was not called")
	}
	cancel()

	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("embedding error = %v, want context cancellation", err)
		}
	case <-time.After(time.Second):
		t.Fatal("embedding did not stop after request cancellation")
	}
}

func TestBuildModelSelectionConfigCarriesMLModelRequest(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{
					ModelType:       "mmbert",
					TargetDimension: 768,
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			ModelSelection: config.ModelSelectionConfig{
				ML: config.MLSelectionConfig{
					ModelsPath:   "models/ml-selection",
					ModelType:    config.EmbeddingModelTypeQwen3,
					EmbeddingDim: 1024,
				},
			},
		},
	}

	mlCfg := buildModelSelectionConfig(cfg).ML
	if mlCfg.ModelType != config.EmbeddingModelTypeQwen3 {
		t.Fatalf("ML selection embedding model = %q, want %q", mlCfg.ModelType, config.EmbeddingModelTypeQwen3)
	}
	if mlCfg.EmbeddingDim != 1024 {
		t.Fatalf("ML selection embedding dimension = %d, want 1024", mlCfg.EmbeddingDim)
	}
	if cfg.EmbeddingConfig.ModelType != "mmbert" {
		t.Fatalf("default embedding model = %q, want mmbert", cfg.EmbeddingConfig.ModelType)
	}
	if cfg.EmbeddingConfig.TargetDimension != 768 {
		t.Fatalf("default embedding dimension = %d, want 768", cfg.EmbeddingConfig.TargetDimension)
	}
}

func TestLegacyMLDimensionDoesNotSelectAnotherModel(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{
					ModelType:       "mmbert",
					TargetDimension: 768,
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			ModelSelection: config.ModelSelectionConfig{
				ML: config.MLSelectionConfig{ModelsPath: "models/ml-selection", EmbeddingDim: 1024},
			},
		},
	}

	mlCfg := buildModelSelectionConfig(cfg).ML
	if mlCfg.ModelType != "" {
		t.Fatalf("legacy embedding_dim selected model %q, want factory default", mlCfg.ModelType)
	}
	if mlCfg.EmbeddingDim != 1024 {
		t.Fatalf("legacy embedding dimension = %d, want 1024", mlCfg.EmbeddingDim)
	}
}

func TestQwenMLRequestUsesModelDefaultDimension(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{
					ModelType:       "mmbert",
					TargetDimension: 768,
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			ModelSelection: config.ModelSelectionConfig{
				ML: config.MLSelectionConfig{
					ModelsPath: "models/ml-selection",
					ModelType:  config.EmbeddingModelTypeQwen3,
				},
			},
		},
	}

	mlCfg := buildModelSelectionConfig(cfg).ML
	if mlCfg.ModelType != config.EmbeddingModelTypeQwen3 || mlCfg.EmbeddingDim != 0 {
		t.Fatalf("ML selection embedding config = %s/%d, want Qwen3/0 (model-native dimension)", mlCfg.ModelType, mlCfg.EmbeddingDim)
	}
}

// TestSelectionEmbeddingModelTypeNormalizesCase guards against a configured
// modelType (e.g. "Qwen3") passing validation case-insensitively but then
// reaching the embedding provider lookup unnormalized: the prepared providers
// are keyed by the normalized name, so the lookup would miss.
func TestSelectionEmbeddingModelTypeNormalizesCase(t *testing.T) {
	cases := []struct {
		name      string
		modelType string
		want      string
	}{
		{"mixed case", "Qwen3", "qwen3"},
		{"padded whitespace", "  qwen3  ", "qwen3"},
		{"already normalized", "mmbert", "mmbert"},
		{"empty falls back to default", "", config.EmbeddingModelTypeQwen3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			models := config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{ModelType: tc.modelType},
			}
			if got := selectionEmbeddingModelType(models, config.EmbeddingBackendModelRuntime); got != tc.want {
				t.Errorf("selectionEmbeddingModelType(%q) = %q, want %q", tc.modelType, got, tc.want)
			}
		})
	}
}

// TestBuildMLSelectionConfigNormalizesModelTypeCase guards the same
// unnormalized-modelType bug as TestSelectionEmbeddingModelTypeNormalizesCase,
// but on the sibling ml.model_type path: nothing validates or rewrites it, so
// it reaches factory.go's mlEmbeddingConfig -- and the same provider lookup --
// independently of the default embedding model type.
func TestBuildMLSelectionConfigNormalizesModelTypeCase(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			ModelSelection: config.ModelSelectionConfig{
				ML: config.MLSelectionConfig{
					ModelsPath: "models/ml-selection",
					ModelType:  "Qwen3",
				},
			},
		},
	}

	mlCfg := buildModelSelectionConfig(cfg).ML
	if mlCfg.ModelType != config.EmbeddingModelTypeQwen3 {
		t.Fatalf("ML selection model type = %q, want %q", mlCfg.ModelType, config.EmbeddingModelTypeQwen3)
	}
}
