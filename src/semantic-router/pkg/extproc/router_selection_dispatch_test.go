package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

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

	embed, defaultConfig := resolveSelectionEmbeddingFunc(&config.RouterConfig{
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
	})

	embedding, err := embed("hello", defaultConfig)
	if err != nil {
		t.Fatalf("selection embedding function error = %v", err)
	}
	if len(embedding) != 2 || embedding[0] != float32(0.1) {
		t.Fatalf("embedding = %#v, want two remote values", embedding)
	}
}

func TestBuildModelSelectionConfigUsesDecisionScopedMLConfig(t *testing.T) {
	cfg := mlSelectionRouterConfig(config.HNSWConfig{
		ModelType:       "mmbert",
		TargetDimension: 768,
	}, config.EmbeddingModelTypeQwen3, 1024)

	mlCfg := buildModelSelectionConfig(cfg).ML
	if mlCfg == nil {
		t.Fatal("ML selection config is nil")
	}
	if mlCfg.ModelsPath != "models/ml-selection" {
		t.Fatalf("ML selection models path = %q, want models/ml-selection", mlCfg.ModelsPath)
	}
	if mlCfg.ModelType != config.EmbeddingModelTypeQwen3 {
		t.Fatalf("ML selection model type = %q, want %q", mlCfg.ModelType, config.EmbeddingModelTypeQwen3)
	}
	if mlCfg.EmbeddingDim != 1024 {
		t.Fatalf("ML selection artifact dimension = %d, want 1024", mlCfg.EmbeddingDim)
	}
	if mlCfg.KNN == nil || mlCfg.KNN.K != 5 {
		t.Fatalf("ML selection KNN config = %#v, want k=5", mlCfg.KNN)
	}
}

func TestMLSelectorEmbeddingDimensionOverridesSemanticDefault(t *testing.T) {
	cfg := mlSelectionRouterConfig(config.HNSWConfig{
		ModelType:       "mmbert",
		TargetDimension: 768,
	}, "", 1024)

	_, defaultConfig := resolveSelectionEmbeddingFunc(cfg)
	if defaultConfig.ModelType != "mmbert" {
		t.Fatalf("selection embedding model = %q, want mmbert", defaultConfig.ModelType)
	}
	if defaultConfig.TargetDimension != 768 {
		t.Fatalf("selection embedding dimension = %d, want 768", defaultConfig.TargetDimension)
	}
	mlCfg := buildModelSelectionConfig(cfg)
	if mlCfg.ML.ModelType != "" {
		t.Fatal("decision-scoped ML config unexpectedly overrides the semantic embedding catalog")
	}
	// Use an empty training selector here: artifact loading is covered by the
	// configuration test above, while this test exercises the embedding seam.
	mlCfg.ML.ModelsPath = ""

	var requested selection.EmbeddingConfig
	registry := selection.NewFactory(mlCfg).
		WithEmbeddingFunc(func(_ string, embeddingConfig selection.EmbeddingConfig) ([]float32, error) {
			requested = embeddingConfig
			return []float32{0.1}, nil
		}, defaultConfig).
		CreateAll()
	mlSelector, ok := registry.Get(selection.MethodKNN)
	if !ok {
		t.Fatal("KNN selector was not registered")
	}
	_, _ = mlSelector.Select(context.Background(), &selection.SelectionContext{
		Query:           "route this request",
		CandidateModels: []config.ModelRef{{Model: "local/small"}, {Model: "hosted/frontier"}},
	})
	want := selection.EmbeddingConfig{ModelType: "mmbert", TargetDimension: 1024}
	if requested != want {
		t.Fatalf("ML selector requested embedding config = %#v, want %#v", requested, want)
	}
}

func TestMLSelectorModelTypeOverridesSemanticDefault(t *testing.T) {
	cfg := mlSelectionRouterConfig(config.HNSWConfig{
		ModelType:       "mmbert",
		TargetDimension: 768,
	}, config.EmbeddingModelTypeQwen3, 0)
	mlCfg := buildModelSelectionConfig(cfg)
	mlCfg.ML.ModelsPath = ""

	var requested selection.EmbeddingConfig
	registry := selection.NewFactory(mlCfg).
		WithEmbeddingFunc(func(_ string, embeddingConfig selection.EmbeddingConfig) ([]float32, error) {
			requested = embeddingConfig
			return []float32{0.1}, nil
		}, selection.EmbeddingConfig{ModelType: "mmbert", TargetDimension: 768}).
		CreateAll()
	mlSelector, ok := registry.Get(selection.MethodKNN)
	if !ok {
		t.Fatal("KNN selector was not registered")
	}
	_, _ = mlSelector.Select(context.Background(), &selection.SelectionContext{
		Query:           "route this request",
		CandidateModels: []config.ModelRef{{Model: "local/small"}, {Model: "hosted/frontier"}},
	})
	want := selection.EmbeddingConfig{ModelType: config.EmbeddingModelTypeQwen3}
	if requested != want {
		t.Fatalf("ML selector requested embedding config = %#v, want %#v", requested, want)
	}
}

func TestQwenSemanticEmbeddingUsesModelDefaultDimension(t *testing.T) {
	cfg := mlSelectionRouterConfig(config.HNSWConfig{
		ModelType: config.EmbeddingModelTypeQwen3,
	}, "", 1024)

	_, defaultConfig := resolveSelectionEmbeddingFunc(cfg)
	if defaultConfig.ModelType != config.EmbeddingModelTypeQwen3 || defaultConfig.TargetDimension != 1024 {
		t.Fatalf(
			"selection embedding config = %s/%d, want Qwen3/1024",
			defaultConfig.ModelType,
			defaultConfig.TargetDimension,
		)
	}
}

func mlSelectionRouterConfig(embeddingConfig config.HNSWConfig, modelType string, artifactDimension int) *config.RouterConfig {
	return &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{EmbeddingConfig: embeddingConfig},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name: "math",
					Algorithm: &config.AlgorithmConfig{
						Type: config.DecisionAlgorithmKNN,
						ML: &config.MLSelectionConfig{
							ModelsPath:   "models/ml-selection",
							ModelType:    modelType,
							EmbeddingDim: artifactDimension,
							KNN:          &config.MLKNNConfig{K: 5},
						},
					},
				},
			},
		},
	}
}
