package extproc

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

// TestSelectionEmbeddingModelTypeDefersNormalization verifies that native
// model names are preserved for the capability query, which is the single
// owner of canonicalization.
func TestSelectionEmbeddingModelTypeDefersNormalization(t *testing.T) {
	cases := []struct {
		name      string
		modelType string
		want      string
	}{
		{"mixed case", "Qwen3", "Qwen3"},
		{"padded whitespace", "  qwen3  ", "  qwen3  "},
		{"already normalized", "mmbert", "mmbert"},
		{"empty falls back to default", "", config.EmbeddingModelTypeQwen3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			models := config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{ModelType: tc.modelType},
			}
			if got := selectionEmbeddingModelType(models, config.EmbeddingBackendCandle); got != tc.want {
				t.Errorf("selectionEmbeddingModelType(%q) = %q, want %q", tc.modelType, got, tc.want)
			}
		})
	}
}

// TestBuildMLSelectionConfigDefersModelTypeNormalization keeps native
// canonicalization behind the binding capability interface.
func TestBuildMLSelectionConfigDefersModelTypeNormalization(t *testing.T) {
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
	if mlCfg.ModelType != "Qwen3" {
		t.Fatalf("ML selection model type = %q, want original value %q", mlCfg.ModelType, "Qwen3")
	}
}
