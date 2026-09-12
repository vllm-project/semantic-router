package modelruntime

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestOwnedRemoteEmbeddingRejectsLocalCapabilitiesBeforeProvisioning(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"index": 0, "embedding": []float32{1, 0}}}})
	}))
	defer server.Close()
	base := func() *config.RouterConfig {
		cfg := &config.RouterConfig{}
		cfg.EmbeddingConfig = config.HNSWConfig{Backend: config.EmbeddingBackendOpenAICompatible, ModelType: "remote", TargetDimension: 2, TargetLayer: 22}
		cfg.Endpoint = config.EmbeddingEndpointConfig{BaseURL: server.URL, Model: "embedding", Dimensions: 2}
		cfg.MmBertModelPath = "/unavailable/unused/catalog"
		cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "text", Candidates: []string{"candidate"}}}
		return cfg
	}
	prepared, err := PrepareOwnedEmbeddings(context.Background(), base(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	if !prepared.Ready() || prepared.Has("mmbert") {
		t.Fatal("remote mode loaded unused local catalog")
	}
	for _, test := range []struct {
		name   string
		mutate func(*config.RouterConfig)
	}{
		{"cache windows", func(cfg *config.RouterConfig) {
			cfg.SemanticCache.Enabled = true
			cfg.SemanticCache.EmbeddingModel = "mmbert"
		}},
		{"image query", func(cfg *config.RouterConfig) { cfg.EmbeddingRules[0].QueryModality = config.QueryModalityImage }},
		{"image candidates", func(cfg *config.RouterConfig) {
			cfg.ComplexityRules = []config.ComplexityRule{{Name: "image", Hard: config.ComplexityCandidates{ImageCandidates: []string{"image.png"}}}}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := base()
			test.mutate(cfg)
			before := calls.Load()
			_, prepareErr := PrepareOwnedEmbeddings(context.Background(), cfg, nil)
			if !errors.Is(prepareErr, binding.ErrCapability) {
				t.Fatalf("capability error = %v", prepareErr)
			}
			if calls.Load() != before {
				t.Fatal("unsupported candidate attempted HTTP inference")
			}
			provider, _ := prepared.Default()
			if _, callErr := provider.Embed(context.Background(), "previous generation remains available"); callErr != nil {
				t.Fatal(callErr)
			}
		})
	}
}
