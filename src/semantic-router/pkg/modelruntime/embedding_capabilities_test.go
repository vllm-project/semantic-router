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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

type audioCapabilityProvider struct {
	embedding.Provider
	info embedding.ModelInfo
}

func (p audioCapabilityProvider) EmbeddingInfo() embedding.ModelInfo { return p.info }

func TestOriginalAudioRequirementRejectsFeatureOnlyEncoder(t *testing.T) {
	provider := audioCapabilityProvider{info: embedding.ModelInfo{Modalities: []string{"text", "audio"}}}
	requirements := []config.EmbeddingRequirement{{Model: "multimodal", Consumer: "audio signal", Modality: "audio"}}
	providers := map[string]embedding.Provider{"multimodal": provider}
	base, err := embedding.NewFuncProvider("ort", 384, func(context.Context, string) ([]float32, error) {
		t.Fatal("capability validation must precede inference")
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	provider.Provider = base
	providers["multimodal"] = provider
	if err := validatePreparedEmbeddings(context.Background(), requirements, providers); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("feature-only audio encoder accepted for original PCM: %v", err)
	}
	provider.info.Audio = &binding.AudioCapability{MaxSeconds: 30, MaxChannels: 8}
	providers["multimodal"] = provider
	if err := validatePreparedEmbeddings(context.Background(), requirements, providers); err != nil {
		t.Fatalf("original-PCM capability rejected: %v", err)
	}
}

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
		{"audio query", func(cfg *config.RouterConfig) { cfg.EmbeddingRules[0].QueryModality = config.QueryModalityAudio }},
		{"image query", func(cfg *config.RouterConfig) { cfg.EmbeddingRules[0].QueryModality = config.QueryModalityImage }},
		{"image candidates", func(cfg *config.RouterConfig) {
			cfg.ComplexityRules = []config.ComplexityRule{{Name: "image", Hard: config.ComplexityCandidates{ImageCandidates: []string{"image.png"}}}}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := base()
			test.mutate(cfg)
			before := calls.Load()
			prepare := PrepareOwnedEmbeddings
			if test.name == "cache windows" {
				prepare = PrepareOwnedResponseCacheEmbeddings
			}
			_, prepareErr := prepare(context.Background(), cfg, nil)
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
