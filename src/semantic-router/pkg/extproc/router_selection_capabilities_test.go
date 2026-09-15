//go:build !windows && cgo

package extproc

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestSelectionEmbeddingUsesPreparedModels(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig = config.HNSWConfig{Backend: config.EmbeddingBackendCandle, ModelType: "  MMBERT  "}
	calls := make(map[string]int)
	vectors := map[string][]float32{"mmbert": {1, 0}, "qwen3": {0, 1}}
	providers := make(map[string]embedding.Provider)
	for model, vector := range vectors {
		provider, err := embedding.NewFuncProvider(config.EmbeddingBackendCandle, 2, func(_ context.Context, text string) ([]float32, error) {
			if text != "hello" {
				t.Fatalf("embedding text = %q, want hello", text)
			}
			calls[model]++
			return vector, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		providers[model] = provider
	}
	prepared := embedding.NewSet(providers, "mmbert")
	defer prepared.Close()
	embed, defaultConfig := resolveSelectionEmbeddingFunc(cfg, prepared)
	if defaultConfig.ModelType != "mmbert" {
		t.Fatalf("default model = %q, want mmbert", defaultConfig.ModelType)
	}
	if len(calls) != 0 {
		t.Fatal("constructing selection performed inference")
	}
	for range 2 {
		for _, request := range []selection.EmbeddingConfig{defaultConfig, {ModelType: " Qwen3 "}} {
			got, err := embed("hello", request)
			model := strings.ToLower(strings.TrimSpace(request.ModelType))
			if err != nil || !slices.Equal(got, vectors[model]) {
				t.Fatalf("prepared %s embedding = %v, %v", model, got, err)
			}
		}
	}
	if calls["mmbert"] != 2 || calls["qwen3"] != 2 {
		t.Fatalf("prepared provider calls = %v, want two per model", calls)
	}
	if _, err := embed("hello", selection.EmbeddingConfig{ModelType: "gemma"}); err == nil || !strings.Contains(err.Error(), "not prepared") {
		t.Fatalf("unprepared model error = %v", err)
	}
}

func TestSelectionEmbeddingRequiresPreparedSet(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig = config.HNSWConfig{Backend: config.EmbeddingBackendCandle, ModelType: "qwen3"}
	embed, defaultConfig := resolveSelectionEmbeddingFunc(cfg)
	if _, err := embed("hello", defaultConfig); err == nil || !strings.Contains(err.Error(), "not prepared") {
		t.Fatalf("missing prepared set error = %v", err)
	}
}
