//go:build !windows && cgo

package extproc

import (
	"errors"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSelectionEmbeddingCapabilitiesCanonicalizeModelType(t *testing.T) {
	_, defaultConfig := resolveSelectionEmbeddingFunc(&config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{ModelType: "  MMBERT  "},
			},
		},
	})

	if defaultConfig.ModelType != "mmbert" {
		t.Fatalf("default model type = %q, want native canonical value %q", defaultConfig.ModelType, "mmbert")
	}
}

func TestSelectionEmbeddingCapabilitiesRejectUnknownModel(t *testing.T) {
	embed, defaultConfig := resolveSelectionEmbeddingFunc(&config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{ModelType: "unknown"},
			},
		},
	})

	_, err := embed("hello", defaultConfig)
	if !errors.Is(err, candle_binding.ErrUnsupportedModelType) {
		t.Fatalf("embedding error = %v, want ErrUnsupportedModelType", err)
	}
}
