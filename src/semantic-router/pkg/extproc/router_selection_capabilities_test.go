//go:build !windows && cgo

package extproc

import (
	"errors"
	"fmt"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSelectionEmbeddingCapabilitiesResolvedOnceDuringConstruction(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{ModelType: "unknown"},
			},
		},
	}
	queryCalls := 0
	embed, defaultConfig := resolveSelectionEmbeddingFuncWithCapabilities(
		cfg,
		[]*config.RouterConfig{cfg},
		func(modelType string) (candle_binding.EmbeddingCapabilities, error) {
			queryCalls++
			return candle_binding.EmbeddingCapabilities{}, fmt.Errorf("%w: %q", candle_binding.ErrUnsupportedModelType, modelType)
		},
	)
	if queryCalls != 1 {
		t.Fatalf("capability queries during construction = %d, want 1", queryCalls)
	}

	_, err := embed("hello", defaultConfig)
	if !errors.Is(err, candle_binding.ErrUnsupportedModelType) {
		t.Fatalf("embedding error = %v, want captured ErrUnsupportedModelType", err)
	}
	if queryCalls != 1 {
		t.Fatalf("capability queries after request = %d, want construction-time count 1", queryCalls)
	}
}

func TestSelectionEmbeddingCapabilitiesResolveRecipeMLModelsDuringConstruction(t *testing.T) {
	root := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{ModelType: "mmbert"},
			},
		},
	}
	recipe := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			ModelSelection: config.ModelSelectionConfig{
				ML: config.MLSelectionConfig{ModelType: " Qwen3 "},
			},
		},
	}
	queries := make(map[string]int)
	_, defaultConfig := resolveSelectionEmbeddingFuncWithCapabilities(
		root,
		[]*config.RouterConfig{root, recipe},
		func(modelType string) (candle_binding.EmbeddingCapabilities, error) {
			key := normalizeSelectionEmbeddingModelType(modelType)
			queries[key]++
			return candle_binding.EmbeddingCapabilities{
				ModelType:        candle_binding.ModelType(key),
				SupportsBatching: key == "qwen3",
			}, nil
		},
	)

	if defaultConfig.ModelType != "mmbert" {
		t.Fatalf("default model type = %q, want mmbert", defaultConfig.ModelType)
	}
	if queries["mmbert"] != 1 || queries["qwen3"] != 1 || len(queries) != 2 {
		t.Fatalf("construction-time capability queries = %#v, want mmbert and qwen3 once", queries)
	}
}

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
