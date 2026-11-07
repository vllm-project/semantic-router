package embedding

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestSelectPrefersServedModelsByPriority(t *testing.T) {
	provider, err := NewFuncProvider("test", 4, func(context.Context, string) ([]float32, error) { return make([]float32, 4), nil })
	if err != nil {
		t.Fatal(err)
	}
	both := NewSet(map[string]Provider{"mmbert": provider, "qwen3": provider}, "mmbert")
	long := strings.Repeat("word ", 600)
	for _, test := range []struct {
		name             string
		text             string
		quality, latency float32
		dimension        int
		want             string
	}{
		{"quality first", "short text", 0.9, 0.1, 0, "qwen3"},
		{"latency first", "short text", 0.5, 0.9, 0, "mmbert"},
		{"long input", long, 0.9, 0.1, 0, "mmbert"},
		{"small dimension", "short text", 0.9, 0.6, 256, "mmbert"},
	} {
		if got, err := both.Select(test.text, test.quality, test.latency, test.dimension); err != nil || got != test.want {
			t.Errorf("%s: selected %q (%v), want %q", test.name, got, err, test.want)
		}
	}
	only := NewSet(map[string]Provider{"mmbert": provider}, "mmbert")
	if got, err := only.Select("short text", 0.9, 0.1, 0); err != nil || got != "mmbert" {
		t.Errorf("only mmbert prepared: selected %q (%v)", got, err)
	}
	if _, err := NewSet(nil, "").Select("short text", 0.5, 0.5, 0); err == nil {
		t.Error("an empty set selected a model")
	}
}

func TestGetReportsAModelTheGenerationDidNotPrepare(t *testing.T) {
	_, err := NewSet(nil, "mmbert").Get("mmbert", 0, 0)
	if !errors.Is(err, ErrModelNotPrepared) || !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("Get() error = %v, want ErrModelNotPrepared and ErrCapability", err)
	}
	if want := `embedding model "mmbert" was not prepared for this generation`; !strings.Contains(err.Error(), want) {
		t.Fatalf("Get() error = %q, want it to contain %q", err, want)
	}
}

func TestSelectTreatsDimensionAsACapability(t *testing.T) {
	base, baseErr := NewFuncProvider("test", 1024, func(context.Context, string) ([]float32, error) { return make([]float32, 1024), nil })
	if baseErr != nil {
		t.Fatal(baseErr)
	}
	qwen3 := dimensionCapabilityProvider{base, []int{1024, 768, 512}}
	mmbert := dimensionCapabilityProvider{base, []int{768, 256}}
	both := NewSet(map[string]Provider{"mmbert": mmbert, "qwen3": qwen3}, "mmbert")
	long := strings.Repeat("word ", 600)
	for _, test := range []struct {
		name             string
		text             string
		quality, latency float32
		dimension        int
		want             string
	}{
		{"long input keeps mmbert when it supports the width", long, 0.9, 0.1, 768, "mmbert"},
		{"quality keeps qwen3 when it supports the width", "short text", 0.9, 0.1, 768, "qwen3"},
		{"long input falls back to qwen3 for 1024", long, 0.9, 0.1, 1024, "qwen3"},
		{"small dimension falls back to qwen3 for 512", "short text", 0.9, 0.6, 512, "qwen3"},
		{"quality falls back to mmbert for 256", "short text", 0.9, 0.1, 256, "mmbert"},
	} {
		if got, err := both.Select(test.text, test.quality, test.latency, test.dimension); err != nil || got != test.want {
			t.Errorf("%s: selected %q (%v), want %q", test.name, got, err, test.want)
		}
	}

	custom := NewSet(map[string]Provider{"mmbert": mmbert, "custom": dimensionCapabilityProvider{base, []int{2048}}}, "custom")
	if got, err := custom.Select("short text", 0.5, 0.9, 2048); err != nil || got != "custom" {
		t.Errorf("primary fallback: selected %q (%v), want custom", got, err)
	}

	// multimodal declares 384 but is not an auto text candidate: the error
	// must speak for the candidates checked, not for every prepared model.
	withMultimodal := NewSet(map[string]Provider{"mmbert": mmbert, "qwen3": qwen3, "multimodal": dimensionCapabilityProvider{base, []int{384, 256}}}, "mmbert")
	_, err := withMultimodal.Select("short text", 0.9, 0.1, 384)
	if !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("unsupported dimension error = %v, want ErrCapability", err)
	}
	for _, want := range []string{"no auto embedding candidate supports dimension 384", "mmbert supports [768 256]", "qwen3 supports [1024 768 512]"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("unsupported dimension error = %q, want it to contain %q", err, want)
		}
	}
	if strings.Contains(err.Error(), "multimodal") {
		t.Errorf("unsupported dimension error = %q, want only the checked candidates", err)
	}
}
