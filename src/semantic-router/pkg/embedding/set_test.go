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
