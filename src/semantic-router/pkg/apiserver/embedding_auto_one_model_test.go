//go:build !windows

package apiserver

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func nativeWidthSet(t *testing.T) *embedding.Set {
	t.Helper()
	provider := func(width int) embedding.Provider {
		p, err := embedding.NewFuncProvider("synthetic", width, func(context.Context, string) ([]float32, error) {
			vector := make([]float32, width)
			vector[0] = 1
			return vector, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	return embedding.NewSet(map[string]embedding.Provider{"qwen3": provider(1024), "mmbert": provider(768)}, "mmbert")
}

// A short text alone prefers qwen3 and a 600-word text alone prefers mmbert.
// Sent together under auto, they must come back from one model, chosen by the
// first text, or the response mixes embedding spaces and widths.
func TestOwnedAutoEmbeddingUsesOneModelPerRequest(t *testing.T) {
	short, long := "short text", strings.Repeat("word ", 600)
	for _, test := range []struct {
		name      string
		texts     []string
		model     string
		dimension int
	}{
		{"short text first", []string{short, long}, "qwen3", 1024},
		{"long text first", []string{long, short}, "mmbert", 768},
	} {
		results, _, err := buildOwnedEmbeddingResults(context.Background(), nativeWidthSet(t), EmbeddingRequest{Model: "auto", Texts: test.texts, QualityPriority: 0.9, LatencyPriority: 0.1})
		if err != nil {
			t.Fatalf("%s: %v", test.name, err)
		}
		if len(results) != len(test.texts) {
			t.Fatalf("%s: %d results, want %d", test.name, len(results), len(test.texts))
		}
		for index, result := range results {
			if result.ModelUsed != test.model || result.Dimension != test.dimension {
				t.Errorf("%s: texts[%d] used %s at %d, want %s at %d", test.name, index, result.ModelUsed, result.Dimension, test.model, test.dimension)
			}
		}
	}
}
