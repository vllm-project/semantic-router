package classification

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

type reaskCoverageProvider struct {
	embedding.Provider
	fits bool
	err  error
}

func (p reaskCoverageProvider) FitsInput(context.Context, string) (bool, error) {
	return p.fits, p.err
}

func TestReaskEmbeddingCoverageIsNotAFalseMatch(t *testing.T) {
	calls := 0
	base, err := embedding.NewFuncProvider("test", 2, func(context.Context, string) ([]float32, error) {
		calls++
		return []float32{1, 0}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name     string
		provider embedding.Provider
		want     error
	}{
		{"unknown", base, binding.ErrCapability},
		{"truncated", reaskCoverageProvider{Provider: base}, binding.ErrInputLimit},
		{"coverage error", reaskCoverageProvider{Provider: base, err: context.DeadlineExceeded}, context.DeadlineExceeded},
	} {
		t.Run(test.name, func(t *testing.T) {
			classifier, err := NewReaskClassifierWithProvider([]config.ReaskRule{{Name: "repeat", Threshold: .8}}, "test", test.provider)
			if err != nil {
				t.Fatal(err)
			}
			prefix := strings.Repeat("shared context ", 10000)
			matches, err := classifier.Classify(prefix+"translate it", []string{prefix + "delete it"})
			if !errors.Is(err, test.want) || matches != nil {
				t.Fatalf("matches=%v error=%v, want unknown with %v", matches, err, test.want)
			}
			matches, err = classifier.Classify(prefix+"translate it", []string{prefix + "translate it"})
			if err != nil || len(matches) != 1 || matches[0].MinSimilarity != 1 {
				t.Fatalf("complete equality: matches=%v error=%v", matches, err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if _, err := classifier.ClassifyContext(ctx, prefix, []string{prefix}); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation: %v", err)
			}
		})
	}
	if calls != 0 {
		t.Fatalf("unproven or identical inputs made %d model calls", calls)
	}
}

func TestReaskEmbeddingReceivesCompleteDifferentTails(t *testing.T) {
	prefix := strings.Repeat("shared context ", 10000)
	current, prior := prefix+"translate it", prefix+"delete it"
	seen := map[string]int{}
	provider := newTestTextProvider(func(text string) ([]float32, error) {
		seen[text]++
		if text == current {
			return []float32{1, 0}, nil
		}
		if text == prior {
			return []float32{0, 1}, nil
		}
		t.Fatal("embedding received a clipped input")
		return nil, nil
	})
	classifier, err := NewReaskClassifierWithProvider([]config.ReaskRule{{Name: "repeat", Threshold: .8}}, "test", provider)
	if err != nil {
		t.Fatal(err)
	}
	matches, err := classifier.Classify(current, []string{prior})
	if err != nil || len(matches) != 0 || seen[current] != 1 || seen[prior] != 1 {
		t.Fatalf("matches=%v error=%v complete inputs=%d", matches, err, len(seen))
	}
}
