//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

func TestLexicalPolarityLookupPrecedesNLI(t *testing.T) {
	const stored = "How do I enable dark mode?"
	const opposite = "How do I disable dark mode?"
	const paraphrase = "How can I enable dark mode?"
	for _, hnsw := range []bool{false, true} {
		for _, nli := range []bool{false, true} {
			t.Run(fmt.Sprintf("hnsw=%t/nli=%t", hnsw, nli), func(t *testing.T) {
				// Identical fixture vectors deliberately put both queries above threshold;
				// this exercises candidate filtering, not embedding quality.
				c := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, SimilarityThreshold: 0.8, MaxEntries: 16, UseHNSW: hnsw, EmbeddingModel: "bert", EmbeddingProvider: storagetest.Vectors{Size: 384, Aliases: map[string]string{opposite: stored, paraphrase: stored}}, PolarityGuard: PolarityGuardOptions{UseNLI: nli, ContradictionThreshold: 0.5}})
				t.Cleanup(func() { _ = c.Close() })
				calls := 0
				c.SetPolarityVerifier(func(_ context.Context, cached, incoming string) (float32, error) {
					calls++
					if cached != stored || incoming != paraphrase {
						t.Errorf("NLI received a lexically rejected candidate: cached=%q incoming=%q", cached, incoming)
					}
					return 0.01, nil
				})
				if err := c.AddEntry(context.Background(), "entry1", "model1", stored, []byte(`{}`), []byte("ENABLE_ANSWER"), 60); err != nil {
					t.Fatal(err)
				}
				rejected, err := c.LookupSimilarWithThreshold(context.Background(), "model1", opposite, 0.8)
				if err != nil || rejected.Found || rejected.Similarity < 0.8 || calls != 0 {
					t.Fatalf("opposite query: result=%+v NLI calls=%d err=%v", rejected, calls, err)
				}
				hit, err := c.LookupSimilarWithThreshold(context.Background(), "model1", paraphrase, 0.8)
				expectedCalls := 0
				if nli {
					expectedCalls = 1
				}
				if err != nil || !hit.Found || string(hit.ResponseBody) != "ENABLE_ANSWER" || calls != expectedCalls {
					t.Fatalf("paraphrase: result=%+v NLI calls=%d err=%v", hit, calls, err)
				}
			})
		}
	}
}

func TestInMemoryHitReportsNegationGuard(t *testing.T) {
	for _, tc := range negationGuardServedPairs {
		t.Run(tc.name, func(t *testing.T) {
			c := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, SimilarityThreshold: 0.8, MaxEntries: 16, EmbeddingModel: "bert", EmbeddingProvider: storagetest.Vectors{Size: 384, Aliases: map[string]string{tc.incoming: tc.cached}}})
			t.Cleanup(func() { _ = c.Close() })
			if err := c.AddEntry(context.Background(), "entry1", "model1", tc.cached, []byte(`{}`), []byte("ANSWER"), 60); err != nil {
				t.Fatal(err)
			}
			hit, err := c.LookupSimilarWithThreshold(context.Background(), "model1", tc.incoming, 0.8)
			if err != nil || !hit.Found || hit.NegationGuard != tc.want {
				t.Fatalf("lookup: found=%t negation guard=%q err=%v, want a hit reporting %q", hit.Found, hit.NegationGuard, err, tc.want)
			}
		})
	}
}
