//go:build !windows

package cache

import (
	"context"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

func TestLexicalPolarityLookupRejectsOppositeQueries(t *testing.T) {
	const stored = "How do I enable dark mode?"
	const opposite = "How do I disable dark mode?"
	const paraphrase = "How can I enable dark mode?"
	for _, hnsw := range []bool{false, true} {
		t.Run(fmt.Sprintf("hnsw=%t", hnsw), func(t *testing.T) {
			// Identical fixture vectors deliberately put both queries above threshold;
			// this exercises candidate filtering, not embedding quality.
			c := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, SimilarityThreshold: 0.8, MaxEntries: 16, UseHNSW: hnsw, EmbeddingModel: "qwen3", EmbeddingProvider: storagetest.Vectors{Size: 384, Aliases: map[string]string{opposite: stored, paraphrase: stored}}})
			t.Cleanup(func() { _ = c.Close() })
			if err := c.AddEntry(context.Background(), "entry1", "model1", stored, []byte(`{}`), []byte("ENABLE_ANSWER"), 60); err != nil {
				t.Fatal(err)
			}
			rejected, err := c.LookupSimilarWithThreshold(context.Background(), "model1", opposite, 0.8)
			if err != nil || rejected.Found || rejected.Similarity < 0.8 {
				t.Fatalf("opposite query: result=%+v err=%v", rejected, err)
			}
			hit, err := c.LookupSimilarWithThreshold(context.Background(), "model1", paraphrase, 0.8)
			if err != nil || !hit.Found || string(hit.ResponseBody) != "ENABLE_ANSWER" {
				t.Fatalf("paraphrase: result=%+v err=%v", hit, err)
			}
		})
	}
}

func TestInMemoryHitReportsNegationGuard(t *testing.T) {
	for _, tc := range negationGuardServedPairs {
		t.Run(tc.name, func(t *testing.T) {
			c := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, SimilarityThreshold: 0.8, MaxEntries: 16, EmbeddingModel: "qwen3", EmbeddingProvider: storagetest.Vectors{Size: 384, Aliases: map[string]string{tc.incoming: tc.cached}}})
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
