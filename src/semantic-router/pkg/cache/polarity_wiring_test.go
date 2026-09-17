//go:build !windows && cgo

package cache

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

// These tests drive finishFindSimilarSearch with a hand-built winning candidate
// so the lexical polarity tier is exercised without an embedding model. They
// share the fixtures in polarity_nli_test.go; negation_regression_test.go
// covers the model-backed path.

func TestLexicalPolarityGuardRejectsAntonymSwap(t *testing.T) {
	c, entry := newPolarityTestCache(t, false)

	const query = "How do I disable two-factor authentication?"
	result, err := finishWithCandidate(c, context.Background(), query, entry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Found || result.ResponseBody != nil {
		t.Fatalf("antonym swap must be a miss, got %+v", result)
	}
	if result.Similarity != 1.0 {
		t.Fatalf("rejected candidate score must still be reported, got %.4f", result.Similarity)
	}
	if atomic.LoadInt64(&c.missCount) != 1 || atomic.LoadInt64(&c.hitCount) != 0 {
		t.Fatalf("reject must count as a miss: miss=%d hit=%d", c.missCount, c.hitCount)
	}
	if !c.entries[0].LastAccessAt.IsZero() || c.entries[0].HitCount != 0 {
		t.Fatalf("rejected candidate must not have its access info touched: %+v", c.entries[0])
	}
}

func TestLexicalPolarityGuardServesParaphrase(t *testing.T) {
	c, entry := newPolarityTestCache(t, false)

	result, err := finishWithCandidate(c, context.Background(), "How can I enable two-factor authentication?", entry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Found || string(result.ResponseBody) != "ENABLE-ANSWER" {
		t.Fatalf("paraphrase must hit, got %+v", result)
	}
	if atomic.LoadInt64(&c.hitCount) != 1 || c.entries[0].HitCount != 1 || c.entries[0].LastAccessAt.IsZero() {
		t.Fatalf("served hit must update counters and access info: hit=%d entry=%+v", c.hitCount, c.entries[0])
	}
}

// TestLexicalPolarityGuardRunsBeforeNLI pins the tier order: the model-free
// floor rejects on its own, so an obvious antonym swap never pays for an NLI
// call even when that tier is enabled.
func TestLexicalPolarityGuardRunsBeforeNLI(t *testing.T) {
	c, entry := newPolarityTestCache(t, true)
	installVerifier(t, c, func(context.Context, string, string) (float32, error) {
		t.Fatal("verifier must not run for a candidate the lexical tier already rejects")
		return 0, nil
	})

	result, err := finishWithCandidate(c, context.Background(), "How do I disable two-factor authentication?", entry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Found {
		t.Fatal("lexical rejection must stand on its own")
	}
}

// TestLexicalPolarityGuardBelowThresholdSkipsGuard keeps the guard behind the
// similarity gate: a candidate that never clears the threshold is already a
// miss and must not be tokenized.
func TestLexicalPolarityGuardBelowThresholdSkipsGuard(t *testing.T) {
	c, entry := newPolarityTestCache(t, false)

	result, err := c.finishFindSimilarSearch(context.Background(), time.Now(), entry.Model,
		"How do I disable two-factor authentication?", polarityTestThreshold, 0, entry, 0.42, 1, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Found || result.Similarity != 0.42 {
		t.Fatalf("below-threshold candidate must miss with its score reported, got %+v", result)
	}
}
