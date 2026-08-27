//go:build !windows && cgo

package cache

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// These tests drive finishFindSimilarSearch directly with a hand-built
// candidate so the NLI polarity tier is exercised without an embedding model.
// polarity_nli_regression_test.go covers the model-backed path.

const polarityTestThreshold = float32(0.80)

func newPolarityTestCache(t *testing.T, useNLI bool) (*InMemoryCache, CacheEntry) {
	t.Helper()
	c := NewInMemoryCache(InMemoryCacheOptions{
		SimilarityThreshold: polarityTestThreshold,
		MaxEntries:          16,
		TTLSeconds:          0, // keep updateAccessInfo off the expiration heap
		Enabled:             true,
		EvictionPolicy:      FIFOEvictionPolicyType,
		PolarityGuard: PolarityGuardOptions{
			UseNLI:                 useNLI,
			ContradictionThreshold: 0.5,
		},
	})
	t.Cleanup(func() { _ = c.Close() })
	entry := CacheEntry{
		RequestID:    "e1",
		Model:        "model-x",
		Query:        "How do I enable two-factor authentication?",
		ResponseBody: []byte("ENABLE-ANSWER"),
		Embedding:    []float32{1, 0, 0},
		Timestamp:    time.Now(),
	}
	c.entries = append(c.entries, entry)
	c.entryMap[entry.RequestID] = 0
	return c, entry
}

// installVerifier swaps in a fake verifier for the test and restores the
// package state afterwards.
func installVerifier(t *testing.T, fn PolarityVerifyFunc) {
	t.Helper()
	previous := polarityVerifier
	SetPolarityVerifier(fn)
	t.Cleanup(func() { SetPolarityVerifier(previous) })
}

func finishWithCandidate(c *InMemoryCache, ctx context.Context, query string, entry CacheEntry) (LookupResult, error) {
	const aboveThreshold = float32(1.0) // identical unit vectors
	return c.finishFindSimilarSearch(ctx, time.Now(), entry.Model, query, polarityTestThreshold,
		0, entry, aboveThreshold, 1, 0)
}

func TestPolarityNLIGuardRejectsContradiction(t *testing.T) {
	c, entry := newPolarityTestCache(t, true)
	var calls int32
	var gotCached, gotIncoming string
	installVerifier(t, func(_ context.Context, cached, incoming string) (float32, error) {
		atomic.AddInt32(&calls, 1)
		gotCached, gotIncoming = cached, incoming
		return 0.97, nil
	})

	const query = "How do I disable two-factor authentication?"
	result, err := finishWithCandidate(c, context.Background(), query, entry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Found || result.ResponseBody != nil {
		t.Fatalf("contradiction must be a miss, got %+v", result)
	}
	if result.Similarity != 1.0 {
		t.Fatalf("rejected candidate score must still be reported, got %.4f", result.Similarity)
	}
	if calls != 1 {
		t.Fatalf("verifier must run exactly once per lookup, ran %d times", calls)
	}
	if gotCached != entry.Query || gotIncoming != query {
		t.Fatalf("verifier direction: premise=%q hypothesis=%q, want cached=%q incoming=%q",
			gotCached, gotIncoming, entry.Query, query)
	}
	if atomic.LoadInt64(&c.missCount) != 1 || atomic.LoadInt64(&c.hitCount) != 0 {
		t.Fatalf("reject must count as a miss: miss=%d hit=%d", c.missCount, c.hitCount)
	}
	if !c.entries[0].LastAccessAt.IsZero() || c.entries[0].HitCount != 0 {
		t.Fatalf("rejected candidate must not have its access info touched: %+v", c.entries[0])
	}
}

func TestPolarityNLIGuardServesCompatibleCandidate(t *testing.T) {
	c, entry := newPolarityTestCache(t, true)
	installVerifier(t, func(context.Context, string, string) (float32, error) { return 0.01, nil })

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

func TestPolarityNLIGuardThresholdIsExclusive(t *testing.T) {
	c, entry := newPolarityTestCache(t, true)
	installVerifier(t, func(context.Context, string, string) (float32, error) { return 0.5, nil })

	result, err := finishWithCandidate(c, context.Background(), "How can I enable 2FA?", entry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Found {
		t.Fatal("contradiction equal to the threshold must not reject (guard is strictly greater-than)")
	}
}

func TestPolarityNLIGuardDisabledNeverCallsVerifier(t *testing.T) {
	c, entry := newPolarityTestCache(t, false)
	installVerifier(t, func(context.Context, string, string) (float32, error) {
		t.Fatal("verifier must not run when the NLI tier is off")
		return 0, nil
	})

	result, err := finishWithCandidate(c, context.Background(), "How do I disable two-factor authentication?", entry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Found {
		t.Fatal("with the NLI tier off the above-threshold candidate is served (lexical tier is #2728's scope)")
	}
}

func TestPolarityNLIGuardFailsOpen(t *testing.T) {
	t.Run("verifier error serves the hit", func(t *testing.T) {
		c, entry := newPolarityTestCache(t, true)
		installVerifier(t, func(context.Context, string, string) (float32, error) {
			return 0, errors.New("nli backend unavailable")
		})
		result, err := finishWithCandidate(c, context.Background(), "How do I disable two-factor authentication?", entry)
		if err != nil {
			t.Fatalf("verifier errors must not surface to the caller: %v", err)
		}
		if !result.Found {
			t.Fatal("verifier error must fail open and serve the threshold-verified hit")
		}
	})

	t.Run("nil verifier serves the hit", func(t *testing.T) {
		c, entry := newPolarityTestCache(t, true)
		installVerifier(t, nil)
		result, err := finishWithCandidate(c, context.Background(), "How do I disable two-factor authentication?", entry)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !result.Found {
			t.Fatal("unconfigured verifier must fail open")
		}
	})
}

func TestPolarityNLIGuardHonorsCancellation(t *testing.T) {
	c, entry := newPolarityTestCache(t, true)
	installVerifier(t, func(context.Context, string, string) (float32, error) {
		t.Fatal("verifier must not run for a cancelled request")
		return 0, nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := finishWithCandidate(c, ctx, "How do I disable two-factor authentication?", entry)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got result=%+v err=%v", result, err)
	}
	if result.Found {
		t.Fatal("cancelled lookup must not serve a result")
	}
}

func TestPolarityNLIGuardBelowThresholdSkipsVerifier(t *testing.T) {
	c, entry := newPolarityTestCache(t, true)
	installVerifier(t, func(context.Context, string, string) (float32, error) {
		t.Fatal("verifier must not run for a below-threshold candidate")
		return 0, nil
	})
	result, err := c.finishFindSimilarSearch(context.Background(), time.Now(), entry.Model,
		"Something unrelated", polarityTestThreshold, 0, entry, 0.42, 1, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Found || result.Similarity != 0.42 {
		t.Fatalf("below-threshold candidate must miss with its score reported, got %+v", result)
	}
}

func TestValidateCacheConfigPolarityThreshold(t *testing.T) {
	base := GetDefaultCacheConfig()
	base.PolarityGuard = PolarityGuardOptions{UseNLI: true, ContradictionThreshold: 1.5}
	if err := ValidateCacheConfig(base); err == nil {
		t.Fatal("contradiction threshold above 1.0 must be rejected when the NLI tier is on")
	}
	base.PolarityGuard.UseNLI = false
	if err := ValidateCacheConfig(base); err != nil {
		t.Fatalf("threshold is irrelevant with the NLI tier off: %v", err)
	}
}
