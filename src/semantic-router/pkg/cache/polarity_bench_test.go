//go:build !windows && cgo

package cache

import (
	"context"
	"testing"
	"time"
)

// Benchmarks run through finishFindSimilarSearch — the production lookup
// finisher — rather than calling the guard directly, so the reported cost is
// what a cache hit actually pays: one tokenization of the incoming query and
// one comparison against the single winning candidate.

func benchmarkPolarityGuard(b *testing.B, cachedQuery, incomingQuery string) {
	b.Helper()
	c := NewInMemoryCache(InMemoryCacheOptions{
		SimilarityThreshold: polarityTestThreshold,
		MaxEntries:          16,
		TTLSeconds:          0,
		Enabled:             true,
		EvictionPolicy:      FIFOEvictionPolicyType,
	})
	b.Cleanup(func() { _ = c.Close() })
	entry := CacheEntry{
		RequestID:    "e1",
		Model:        "model-x",
		Query:        cachedQuery,
		ResponseBody: []byte("ANSWER"),
		Embedding:    []float32{1, 0, 0},
		Timestamp:    time.Now(),
	}
	c.entries = append(c.entries, entry)
	c.entryMap[entry.RequestID] = 0

	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.finishFindSimilarSearch(ctx, time.Now(), entry.Model, incomingQuery,
			polarityTestThreshold, 0, entry, 1.0, 1, 0); err != nil {
			b.Fatalf("lookup failed: %v", err)
		}
	}
}

// BenchmarkPolarityGuardServesHit is the common case: the guard passes and the
// candidate is served, so this is the cost the tier adds to every cache hit.
func BenchmarkPolarityGuardServesHit(b *testing.B) {
	benchmarkPolarityGuard(b,
		"How do I enable two-factor authentication?",
		"How can I enable two-factor authentication?")
}

// BenchmarkPolarityGuardRejects covers the rejecting path, which walks both
// diffs and the antonym table before turning the hit into a miss.
func BenchmarkPolarityGuardRejects(b *testing.B) {
	benchmarkPolarityGuard(b,
		"How do I enable two-factor authentication?",
		"How do I disable two-factor authentication?")
}

// BenchmarkPolarityGuardLongQuery bounds the worst case the tier can be handed:
// tokenization is linear in the query, and the router admits requests far
// larger than a typical question.
func BenchmarkPolarityGuardLongQuery(b *testing.B) {
	long := "Given the deployment topology described above, including the sidecar " +
		"configuration, the upstream timeouts, and the retry budget for each route, " +
		"how do I enable two-factor authentication for the administrative console?"
	benchmarkPolarityGuard(b, long, long+" Please include the rollback steps.")
}

// BenchmarkPolarityTierAlone isolates what this tier adds to a lookup, so the
// production-path numbers above can be read as guard cost plus the finisher's
// own logging and metrics.
func BenchmarkPolarityTierAlone(b *testing.B) {
	const cached = "How do I enable two-factor authentication?"
	for _, bc := range []struct {
		name     string
		incoming string
	}{
		{"serves", "How can I enable two-factor authentication?"},
		{"rejects", "How do I disable two-factor authentication?"},
	} {
		b.Run(bc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				polarityMismatch(bc.incoming, cached)
			}
		})
	}
}
