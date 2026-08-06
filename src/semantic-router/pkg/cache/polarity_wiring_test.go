//go:build !windows && cgo

package cache

import (
	"testing"
	"time"
)

// TestFinishFindSimilarSearchPolarityWiring exercises candidate selection and
// final hit/miss wiring without a model. negation_regression_test.go covers the
// model-backed path.
func TestFinishFindSimilarSearchPolarityWiring(t *testing.T) {
	const threshold = float32(0.80)
	// Identical unit vectors give a deterministic above-threshold score.
	embedding := []float32{1, 0, 0}
	const aboveThreshold = float32(1.0)

	newCacheWithEntry := func() (*InMemoryCache, CacheEntry) {
		c := NewInMemoryCache(InMemoryCacheOptions{
			SimilarityThreshold: threshold,
			MaxEntries:          16,
			TTLSeconds:          0, // keep updateAccessInfo off the expiration heap
			Enabled:             true,
			EvictionPolicy:      FIFOEvictionPolicyType,
		})
		entry := CacheEntry{
			RequestID:    "e1",
			Model:        "model-x",
			Query:        "How do I enable two-factor authentication?",
			ResponseBody: []byte("ENABLE-ANSWER"),
			Embedding:    embedding,
			Timestamp:    time.Now(),
		}
		c.entries = append(c.entries, entry)
		return c, entry
	}

	buildResult := func(c *InMemoryCache, query string, entry CacheEntry) cacheSearchResult {
		result := cacheSearchResult{bestIndex: -1}
		c.considerSearchCandidate(&result, query, threshold, 0, entry, embedding)
		return result
	}

	t.Run("polarity mismatch flips above-threshold candidate to miss", func(t *testing.T) {
		c, entry := newCacheWithEntry()
		const query = "How do I disable two-factor authentication?"
		result := buildResult(c, query, entry)
		if !result.polarityRejected {
			t.Fatalf("expected candidate diverted to polarityRejected, got %+v", result)
		}
		body, hit, err := c.finishFindSimilarSearch(time.Now(), "model-x", query, threshold, result)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if hit || body != nil {
			t.Fatalf("expected miss on polarity mismatch, got hit=%v body=%q", hit, string(body))
		}
		if got := c.LastSimilarity(); got != aboveThreshold {
			t.Errorf("LastSimilarity = %.4f, want %.4f (similarity must still be recorded on reject)", got, aboveThreshold)
		}
	})

	t.Run("genuine match above threshold is still served", func(t *testing.T) {
		c, entry := newCacheWithEntry()
		const query = "How do I enable two-factor authentication?"
		result := buildResult(c, query, entry)
		if result.polarityRejected || result.bestIndex != 0 {
			t.Fatalf("expected genuine match promoted to bestIndex 0, got %+v", result)
		}
		body, hit, err := c.finishFindSimilarSearch(time.Now(), "model-x", query, threshold, result)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !hit || string(body) != "ENABLE-ANSWER" {
			t.Fatalf("expected hit returning the cached answer, got hit=%v body=%q", hit, string(body))
		}
	})
}
