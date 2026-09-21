//go:build !windows && cgo

package cache

import (
	"context"
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

	// Exercise the candidate-selection seam used by both search paths.
	buildResult := func(c *InMemoryCache, query string, entry CacheEntry) cacheSearchResult {
		result := cacheSearchResult{bestIndex: -1}
		c.considerSearchCandidate(&result, tokenizeForPolarity(query, nil), threshold, 0, entry, embedding)
		return result
	}

	t.Run("polarity mismatch flips above-threshold candidate to miss", func(t *testing.T) {
		c, entry := newCacheWithEntry()
		const query = "How do I disable two-factor authentication?"
		result := buildResult(c, query, entry)
		if !result.polarityRejected {
			t.Fatalf("expected candidate diverted to polarityRejected, got %+v", result)
		}
		lookup, err := c.finishFindSimilarSearch(
			context.Background(), time.Now(), "model-x", query, threshold, result,
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if lookup.Found || lookup.ResponseBody != nil {
			t.Fatalf("expected miss on polarity mismatch, got %+v", lookup)
		}
		if lookup.Similarity != aboveThreshold {
			t.Errorf("Similarity = %.4f, want %.4f (similarity must still be recorded on reject)", lookup.Similarity, aboveThreshold)
		}
	})

	t.Run("genuine match above threshold is still served", func(t *testing.T) {
		c, entry := newCacheWithEntry()
		const query = "How do I enable two-factor authentication?"
		result := buildResult(c, query, entry)
		if result.polarityRejected || result.bestIndex != 0 {
			t.Fatalf("expected genuine match promoted to bestIndex 0, got %+v", result)
		}
		lookup, err := c.finishFindSimilarSearch(
			context.Background(), time.Now(), "model-x", query, threshold, result,
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !lookup.Found || string(lookup.ResponseBody) != "ENABLE-ANSWER" {
			t.Fatalf("expected hit returning the cached answer, got %+v", lookup)
		}
	})
}
