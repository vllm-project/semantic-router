package testcases

import (
	"strings"
	"testing"
)

func TestEvaluateCacheAssertionsRejectsZeroHits(t *testing.T) {
	results := []CacheResult{{SimilarQuestion: "miss", CacheHit: false}}
	err := evaluateCacheAssertions(results, 1, 0, nil)
	if err == nil || !strings.Contains(err.Error(), "zero cache hits") {
		t.Fatalf("expected zero-hit failure, got %v", err)
	}
}

func TestEvaluateCacheAssertionsAcceptsValidHit(t *testing.T) {
	results := []CacheResult{{SimilarQuestion: "hit", CacheHit: true, Similarity: 0.9}}
	if err := evaluateCacheAssertions(results, 1, 1, nil); err != nil {
		t.Fatalf("expected valid hit to pass, got %v", err)
	}
}

func TestParseCacheSimilarityRejectsPositiveMissScore(t *testing.T) {
	if _, msg := parseCacheSimilarity("0.8", false); msg == "" {
		t.Fatal("expected positive miss similarity to fail")
	}
}

func TestParseCacheSimilarityAcceptsZeroOrAbsentMissScore(t *testing.T) {
	for _, header := range []string{"", "0"} {
		if _, msg := parseCacheSimilarity(header, false); msg != "" {
			t.Fatalf("header %q: %s", header, msg)
		}
	}
}

func TestParseCacheSimilarityRejectsNonFiniteScores(t *testing.T) {
	tests := []struct {
		name     string
		cacheHit bool
	}{
		{name: "hit", cacheHit: true},
		{name: "miss", cacheHit: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, header := range []string{"NaN", "+Inf", "-Inf"} {
				if _, msg := parseCacheSimilarity(header, tt.cacheHit); msg == "" {
					t.Errorf("header=%q: expected non-finite similarity to fail", header)
				}
			}
		})
	}
}
