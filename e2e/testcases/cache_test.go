package testcases

import (
	"strings"
	"testing"
)

func TestEvaluateCacheAssertionsRejectsZeroRequests(t *testing.T) {
	err := evaluateCacheAssertions(nil, 0, 0, nil)
	if err == nil || !strings.Contains(err.Error(), "zero similar-question requests") {
		t.Fatalf("expected zero-request failure, got %v", err)
	}
}

func TestEvaluateCacheAssertionsReportsSetupFailures(t *testing.T) {
	err := evaluateCacheAssertions(nil, 0, 0, []string{`original question "q": connection refused`})
	if err == nil {
		t.Fatal("expected failure when every priming request failed")
	}
	if !strings.Contains(err.Error(), "connection refused") {
		t.Fatalf("setup failure detail must reach the verdict, got %v", err)
	}
}

func TestEvaluateCacheAssertionsPropagatesPerRequestErrors(t *testing.T) {
	results := []CacheResult{
		{SimilarQuestion: "good", CacheHit: true, Similarity: 0.9},
		{SimilarQuestion: "bad", CacheHit: true, Error: "cache hit missing x-vsr-cache-similarity header"},
	}
	err := evaluateCacheAssertions(results, 2, 2, nil)
	if err == nil || !strings.Contains(err.Error(), "bad") {
		t.Fatalf("expected the per-request error to fail the run, got %v", err)
	}
}

// Hit rate is a measurement, not a gate here — the repository enforces its floor
// through the "semantic-cache" acceptance contract instead.
func TestEvaluateCacheAssertionsDoesNotGateOnHitRate(t *testing.T) {
	results := []CacheResult{{SimilarQuestion: "miss", CacheHit: false}}
	if err := evaluateCacheAssertions(results, 1, 0, nil); err != nil {
		t.Fatalf("zero hits must not fail the testcase, got %v", err)
	}
}

func TestEvaluateCacheAssertionsAcceptsValidHit(t *testing.T) {
	results := []CacheResult{{SimilarQuestion: "hit", CacheHit: true, Similarity: 0.9}}
	if err := evaluateCacheAssertions(results, 1, 1, nil); err != nil {
		t.Fatalf("expected valid hit to pass, got %v", err)
	}
}

// The blocker this file exists for: a hit whose per-request score never reached
// the response surface must fail, not pass silently.
func TestParseCacheSimilarityRejectsHitWithoutHeader(t *testing.T) {
	if _, msg := parseCacheSimilarity("", true); msg == "" {
		t.Fatal("expected a hit with an absent similarity header to fail")
	}
}

func TestParseCacheSimilarityRejectsOutOfRangeHitScore(t *testing.T) {
	for _, header := range []string{"0", "0.0000", "-0.5", "1.5"} {
		if _, msg := parseCacheSimilarity(header, true); msg == "" {
			t.Errorf("header %q: expected out-of-(0,1] hit similarity to fail", header)
		}
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
