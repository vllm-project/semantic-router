//go:build !windows && cgo

package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type pawsPolarityPair struct {
	Name     string `json:"name"`
	Cached   string `json:"sentence1"`
	Incoming string `json:"sentence2"`
	Reject   bool   `json:"want_lexical_mismatch"`
}

func loadPAWSPolarityPairs(t *testing.T) []pawsPolarityPair {
	t.Helper()
	data, err := os.ReadFile("testdata/paws-polarity/pairs.json")
	require.NoError(t, err)
	var fixture struct {
		Pairs []pawsPolarityPair `json:"fixtures"`
	}
	require.NoError(t, json.Unmarshal(data, &fixture))
	require.Len(t, fixture.Pairs, 6)
	return fixture.Pairs
}

func TestPAWSDerivedPolarityCandidateLookup(t *testing.T) {
	for _, hnsw := range []bool{false, true} {
		for _, pair := range loadPAWSPolarityPairs(t) {
			t.Run(fmt.Sprintf("hnsw=%t/%s", hnsw, pair.Name), func(t *testing.T) {
				// Identical vectors deliberately make every candidate eligible.
				// The published-model regression separately measures similarity.
				cache := NewInMemoryCache(InMemoryCacheOptions{
					Enabled: true, SimilarityThreshold: 0.8, MaxEntries: 4, TTLSeconds: 60,
					UseHNSW: hnsw, EmbeddingModel: "bert",
					EmbeddingProvider: storagetest.Vectors{Size: 384, Aliases: map[string]string{pair.Incoming: pair.Cached}},
				})
				t.Cleanup(func() { _ = cache.Close() })
				assertPAWSPolarityLookup(t, cache, pair)
			})
		}
	}
}

// Called by the existing strict published-model inventory, so this does not
// introduce a silently skipped model test into ordinary core validation.
func runPAWSPolarityModelRegression(t *testing.T, provider embedding.Provider) {
	t.Helper()
	for _, pair := range loadPAWSPolarityPairs(t) {
		t.Run(pair.Name, func(t *testing.T) {
			cache := newMmbertCache(t, provider)
			t.Cleanup(func() { _ = cache.Close() })
			assertPAWSPolarityLookup(t, cache, pair)
		})
	}
}

func assertPAWSPolarityLookup(t *testing.T, cache *InMemoryCache, pair pawsPolarityPair) {
	t.Helper()
	require.NoError(t, cache.AddEntry(context.Background(), "paws", "model-a", pair.Cached, []byte(`{}`), []byte("SOURCE_ANSWER"), 60))
	result, err := cache.LookupSimilarWithThreshold(context.Background(), "model-a", pair.Incoming, negationRegressionThreshold)
	require.NoError(t, err)
	t.Logf("similarity=%.6f threshold=%.2f found=%t", result.Similarity, negationRegressionThreshold, result.Found)
	require.GreaterOrEqual(t, result.Similarity, float32(negationRegressionThreshold), "below-threshold result does not exercise the candidate guard or paraphrase recall")
	require.Equal(t, !pair.Reject, result.Found)
	if pair.Reject {
		require.Empty(t, result.ResponseBody)
	} else {
		require.Equal(t, "SOURCE_ANSWER", string(result.ResponseBody))
	}
}
