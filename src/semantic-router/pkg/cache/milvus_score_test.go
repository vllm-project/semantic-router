package cache

import (
	"context"
	"errors"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMilvusScoreToSimilarity(t *testing.T) {
	tests := []struct {
		name       string
		metricType string
		score      float32
		want       float32
	}{
		// Non-L2 scores are similarities already and must pass through.
		{name: "IP passes through", metricType: "IP", score: 3.7, want: 3.7},
		{name: "COSINE passes through", metricType: "COSINE", score: -1.0, want: -1.0},
		{name: "unknown metric passes through", metricType: "HAMMING", score: 0.3, want: 0.3},
		{name: "empty metric passes through", metricType: "", score: 0.3, want: 0.3},

		// L2 is a squared distance and must be inverted into a similarity.
		{name: "L2 identical vectors", metricType: "L2", score: 0.0, want: 1.0},
		{name: "L2 close vectors", metricType: "L2", score: 0.25, want: 0.8},
		{name: "L2 distant vectors", metricType: "L2", score: 4.0, want: 0.2},
		{name: "L2 lowercase", metricType: "l2", score: 0.25, want: 0.8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, milvusScoreToSimilarity(tt.metricType, tt.score), 1e-6)
		})
	}
}

// The #2716 regression shape: an L2 exact match must hit, a distant one must miss.
func TestMilvusScoreToSimilarity_L2OrderingInverted(t *testing.T) {
	threshold := float32(0.8)

	exact := milvusScoreToSimilarity("L2", 0.0)
	near := milvusScoreToSimilarity("L2", 0.1)
	far := milvusScoreToSimilarity("L2", 1.9)

	assert.GreaterOrEqual(t, exact, threshold, "exact match must be a hit")
	assert.Greater(t, exact, near, "smaller distance must map to larger similarity")
	assert.Greater(t, near, far, "smaller distance must map to larger similarity")
	assert.Less(t, far, threshold, "distant match must be a miss")
}

func TestNormalizeMilvusMetricType(t *testing.T) {
	assert.Equal(t, "IP", normalizeMilvusMetricType(""), "empty defaults to IP")
	assert.Equal(t, "L2", normalizeMilvusMetricType("l2"))
	assert.Equal(t, "COSINE", normalizeMilvusMetricType("cosine"))
	assert.Equal(t, "IP", normalizeMilvusMetricType("IP"))
}

// fakeMilvusSearchClient stubs Search; the embedded nil interface covers the rest.
type fakeMilvusSearchClient struct {
	client.Client
	results   []client.SearchResult
	gotMetric entity.MetricType
}

func (f *fakeMilvusSearchClient) Search(ctx context.Context, collName string, partitions []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
	f.gotMetric = metricType
	return f.results, nil
}

func milvusCacheWithFakeSearch(metricType string, results []client.SearchResult) (*MilvusCache, *fakeMilvusSearchClient) {
	fake := &fakeMilvusSearchClient{results: results}
	cfg := &config.MilvusConfig{}
	// Decoy sibling field: passing it to the converter would fail the tests.
	cfg.Collection.VectorField.Name = "embedding"
	cfg.Collection.VectorField.MetricType = metricType
	cfg.Search.Params.Ef = 8
	return &MilvusCache{enabled: true, client: fake, config: cfg}, fake
}

func TestSearchDocumentsConvertsL2ScoresBeforeThreshold(t *testing.T) {
	results := []client.SearchResult{{
		ResultCount: 2,
		Scores:      []float32{0.25, 4.0}, // squared L2 distances: close, far
		Fields:      client.ResultSet{entity.NewColumnVarChar("content", []string{"close doc", "far doc"})},
	}}
	cache, fake := milvusCacheWithFakeSearch("L2", results)

	contents, scores, err := cache.SearchDocuments(context.Background(), "kb", []float32{0.1}, 0.5, 5, "", "content", "", "", 0)

	require.NoError(t, err)
	assert.Equal(t, entity.MetricType("L2"), fake.gotMetric)
	assert.Equal(t, []string{"close doc"}, contents, "only the close document clears a 0.5 similarity threshold")
	require.Len(t, scores, 1)
	assert.InDelta(t, 0.8, scores[0], 1e-6, "returned scores are similarities, not raw distances")
}

func TestSearchDocumentsNormalizesCallerMetricType(t *testing.T) {
	results := []client.SearchResult{{
		ResultCount: 1,
		Scores:      []float32{0.25},
		Fields:      client.ResultSet{entity.NewColumnVarChar("content", []string{"doc"})},
	}}
	cache, fake := milvusCacheWithFakeSearch("IP", results)

	_, scores, err := cache.SearchDocuments(context.Background(), "kb", []float32{0.1}, 0.5, 5, "", "content", "", "l2", 0)

	require.NoError(t, err)
	assert.Equal(t, entity.MetricType("L2"), fake.gotMetric, "caller-supplied metric is canonicalized before the search")
	require.Len(t, scores, 1)
	assert.InDelta(t, 0.8, scores[0], 1e-6)
}

func TestSearchDocumentsSurfacesResultError(t *testing.T) {
	results := []client.SearchResult{{
		ResultCount: 1,
		Scores:      []float32{0.0},
		Err:         errors.New("field content not found"),
	}}
	cache, _ := milvusCacheWithFakeSearch("L2", results)

	_, _, err := cache.SearchDocuments(context.Background(), "kb", []float32{0.1}, 0.5, 5, "", "content", "", "", 0)

	assert.ErrorContains(t, err, "field content not found")
}

// Err can coexist with ResultCount == 0 and must not read as "no results".
func TestSearchDocumentsSurfacesErrorWithZeroResults(t *testing.T) {
	results := []client.SearchResult{{
		ResultCount: 0,
		Err:         errors.New("collection not loaded"),
	}}
	cache, _ := milvusCacheWithFakeSearch("L2", results)

	_, _, err := cache.SearchDocuments(context.Background(), "kb", []float32{0.1}, 0.5, 5, "", "content", "", "", 0)

	assert.ErrorContains(t, err, "collection not loaded")
}
