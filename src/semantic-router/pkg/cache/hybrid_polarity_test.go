//go:build !windows && cgo

package cache

import (
	"context"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

func TestHybridPolarityLookupAndService(t *testing.T) {
	const query = "enable logging for this service"
	const opposite = "disable logging for this service"
	const paraphrase = "turn on service logging"
	for _, throughService := range []bool{false, true} {
		for _, tc := range []struct {
			name         string
			storedQuery  string
			fallback     string
			wantResponse string
		}{
			{"antonym miss", opposite, "", ""},
			{"negation miss", "do not enable logging for this service", "", ""},
			{"missing original query", "", "", ""},
			{"paraphrase hit", paraphrase, "", "HNSW_ANSWER"},
			{"rejected HNSW uses valid Milvus candidate", opposite, paraphrase, "FALLBACK_ANSWER"},
			{"rejected HNSW and rejected Milvus stay miss", opposite, opposite, ""},
		} {
			name := "direct/" + tc.name
			if throughService {
				name = "service/" + tc.name
			}
			t.Run(name, func(t *testing.T) {
				provider := storagetest.Vectors{Size: 384, Aliases: map[string]string{opposite: query, paraphrase: query}}
				storedAt := time.Now().Truncate(time.Second)
				expiresAt := storedAt.Add(time.Hour)
				fields := func(stored, body string) client.ResultSet {
					// Deliberately change column order: neither IDs nor body contents
					// may substitute for the named original query.
					return client.ResultSet{
						entity.NewColumnVarChar("response_body", []string{body}),
						entity.NewColumnVarChar("query", []string{stored}),
						entity.NewColumnInt64("expires_at", []int64{expiresAt.Unix()}),
						entity.NewColumnInt64("timestamp", []int64{storedAt.Unix()}),
					}
				}
				expectedModel := "tenant-a"
				searchCalls := 0
				milvus := &MilvusCache{
					enabled: true, config: milvusCacheTestConfig("Strong"),
					embeddingModel: "bert", embeddingProvider: provider,
					queryByIDFn: func(_ context.Context, id, model string) (client.ResultSet, error) {
						require.Equal(t, "candidate-id", id)
						require.Equal(t, expectedModel, model)
						return fields(tc.storedQuery, "HNSW_ANSWER"), nil
					},
					searchFn: func(_ context.Context, model string, _ []float32) ([]client.SearchResult, error) {
						searchCalls++
						require.Equal(t, expectedModel, model)
						if tc.fallback == "" {
							return nil, nil
						}
						return []client.SearchResult{{ResultCount: 1, Scores: []float32{0.93}, Fields: fields(tc.fallback, "FALLBACK_ANSWER")}}, nil
					},
				}
				hybrid := newTestHybridCache(1)
				hybrid.milvusCache = milvus
				vector, err := milvus.getEmbedding(context.Background(), query)
				require.NoError(t, err)
				addToMemoryIndexForTest(hybrid, "candidate-id", vector)
				if throughService {
					service := NewResponseCacheService(NewLegacyBackendAdapter(hybrid, HybridCacheType), ResponseCacheServiceOptions{L1MaxEntries: -1})
					identity := CacheIdentity{Partition: CachePartition{RequestModel: "tenant-a", Protocol: "openai:body"}, SemanticQuery: query}
					expectedModel = service.ResolveIdentity(identity).SemanticPartitionKey()
					result, err := service.LookupSemantic(context.Background(), SemanticLookup{Identity: identity, Threshold: 0.8})
					require.NoError(t, err)
					require.Equal(t, tc.wantResponse != "", result.Found)
					require.Equal(t, tc.wantResponse, string(result.ResponseBody))
					require.GreaterOrEqual(t, result.Similarity, float32(0.8))
					if result.Found {
						require.Equal(t, HitKindSemantic, result.HitKind)
						require.Equal(t, CacheSourceL2, result.Source)
						require.True(t, result.AgeKnown)
						require.Equal(t, expiresAt, result.ExpiresAt)
					} else {
						require.Equal(t, HitKindMiss, result.HitKind)
					}
				} else {
					result, err := hybrid.LookupSimilarWithThreshold(context.Background(), expectedModel, query, 0.8)
					require.NoError(t, err)
					require.Equal(t, tc.wantResponse != "", result.Found)
					require.Equal(t, tc.wantResponse, string(result.ResponseBody))
					require.GreaterOrEqual(t, result.Similarity, float32(0.8))
					if result.Found {
						require.Equal(t, storedAt, result.StoredAt)
						require.Equal(t, expiresAt, result.ExpiresAt)
						wantSimilarity := 1.0
						if tc.fallback != "" {
							wantSimilarity = 0.93
						}
						require.InDelta(t, wantSimilarity, result.Similarity, 1e-5)
					}
				}
				if tc.wantResponse == "HNSW_ANSWER" {
					require.Zero(t, searchCalls)
				} else {
					require.Equal(t, 1, searchCalls)
				}
				stats := hybrid.GetStats()
				if tc.wantResponse == "" {
					require.Zero(t, stats.HitCount)
					require.EqualValues(t, 1, stats.MissCount)
				} else {
					require.EqualValues(t, 1, stats.HitCount)
					require.Zero(t, stats.MissCount)
				}
			})
		}
	}
}

func TestHybridPolarityMissPreservesNegativeScore(t *testing.T) {
	const query = "enable logging"
	cfg := milvusCacheTestConfig("Strong")
	cfg.Collection.VectorField.MetricType = "IP"
	milvus := &MilvusCache{
		enabled: true, config: cfg, embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(),
		queryByIDFn: func(context.Context, string, string) (client.ResultSet, error) {
			// The nearest HNSW record belongs to a different model partition.
			return nil, nil
		},
		searchFn: func(context.Context, string, []float32) ([]client.SearchResult, error) {
			return []client.SearchResult{{ResultCount: 1, Scores: []float32{-0.25}}}, nil
		},
	}
	hybrid := newTestHybridCache(1)
	hybrid.milvusCache = milvus
	vector, err := milvus.getEmbedding(context.Background(), query)
	require.NoError(t, err)
	addToMemoryIndexForTest(hybrid, "other-model", vector)
	result, err := hybrid.LookupSimilarWithThreshold(context.Background(), "requested-model", query, 0.8)
	require.NoError(t, err)
	require.False(t, result.Found)
	require.Equal(t, float32(-0.25), result.Similarity)
}
