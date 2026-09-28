package cache

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type remotePolarityCandidate struct {
	query    string
	response string
	score    float32
}

func milvusPolarityFixture(candidates []remotePolarityCandidate) client.SearchResult {
	queries, responses := []string{}, []string{}
	scores := []float32{}
	timestamps, expiries := []int64{}, []int64{}
	for i, candidate := range candidates {
		queries = append(queries, candidate.query)
		responses = append(responses, candidate.response)
		scores = append(scores, candidate.score)
		timestamps = append(timestamps, 100+int64(i))
		expiries = append(expiries, 200+int64(i))
	}
	return client.SearchResult{ResultCount: len(candidates), Scores: scores, Fields: client.ResultSet{
		entity.NewColumnVarChar("request_id", []string{"ordinary-id", "second-id"}),
		entity.NewColumnVarChar("query", queries),
		entity.NewColumnVarChar("response_body", responses),
		entity.NewColumnInt64("timestamp", timestamps), entity.NewColumnInt64("expires_at", expiries),
	}}
}

func qdrantPolarityFixture(candidates []remotePolarityCandidate) []*qdrant.ScoredPoint {
	points := make([]*qdrant.ScoredPoint, 0, len(candidates))
	for i, candidate := range candidates {
		points = append(points, &qdrant.ScoredPoint{Score: candidate.score, Payload: qdrant.NewValueMap(map[string]any{
			"query": candidate.query, "response_body": candidate.response,
			"timestamp": 100 + int64(i), "expires_at": 200 + int64(i),
		})})
	}
	return points
}

func TestRemoteVectorCachePolarityCandidates(t *testing.T) {
	const query = "enable logging for this service"
	cases := []struct {
		name          string
		candidates    []remotePolarityCandidate
		wantBody      string
		wantScore     float32
		wantTimestamp int64
	}{
		{"negation rejects", []remotePolarityCandidate{{"do not enable logging for this service", "opposite", .99}}, "", .99, 0},
		{"antonym rejects", []remotePolarityCandidate{{"disable logging for this service", "opposite", .99}}, "", .99, 0},
		{"blank query rejects", []remotePolarityCandidate{{"  ", "unknown", .99}}, "", .99, 0},
		{"missing query rejects", []remotePolarityCandidate{{"", "unknown", .99}}, "", .99, 0},
		{"non-finite scores reject", []remotePolarityCandidate{{query, "invalid", float32(math.NaN())}, {query, "invalid", float32(math.Inf(1))}, {query, "invalid", float32(math.Inf(-1))}}, "", 0, 0},
		{"non-finite score fallback", []remotePolarityCandidate{{query, "invalid", float32(math.NaN())}, {query, "correct", .95}}, "correct", .95, 101},
		{"negative miss score retained", []remotePolarityCandidate{{query, "low", -.5}}, "", -.5, 0},
		{"ordinary paraphrase hits", []remotePolarityCandidate{{"turn on service logging", "correct", .95}}, "correct", .95, 100},
		{"valid fetched fallback", []remotePolarityCandidate{{"disable logging for this service", "opposite", .99}, {"enable logging for this service", "correct", .92}}, "correct", .92, 101},
		{"unordered candidates choose best", []remotePolarityCandidate{{query, "lower", .85}, {query, "correct", .95}}, "correct", .95, 101},
		{"missing query fallback", []remotePolarityCandidate{{"", "unknown", .99}, {"turn on service logging", "correct", .92}}, "correct", .92, 101},
		{"missing response fallback", []remotePolarityCandidate{{query, "", .99}, {query, "correct", .92}}, "correct", .92, 101},
		{"below threshold fallback rejected", []remotePolarityCandidate{{"disable logging for this service", "opposite", .99}, {query, "low", .70}}, "", .99, 0},
	}
	for _, backend := range []string{"milvus", "qdrant"} {
		t.Run(backend, func(t *testing.T) {
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					var lookup func(context.Context, string, string, float32) (LookupResult, error)
					if backend == "milvus" {
						c := &MilvusCache{enabled: true, config: milvusCacheTestConfig("Strong"), embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(_ context.Context, model string, _ []float32) ([]client.SearchResult, error) {
							require.Equal(t, "tenant-a", model)
							return []client.SearchResult{milvusPolarityFixture(tc.candidates)}, nil
						}}
						lookup = c.LookupSimilarWithThreshold
					} else {
						c := &QdrantCache{enabled: true, cfg: &config.QdrantConfig{}, embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(_ context.Context, request *qdrant.QueryPoints) ([]*qdrant.ScoredPoint, error) {
							require.True(t, request.GetWithPayload().GetEnable())
							require.Equal(t, float32(.8), request.GetScoreThreshold())
							require.Equal(t, "tenant-a", request.Filter.Must[0].GetField().GetMatch().GetKeyword())
							return qdrantPolarityFixture(tc.candidates), nil
						}}
						lookup = c.LookupSimilarWithThreshold
					}
					result, err := lookup(context.Background(), "tenant-a", query, .8)
					require.NoError(t, err)
					assert.Equal(t, tc.wantBody != "", result.Found)
					assert.Equal(t, tc.wantBody, string(result.ResponseBody))
					assert.InDelta(t, tc.wantScore, result.Similarity, 1e-6)
					if tc.wantTimestamp > 0 {
						assert.Equal(t, time.Unix(tc.wantTimestamp, 0), result.StoredAt)
						assert.Equal(t, time.Unix(tc.wantTimestamp+100, 0), result.ExpiresAt)
					}
				})
			}
		})
	}
}

func TestRemoteVectorCacheHitReportsNegationGuard(t *testing.T) {
	for _, tc := range negationGuardServedPairs {
		candidates := []remotePolarityCandidate{{tc.cached, "ANSWER", .95}}
		milvus := &MilvusCache{enabled: true, config: milvusCacheTestConfig("Strong"), embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(context.Context, string, []float32) ([]client.SearchResult, error) {
			return []client.SearchResult{milvusPolarityFixture(candidates)}, nil
		}}
		qdrantCache := &QdrantCache{enabled: true, cfg: &config.QdrantConfig{}, embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(context.Context, *qdrant.QueryPoints) ([]*qdrant.ScoredPoint, error) {
			return qdrantPolarityFixture(candidates), nil
		}}
		for backend, lookup := range map[string]func(context.Context, string, string, float32) (LookupResult, error){
			"milvus": milvus.LookupSimilarWithThreshold,
			"qdrant": qdrantCache.LookupSimilarWithThreshold,
		} {
			t.Run(backend+"/"+tc.name, func(t *testing.T) {
				result, err := lookup(context.Background(), "tenant-a", tc.incoming, .8)
				require.NoError(t, err)
				require.True(t, result.Found)
				assert.Equal(t, tc.want, result.NegationGuard)
			})
		}
	}
}

func TestMilvusResponseColumnUsesName(t *testing.T) {
	const body = "0123456789abcdef0123456789abcdef"
	c := &MilvusCache{enabled: true, queryByIDFn: func(context.Context, string, string) (client.ResultSet, error) {
		return client.ResultSet{
			entity.NewColumnVarChar("query", []string{"enable service logging"}),
			entity.NewColumnVarChar("response_body", []string{body}),
			entity.NewColumnVarChar("request_id", []string{"ordinary-id"}),
		}, nil
	}}
	got, err := c.GetByID(context.Background(), "ordinary-id", "tenant-a")
	require.NoError(t, err)
	assert.Equal(t, body, string(got))
}

type remoteSearchContext struct {
	context.Context
	failure error
}

func (c *remoteSearchContext) Err() error { return c.failure }

func TestRemoteVectorCacheSearchErrors(t *testing.T) {
	for _, backend := range []string{"milvus", "qdrant"} {
		for _, failure := range []error{errors.New("storage unavailable"), context.DeadlineExceeded, context.Canceled} {
			t.Run(backend+"/"+failure.Error(), func(t *testing.T) {
				ctx := &remoteSearchContext{Context: context.Background()}
				searchFailure := func() error {
					// Cancellation happens during the remote read, after embedding succeeds.
					if errors.Is(failure, context.Canceled) || errors.Is(failure, context.DeadlineExceeded) {
						ctx.failure = failure
					}
					return failure
				}
				var lookup func(context.Context, string, string, float32) (LookupResult, error)
				if backend == "milvus" {
					c := &MilvusCache{enabled: true, config: milvusCacheTestConfig("Strong"), embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(context.Context, string, []float32) ([]client.SearchResult, error) { return nil, searchFailure() }}
					lookup = c.LookupSimilarWithThreshold
				} else {
					c := &QdrantCache{enabled: true, cfg: &config.QdrantConfig{}, embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(context.Context, *qdrant.QueryPoints) ([]*qdrant.ScoredPoint, error) { return nil, searchFailure() }}
					lookup = c.LookupSimilarWithThreshold
				}
				got, err := lookup(ctx, "model", "enable logging", .8)
				assert.False(t, got.Found)
				if errors.Is(failure, context.Canceled) || errors.Is(failure, context.DeadlineExceeded) {
					assert.ErrorIs(t, err, failure)
				} else {
					require.NoError(t, err)
				}
			})
		}
	}
}

func TestRemoteVectorCachePolarityThroughService(t *testing.T) {
	identity := CacheIdentity{Partition: CachePartition{RequestModel: "tenant-a", Protocol: "openai:body"}, SemanticQuery: "enable logging for this service"}
	for _, backend := range []CacheBackendType{MilvusCacheType, QdrantCacheType} {
		for _, fallback := range []bool{false, true} {
			name := string(backend) + "/opposite miss"
			if fallback {
				name = string(backend) + "/valid fallback hit"
			}
			t.Run(name, func(t *testing.T) {
				candidates := []remotePolarityCandidate{{"disable logging for this service", "opposite", .99}}
				if fallback {
					candidates = append(candidates, remotePolarityCandidate{identity.SemanticQuery, "correct", .92})
				}
				var cache CacheBackend
				var expectedPartition string
				if backend == MilvusCacheType {
					cache = &MilvusCache{enabled: true, config: milvusCacheTestConfig("Strong"), embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(_ context.Context, model string, _ []float32) ([]client.SearchResult, error) {
						require.Equal(t, expectedPartition, model)
						return []client.SearchResult{milvusPolarityFixture(candidates)}, nil
					}}
				} else {
					cache = &QdrantCache{enabled: true, cfg: &config.QdrantConfig{}, embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(_ context.Context, request *qdrant.QueryPoints) ([]*qdrant.ScoredPoint, error) {
						require.Equal(t, expectedPartition, request.Filter.Must[0].GetField().GetMatch().GetKeyword())
						return qdrantPolarityFixture(candidates), nil
					}}
				}
				adapter := NewLegacyBackendAdapter(cache, backend).WithEmbeddingProvider(cacheTestEmbeddingProvider())
				service := NewResponseCacheService(adapter, ResponseCacheServiceOptions{L1MaxEntries: -1})
				expectedPartition = service.ResolveIdentity(identity).SemanticPartitionKey()
				result, err := service.LookupSemantic(context.Background(), SemanticLookup{Identity: identity, Threshold: .8})
				require.NoError(t, err)
				assert.Equal(t, fallback, result.Found)
				stats := cache.GetStats()
				if fallback {
					assert.Equal(t, "correct", string(result.ResponseBody))
					assert.Equal(t, CacheSourceL2, result.Source)
					assert.True(t, result.AgeKnown)
					assert.Equal(t, time.Unix(201, 0), result.ExpiresAt)
					assert.Equal(t, int64(1), stats.HitCount)
					assert.Zero(t, stats.MissCount)
				} else {
					assert.Empty(t, result.ResponseBody)
					assert.Equal(t, int64(1), stats.MissCount)
					assert.Zero(t, stats.HitCount)
				}
			})
		}
	}
}

type namedMilvusClient struct {
	client.Client
	fields       []string
	expression   string
	result       client.ResultSet
	searchResult []client.SearchResult
}

func (f *namedMilvusClient) Query(_ context.Context, _ string, _ []string, expr string, fields []string, _ ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
	f.fields, f.expression = fields, expr
	return f.result, nil
}

func (f *namedMilvusClient) Search(_ context.Context, _ string, _ []string, expr string, fields []string, _ []entity.Vector, _ string, _ entity.MetricType, _ int, _ entity.SearchParam, _ ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
	f.fields, f.expression = fields, expr
	return f.searchResult, nil
}

func TestMilvusTypedEntryIncludesQueryAndAge(t *testing.T) {
	now := time.Now().Unix()
	fake := &namedMilvusClient{result: client.ResultSet{
		entity.NewColumnVarChar("query", []string{"enable logging"}),
		entity.NewColumnVarChar("response_body", []string{"answer"}),
		entity.NewColumnInt64("timestamp", []int64{now - 30}),
		entity.NewColumnInt64("expires_at", []int64{now + 30}),
	}}
	c := &MilvusCache{enabled: true, client: fake, config: milvusCacheTestConfig("Strong")}
	result, err := c.getEntryByID(context.Background(), "ordinary-id", "tenant-a")
	require.NoError(t, err)
	assert.Equal(t, "enable logging", result.Query)
	assert.Equal(t, "answer", string(result.ResponseBody))
	assert.Equal(t, time.Unix(now-30, 0), result.Timestamp)
	assert.Equal(t, time.Unix(now+30, 0), result.ExpiresAt)
	assert.Contains(t, fake.fields, "query")
	assert.Contains(t, fake.fields, "response_body")
	assert.Contains(t, fake.expression, `model == "tenant-a"`)
	assert.Contains(t, fake.expression, `expires_at == 0 || expires_at > `)
}

func TestMilvusTypedEntryMissingResponse(t *testing.T) {
	for _, fields := range []client.ResultSet{nil, {entity.NewColumnVarChar("query", []string{"enable logging"})}, {entity.NewColumnVarChar("response_body", []string{""})}, {entity.NewColumnInt64("response_body", []int64{4})}} {
		c := &MilvusCache{enabled: true, queryByIDFn: func(context.Context, string, string) (client.ResultSet, error) { return fields, nil }}
		_, err := c.getEntryByID(context.Background(), "ordinary-id", "model")
		assert.ErrorIs(t, err, errMilvusCacheEntryNotFound)
	}
}

func TestMilvusNamedSemanticFieldsRequested(t *testing.T) {
	fake := &namedMilvusClient{}
	c := &MilvusCache{enabled: true, client: fake, config: milvusCacheTestConfig("Strong")}
	_, err := c.milvusSearchSimilarVectors(context.Background(), "tenant-a", []float32{1})
	require.NoError(t, err)
	assert.Contains(t, fake.fields, "query")
	assert.Contains(t, fake.fields, "response_body")
	assert.Contains(t, fake.expression, `model == "tenant-a"`)
	assert.Contains(t, fake.expression, `expires_at == 0 || expires_at > `)
}

func TestQdrantSemanticCandidateBound(t *testing.T) {
	c := &QdrantCache{enabled: true, cfg: &config.QdrantConfig{}, embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(_ context.Context, request *qdrant.QueryPoints) ([]*qdrant.ScoredPoint, error) {
		require.Equal(t, uint64(5), request.GetLimit())
		return nil, nil
	}}
	_, err := c.LookupSimilarWithThreshold(context.Background(), "tenant-a", "enable logging", .8)
	require.NoError(t, err)
}

func TestMilvusTypedEntryExpiredRecordIsNotFound(t *testing.T) {
	c := &MilvusCache{enabled: true, queryByIDFn: func(context.Context, string, string) (client.ResultSet, error) {
		return client.ResultSet{entity.NewColumnVarChar("query", []string{"enable logging"}), entity.NewColumnVarChar("response_body", []string{"old"}), entity.NewColumnInt64("expires_at", []int64{time.Now().Add(-time.Second).Unix()})}, nil
	}}
	_, err := c.getEntryByID(context.Background(), "old", "model")
	require.ErrorIs(t, err, errMilvusCacheEntryNotFound)
}

func TestRemoteVectorCacheMissingQueryMetadata(t *testing.T) {
	for _, backend := range []string{"milvus", "qdrant"} {
		for _, wrongType := range []bool{false, true} {
			name := backend + "/missing field"
			if wrongType {
				name = backend + "/non-string field"
			}
			t.Run(name, func(t *testing.T) {
				var lookup func(context.Context, string, string, float32) (LookupResult, error)
				if backend == "milvus" {
					result := milvusPolarityFixture([]remotePolarityCandidate{{"enable logging", "unknown", .99}})
					result.Fields = client.ResultSet{entity.NewColumnVarChar("response_body", []string{"unknown"})}
					if wrongType {
						result.Fields = append(result.Fields, entity.NewColumnInt64("query", []int64{1}))
					}
					c := &MilvusCache{enabled: true, config: milvusCacheTestConfig("Strong"), embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(context.Context, string, []float32) ([]client.SearchResult, error) {
						return []client.SearchResult{result}, nil
					}}
					lookup = c.LookupSimilarWithThreshold
				} else {
					points := qdrantPolarityFixture([]remotePolarityCandidate{{"enable logging", "unknown", .99}})
					delete(points[0].Payload, "query")
					if wrongType {
						points[0].Payload["query"] = qdrant.NewValueInt(1)
					}
					c := &QdrantCache{enabled: true, cfg: &config.QdrantConfig{}, embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(context.Context, *qdrant.QueryPoints) ([]*qdrant.ScoredPoint, error) { return points, nil }}
					lookup = c.LookupSimilarWithThreshold
				}
				result, err := lookup(context.Background(), "model", "enable logging", .8)
				require.NoError(t, err)
				require.False(t, result.Found)
				require.Empty(t, result.ResponseBody)
			})
		}
	}
}

func TestMilvusNamedSemanticL2Fallback(t *testing.T) {
	cfg := milvusCacheTestConfig("Strong")
	cfg.Collection.VectorField.MetricType = "L2"
	result := milvusPolarityFixture([]remotePolarityCandidate{{"disable logging", "opposite", .01}, {"enable logging", "correct", .25}})
	c := &MilvusCache{enabled: true, config: cfg, embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(), searchFn: func(context.Context, string, []float32) ([]client.SearchResult, error) {
		return []client.SearchResult{result}, nil
	}}
	got, err := c.LookupSimilarWithThreshold(context.Background(), "model", "enable logging", .75)
	require.NoError(t, err)
	require.True(t, got.Found)
	require.Equal(t, "correct", string(got.ResponseBody))
	require.InDelta(t, .8, got.Similarity, 1e-6)
}
