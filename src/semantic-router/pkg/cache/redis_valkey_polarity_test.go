//go:build !windows && cgo && !riscv64

package cache

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type remotePolarityDocument struct {
	query    string
	response string
	distance string
}

func TestRedisValkeySemanticPolarityCandidates(t *testing.T) {
	const incoming = "How do I enable dark mode?"
	tests := []struct {
		name  string
		docs  []remotePolarityDocument
		body  string
		score float32
	}{
		{"antonym rejected", []remotePolarityDocument{{"How do I disable dark mode?", "OPPOSITE", "0.02"}}, "", .99},
		{"negation rejected", []remotePolarityDocument{{"How do I not enable dark mode?", "OPPOSITE", "0.02"}}, "", .99},
		{"paraphrase retained", []remotePolarityDocument{{"How can I enable dark mode?", "PARAPHRASE", "0.10"}}, "PARAPHRASE", .95},
		{"lower eligible candidate", []remotePolarityDocument{{"How do I disable dark mode?", "OPPOSITE", "0.02"}, {"How can I enable dark mode?", "PARAPHRASE", "0.10"}}, "PARAPHRASE", .95},
		{"unordered candidates", []remotePolarityDocument{{"How can I enable dark mode?", "LOWER", "0.20"}, {incoming, "BEST", "0.04"}}, "BEST", .98},
		{"missing query", []remotePolarityDocument{{"", "UNKNOWN", "0.02"}}, "", .99},
		{"blank query", []remotePolarityDocument{{"  ", "UNKNOWN", "0.02"}}, "", .99},
		{"missing response", []remotePolarityDocument{{incoming, "", "0.02"}}, "", .99},
		{"missing metadata does not hide valid candidate", []remotePolarityDocument{{"", "UNKNOWN", "0.02"}, {"How can I enable dark mode?", "PARAPHRASE", "0.10"}}, "PARAPHRASE", .95},
		{"below threshold stays miss", []remotePolarityDocument{{incoming, "LOW", "0.80"}}, "", .60},
		{"malformed distance stays miss", []remotePolarityDocument{{incoming, "UNKNOWN", "not-a-distance"}}, "", 0},
	}
	for _, backend := range []string{"redis", "valkey"} {
		for _, tc := range tests {
			t.Run(backend+"/"+tc.name, func(t *testing.T) {
				cache := remotePolarityFixture(t, backend, tc.docs, nil)
				result, err := cache.LookupSimilarWithThreshold(context.Background(), "recipe::model", incoming, .8)
				require.NoError(t, err)
				require.Equal(t, tc.body != "", result.Found)
				require.Equal(t, tc.body, string(result.ResponseBody))
				require.InDelta(t, tc.score, result.Similarity, .0001)
				if result.Found {
					require.True(t, result.AgeKnown)
					require.WithinDuration(t, time.Now().Add(-time.Minute), result.StoredAt, 2*time.Second)
					require.WithinDuration(t, time.Now().Add(4*time.Minute), result.ExpiresAt, 2*time.Second)
				}
				stats := cache.GetStats()
				require.Equal(t, int64(boolInt(result.Found)), stats.HitCount)
				require.Equal(t, int64(boolInt(!result.Found)), stats.MissCount)
			})
		}
	}
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func remotePolarityFixture(t *testing.T, backend string, docs []remotePolarityDocument, searchErr error, partition ...string) LegacyCacheBackend {
	t.Helper()
	expectedPartition := func() string {
		if len(partition) > 0 {
			return partition[0]
		}
		return "recipe::model"
	}
	timestamp := fmt.Sprint(time.Now().Add(-time.Minute).Unix())
	if backend == "redis" {
		cfg := &config.RedisConfig{}
		cfg.Index.VectorField.Name = "embedding"
		cfg.Index.VectorField.MetricType = "COSINE"
		cfg.Search.TopK = 4
		return &RedisCache{
			enabled: true, config: cfg, embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(),
			searchFn: func(_ context.Context, _ string, query string, options *redis.FTSearchOptions) (redis.FTSearchResult, error) {
				require.Equal(t, partitionedKNNQuery(expectedPartition(), 4, "embedding"), query)
				var result redis.FTSearchResult
				result.Total = len(docs)
				for _, doc := range docs {
					result.Docs = append(result.Docs, redis.Document{Fields: map[string]string{
						"query": doc.query, "response_body": doc.response, "vector_distance": doc.distance,
						"timestamp": timestamp, "ttl_seconds": "300",
					}})
				}
				return result, searchErr
			},
		}
	}
	cfg := &config.ValkeyConfig{}
	cfg.Index.VectorField.Name = "embedding"
	cfg.Index.VectorField.MetricType = "COSINE"
	cfg.Search.TopK = 4
	return &ValkeyCache{
		enabled: true, config: cfg, embeddingModel: "bert", embeddingProvider: cacheTestEmbeddingProvider(),
		searchFn: func(_ context.Context, command []string) (any, error) {
			require.Equal(t, partitionedKNNQuery(expectedPartition(), 4, "embedding"), command[2])
			values := map[string]interface{}{}
			for i, doc := range docs {
				values[fmt.Sprintf("doc:%d", i)] = map[string]interface{}{
					"query": doc.query, "response_body": doc.response, "vector_distance": doc.distance,
					"timestamp": timestamp, "ttl_seconds": "300",
				}
			}
			return []interface{}{int64(len(docs)), values}, searchErr
		},
	}
}

func TestRedisValkeyServicePolarity(t *testing.T) {
	for _, backend := range []string{"redis", "valkey"} {
		for _, fallback := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/fallback=%t", backend, fallback), func(t *testing.T) {
				docs := []remotePolarityDocument{{"How do I disable dark mode?", "OPPOSITE", "0.02"}}
				if fallback {
					docs = append(docs, remotePolarityDocument{"How can I enable dark mode?", "PARAPHRASE", "0.10"})
				}
				partition := []string{""}
				cache := remotePolarityFixture(t, backend, docs, nil, partition...)
				service := NewResponseCacheService(NewLegacyBackendAdapter(cache, CacheBackendType(backend)), DefaultResponseCacheServiceOptions())
				identity := CacheIdentity{
					Partition:     CachePartition{RequestModel: "recipe::model", Protocol: "openai:body"},
					SemanticQuery: "How do I enable dark mode?",
				}
				partition[0] = service.ResolveIdentity(identity).SemanticPartitionKey()
				result, err := service.LookupSemantic(context.Background(), SemanticLookup{Identity: identity, Threshold: .8})
				require.NoError(t, err)
				require.Equal(t, fallback, result.Found)
				if fallback {
					require.Equal(t, "PARAPHRASE", string(result.ResponseBody))
					require.InDelta(t, .95, result.Similarity, .0001)
				} else {
					require.Empty(t, result.ResponseBody)
					require.InDelta(t, .99, result.Similarity, .0001)
				}
			})
		}
	}
}

// Fixed vectors exercise the actual store/index/lookup contract, not model
// semantic quality. Every query deliberately clears the configured threshold.
func redisValkeyPolarityVectors() storagetest.Vectors {
	return storagetest.Vectors{Size: 384, Aliases: map[string]string{
		"How do I disable dark mode?":    "How do I enable dark mode?",
		"How do I not enable dark mode?": "How do I enable dark mode?",
		"How can I enable dark mode?":    "How do I enable dark mode?",
	}}
}

func runRedisValkeyStoredPolarityCases(t *testing.T, backend LegacyCacheBackend, backendType CacheBackendType) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	service := NewResponseCacheService(NewLegacyBackendAdapter(backend, backendType), DefaultResponseCacheServiceOptions())
	for i, opposite := range []string{"How do I disable dark mode?", "How do I not enable dark mode?"} {
		t.Run(fmt.Sprintf("polarity_%d", i), func(t *testing.T) {
			identity := CacheIdentity{Partition: CachePartition{RequestModel: fmt.Sprintf("polarity-%d", i), Protocol: "openai:body"}, SemanticQuery: opposite}
			partition := service.ResolveIdentity(identity).SemanticPartitionKey()
			require.NoError(t, service.StoreSemantic(ctx, CacheWrite{Identity: identity, RequestID: partition + "-opposite", RequestBody: []byte(`{}`), ResponseBody: []byte("OPPOSITE"), TTL: TTL(time.Minute)}))
			require.Eventually(t, func() bool {
				hit, lookupErr := backend.LookupSimilarWithThreshold(ctx, partition, opposite, .8)
				return lookupErr == nil && hit.Found && string(hit.ResponseBody) == "OPPOSITE"
			}, 5*time.Second, 25*time.Millisecond, "stored query must be searchable before the negative control")
			miss, err := backend.LookupSimilarWithThreshold(ctx, partition, "How do I enable dark mode?", .8)
			require.NoError(t, err)
			require.False(t, miss.Found)
			require.Empty(t, miss.ResponseBody)
			require.GreaterOrEqual(t, miss.Similarity, float32(.8))
			identity.SemanticQuery = "How do I enable dark mode?"
			publicMiss, err := service.LookupSemantic(ctx, SemanticLookup{Identity: identity, Threshold: .8})
			require.NoError(t, err)
			require.False(t, publicMiss.Found)
			require.Empty(t, publicMiss.ResponseBody)
			identity.SemanticQuery = "How can I enable dark mode?"
			require.NoError(t, service.StoreSemantic(ctx, CacheWrite{Identity: identity, RequestID: partition + "-valid", RequestBody: []byte(`{}`), ResponseBody: []byte("PARAPHRASE"), TTL: TTL(time.Minute)}))
			require.Eventually(t, func() bool {
				hit, lookupErr := backend.LookupSimilarWithThreshold(ctx, partition, "How do I enable dark mode?", .8)
				return lookupErr == nil && hit.Found && string(hit.ResponseBody) == "PARAPHRASE"
			}, 5*time.Second, 25*time.Millisecond, "a valid fetched paraphrase must survive the opposite candidate")
			body, found, err := backend.FindSimilarWithThreshold(partition, "How do I enable dark mode?", .8)
			require.NoError(t, err)
			require.True(t, found)
			require.Equal(t, "PARAPHRASE", string(body))
			identity.SemanticQuery = "How do I enable dark mode?"
			publicHit, err := service.LookupSemantic(ctx, SemanticLookup{Identity: identity, Threshold: .8})
			require.NoError(t, err)
			require.True(t, publicHit.Found)
			require.Equal(t, "PARAPHRASE", string(publicHit.ResponseBody))
		})
	}
}

func TestRedisValkeySemanticPolaritySearchContract(t *testing.T) {
	for _, backend := range []string{"redis", "valkey"} {
		t.Run(backend, func(t *testing.T) {
			cache := remotePolarityFixture(t, backend, nil, errors.New("ordinary backend outage"))
			result, err := cache.LookupSimilarWithThreshold(context.Background(), "recipe::model", "Enable dark mode", .8)
			require.NoError(t, err)
			require.False(t, result.Found)
			require.Empty(t, result.ResponseBody)
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			_, err = cache.LookupSimilarWithThreshold(ctx, "recipe::model", "Enable dark mode", .8)
			require.ErrorIs(t, err, context.Canceled)
		})
	}
	redisCache := remotePolarityFixture(t, "redis", nil, nil).(*RedisCache)
	redisCache.searchFn = func(_ context.Context, _ string, _ string, options *redis.FTSearchOptions) (redis.FTSearchResult, error) {
		fields := make([]string, 0, len(options.Return))
		for _, field := range options.Return {
			fields = append(fields, field.FieldName)
		}
		require.Contains(t, fields, "query", "the persisted query must reach the candidate guard")
		return redis.FTSearchResult{}, nil
	}
	_, err := redisCache.LookupSimilarWithThreshold(context.Background(), "recipe::model", "Enable dark mode", .8)
	require.NoError(t, err)
	valkeyCache := remotePolarityFixture(t, "valkey", nil, nil).(*ValkeyCache)
	command := valkeyCache.buildKNNSearchCmd("recipe::model", nil)
	joined := strings.Join(command, " ")
	require.Contains(t, joined, "RETURN 5 vector_distance response_body query timestamp ttl_seconds DIALECT")
}

func TestRedisValkeySemanticPolarityMetadata(t *testing.T) {
	for _, backend := range []string{"redis", "valkey"} {
		for _, field := range []string{"query", "response_body", "expired", "not_finite"} {
			t.Run(backend+"/"+field, func(t *testing.T) {
				cache := remotePolarityFixture(t, backend, []remotePolarityDocument{{"How do I enable dark mode?", "UNKNOWN", "0.02"}}, nil)
				if c, ok := cache.(*RedisCache); ok {
					search := c.searchFn
					c.searchFn = func(ctx context.Context, index, query string, options *redis.FTSearchOptions) (redis.FTSearchResult, error) {
						result, err := search(ctx, index, query, options)
						fields := result.Docs[0].Fields
						switch field {
						case "expired":
							fields["timestamp"], fields["ttl_seconds"] = "1", "1"
						case "not_finite":
							fields["vector_distance"] = "NaN"
						default:
							delete(fields, field)
						}
						return result, err
					}
				} else {
					c := cache.(*ValkeyCache)
					search := c.searchFn
					c.searchFn = func(ctx context.Context, command []string) (any, error) {
						result, err := search(ctx, command)
						fields := result.([]interface{})[1].(map[string]interface{})["doc:0"].(map[string]interface{})
						switch field {
						case "expired":
							fields["timestamp"], fields["ttl_seconds"] = "1", "1"
						case "not_finite":
							fields["vector_distance"] = "NaN"
						default:
							fields[field] = nil
						}
						return result, err
					}
				}
				result, err := cache.LookupSimilarWithThreshold(context.Background(), "recipe::model", "How do I enable dark mode?", .8)
				require.NoError(t, err)
				require.False(t, result.Found)
				require.Empty(t, result.ResponseBody)
				require.Equal(t, int64(0), cache.GetStats().HitCount)
			})
		}
	}
}

func TestRedisValkeySemanticPolarityRetainsNegativeIPScore(t *testing.T) {
	for _, backend := range []string{"redis", "valkey"} {
		t.Run(backend, func(t *testing.T) {
			cache := remotePolarityFixture(t, backend, []remotePolarityDocument{
				{"How do I enable dark mode?", "INVALID", "not-a-distance"},
				{"How do I enable dark mode?", "BELOW_THRESHOLD", "1.25"},
			}, nil)
			if c, ok := cache.(*RedisCache); ok {
				c.config.Index.VectorField.MetricType = "IP"
			} else {
				cache.(*ValkeyCache).config.Index.VectorField.MetricType = "IP"
			}
			result, err := cache.LookupSimilarWithThreshold(context.Background(), "recipe::model", "How do I enable dark mode?", .8)
			require.NoError(t, err)
			require.False(t, result.Found)
			require.Empty(t, result.ResponseBody)
			require.Equal(t, float32(-.25), result.Similarity)
		})
	}
}

func TestRedisValkeySemanticPolarityCancellationAfterSearch(t *testing.T) {
	for _, backend := range []string{"redis", "valkey"} {
		t.Run(backend, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cache := remotePolarityFixture(t, backend, []remotePolarityDocument{{"How do I enable dark mode?", "ANSWER", "0.02"}}, nil)
			if c, ok := cache.(*RedisCache); ok {
				search := c.searchFn
				c.searchFn = func(ctx context.Context, index, query string, options *redis.FTSearchOptions) (redis.FTSearchResult, error) {
					result, err := search(ctx, index, query, options)
					cancel()
					return result, err
				}
			} else {
				c := cache.(*ValkeyCache)
				search := c.searchFn
				c.searchFn = func(ctx context.Context, command []string) (any, error) {
					result, err := search(ctx, command)
					cancel()
					return result, err
				}
			}
			result, err := cache.LookupSimilarWithThreshold(ctx, "recipe::model", "How do I enable dark mode?", .8)
			require.ErrorIs(t, err, context.Canceled)
			require.False(t, result.Found)
			require.Empty(t, result.ResponseBody)
			require.Equal(t, int64(0), cache.GetStats().HitCount)
		})
	}
}
