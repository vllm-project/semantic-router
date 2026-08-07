//go:build !windows && cgo

package cache

import (
	"context"
	"errors"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/qdrant/go-client/qdrant"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = DescribeTable("backend search cancellation wiring",
	func(findSimilar func(context.Context) (LookupResult, error)) {
		ctx := &cancelAfterEmbedCtx{Context: context.Background(), errAfter: 2}

		result, err := findSimilar(ctx)

		Expect(errors.Is(err, context.Canceled)).To(BeTrue(),
			"backend search failure must surface the request cancellation, got %v", err)
		Expect(result).To(Equal(LookupResult{}))

		result, err = findSimilar(context.Background())
		Expect(err).NotTo(HaveOccurred(), "ordinary backend errors must remain fail-open misses")
		Expect(result).To(Equal(LookupResult{}))
	},
	Entry("Milvus", func(ctx context.Context) (LookupResult, error) {
		cfg := &config.MilvusConfig{}
		cfg.Collection.VectorField.Dimension = 384
		cache := &MilvusCache{
			enabled:        true,
			config:         cfg,
			embeddingModel: "bert",
			searchFn: func(context.Context, string, []float32) ([]client.SearchResult, error) {
				return nil, errors.New("milvus unavailable")
			},
		}
		return cache.FindSimilarWithThreshold(ctx, "model", "query", 0.8)
	}),
	Entry("Qdrant", func(ctx context.Context) (LookupResult, error) {
		cache := &QdrantCache{
			enabled:        true,
			cfg:            &config.QdrantConfig{},
			embeddingModel: "bert",
			searchFn: func(context.Context, *qdrant.QueryPoints) ([]*qdrant.ScoredPoint, error) {
				return nil, errors.New("qdrant unavailable")
			},
		}
		return cache.FindSimilarWithThreshold(ctx, "model", "query", 0.8)
	}),
	Entry("Redis", func(ctx context.Context) (LookupResult, error) {
		cfg := &config.RedisConfig{}
		cfg.Index.VectorField.Name = "embedding"
		cfg.Search.TopK = 1
		cache := &RedisCache{
			enabled:        true,
			config:         cfg,
			embeddingModel: "bert",
			searchFn: func(context.Context, string, string, *redis.FTSearchOptions) (redis.FTSearchResult, error) {
				return redis.FTSearchResult{}, errors.New("redis unavailable")
			},
		}
		return cache.FindSimilarWithThreshold(ctx, "model", "query", 0.8)
	}),
	Entry("Valkey", func(ctx context.Context) (LookupResult, error) {
		cfg := &config.ValkeyConfig{}
		cfg.Index.VectorField.Name = "embedding"
		cfg.Search.TopK = 1
		cache := &ValkeyCache{
			enabled:        true,
			config:         cfg,
			embeddingModel: "bert",
			searchFn: func(context.Context, []string) (any, error) {
				return nil, errors.New("valkey unavailable")
			},
		}
		return cache.FindSimilarWithThreshold(ctx, "model", "query", 0.8)
	}),
)
