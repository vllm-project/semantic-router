//go:build !windows && cgo

package cache

import (
	"context"
	"errors"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("Hybrid cache cross-model fallback", func() {
	const threshold = float32(0.8)
	const query = "what is semantic routing"

	newCache := func(queryResult client.ResultSet, queryErr error) *HybridCache {
		cfg := &config.MilvusConfig{}
		cfg.Collection.VectorField.Dimension = 384
		milvus := &MilvusCache{
			enabled:        true,
			config:         cfg,
			embeddingModel: "bert",
			queryByIDFn: func(context.Context, string, string) (client.ResultSet, error) {
				return queryResult, queryErr
			},
			searchFn: func(context.Context, string, []float32) ([]client.SearchResult, error) {
				return nil, nil
			},
		}
		hybrid := newTestHybridCache(1)
		hybrid.milvusCache = milvus

		embedding, err := milvus.getEmbedding(context.Background(), query)
		Expect(err).NotTo(HaveOccurred())
		addToMemoryIndexForTest(hybrid, "candidate-id", embedding)
		return hybrid
	}

	It("records an ordinary miss when the HNSW candidate belongs to another model", func() {
		cache := newCache(nil, nil)

		result, err := cache.LookupSimilarWithThreshold(context.Background(), "requested-model", query, threshold)

		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(LookupResult{}))
		Expect(cache.GetStats().MissCount).To(Equal(int64(1)))
	})

	It("returns zero plus an error when the candidate fetch actually fails", func() {
		backendErr := errors.New("milvus query failed")
		cache := newCache(nil, backendErr)

		result, err := cache.LookupSimilarWithThreshold(context.Background(), "requested-model", query, threshold)

		Expect(errors.Is(err, backendErr)).To(BeTrue())
		Expect(result).To(Equal(LookupResult{}))
		Expect(cache.GetStats().MissCount).To(Equal(int64(1)))
	})
})
