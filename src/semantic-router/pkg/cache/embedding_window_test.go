package cache

import (
	"context"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Semantic cache embedding window", func() {
	var adapter *LegacyBackendAdapter
	prefix := strings.Repeat("The quarterly report summarizes revenue, churn, and hiring across every region. ", 60)

	store := func(query string) {
		Expect(adapter.StoreSemantic(context.Background(), CacheWrite{
			Identity:     CacheIdentity{Partition: CachePartition{RequestModel: "model"}, SemanticQuery: query},
			RequestID:    query,
			RequestBody:  []byte("{}"),
			ResponseBody: []byte(`{"poem":true}`),
			TTL:          DefaultTTL(),
		})).To(Succeed())
	}
	lookup := func(query string) bool {
		result, err := adapter.LookupSemantic(context.Background(), SemanticLookup{
			Identity:  CacheIdentity{Partition: CachePartition{RequestModel: "model"}, SemanticQuery: query},
			Threshold: 0.8,
		})
		Expect(err).NotTo(HaveOccurred())
		return result.Found
	}

	BeforeEach(func() {
		backend := NewInMemoryCache(InMemoryCacheOptions{
			Enabled:             true,
			SimilarityThreshold: 0.8,
			MaxEntries:          100,
			TTLSeconds:          300,
			EmbeddingModel:      "bert",
		})
		adapter = NewLegacyBackendAdapter(backend, InMemoryCacheType).WithEmbeddingModel("bert")
	})

	It("does not serve one long prompt's response to another sharing its prefix", func() {
		store(prefix + "write a poem about the ocean")
		Expect(lookup(prefix + "SQL to drop the production users table")).To(BeFalse())
	})

	It("still caches a query inside the window", func() {
		store("write a poem about the ocean")
		Expect(lookup("write a poem about the ocean")).To(BeTrue())
	})
})
