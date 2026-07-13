//go:build !windows && cgo

package cache

import (
	"context"
	"errors"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// Coverage for #2473: the request context now threads into the embedding work
// of a lookup. Embedding is a synchronous CGO call that cannot be interrupted
// mid-flight, so the contract is best-effort: an already-cancelled context
// short-circuits BEFORE the expensive embed starts, and the lookup returns the
// context error instead of a (potentially stale) hit.
//
// The second spec pins the per-request "best-observed score" semantics of a
// miss: a below-threshold lookup reports its OWN best similarity (> 0, below
// threshold), never zero and never another request's leaked score.
var _ = Describe("Cache lookup cancellation and per-request miss score (#2473)", func() {
	const threshold = float32(0.75)

	newSeededBackend := func() CacheBackend {
		backend, err := NewCacheBackend(CacheConfig{
			BackendType:         InMemoryCacheType,
			Enabled:             true,
			SimilarityThreshold: threshold,
			MaxEntries:          16,
			EmbeddingModel:      "bert",
		})
		Expect(err).NotTo(HaveOccurred())

		// ttlSeconds=-1 means "use cache default TTL"; ttlSeconds=0 would mark
		// the entry as uncacheable and drop it silently.
		Expect(backend.AddEntry(context.Background(),
			"seed-1", "m", "what is the capital of france",
			[]byte("req"), []byte("cached"), -1,
		)).To(Succeed())
		return backend
	}

	Context("with an already-cancelled context", func() {
		It("short-circuits before embedding and returns context.Canceled, not a hit", func() {
			backend := newSeededBackend()
			defer func() { _ = backend.Close() }()

			ctx, cancel := context.WithCancel(context.Background())
			cancel() // cancel before the lookup starts

			res, err := backend.FindSimilarWithThreshold(
				ctx, "m", "what is the capital of france", threshold,
			)

			Expect(err).To(HaveOccurred())
			Expect(errors.Is(err, context.Canceled)).To(BeTrue(),
				"expected context.Canceled, got %v", err)
			// A cancelled lookup must never surface a cached hit.
			Expect(res.Found).To(BeFalse())
			Expect(res.Body).To(BeNil())
		})
	})

	Context("with a below-threshold query", func() {
		It("reports the request's own best-observed similarity (0 < sim < threshold)", func() {
			backend := newSeededBackend()
			defer func() { _ = backend.Close() }()

			res, err := backend.FindSimilarWithThreshold(
				context.Background(), "m",
				"totally unrelated question about database indexing",
				threshold,
			)

			Expect(err).NotTo(HaveOccurred())
			Expect(res.Found).To(BeFalse())
			// Best-observed score is this request's own, not zero (a candidate
			// was scored) and not leaked from another lookup (below threshold).
			Expect(res.Similarity).To(BeNumerically(">", float32(0)),
				"miss should carry the best-observed score, not zero")
			Expect(res.Similarity).To(BeNumerically("<", threshold))
		})
	})
})
