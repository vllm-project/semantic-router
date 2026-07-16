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

	// Write-path cancellation (#2473): embedding is a synchronous CGO call that
	// cannot be interrupted mid-flight, so cancellation is re-checked AFTER the
	// embed completes and BEFORE the entry is published. A request cancelled in
	// that window must return the context error and leave no orphaned state.
	//
	// cancelAfterEmbedCtx trips exactly on the post-embed guard: generateEmbedding
	// calls ctxErr once (sees nil and proceeds), then the write method's guard
	// calls ctxErr again and observes cancellation — simulating a context that
	// was cancelled while the CGO embed was running.
	Context("with a context cancelled during the CGO embedding", func() {
		It("AddEntry returns the context error and publishes no entry", func() {
			backend := newSeededBackend()
			defer func() { _ = backend.Close() }()
			before := backend.GetStats().TotalEntries

			ctx := &cancelAfterEmbedCtx{Context: context.Background(), errAfter: 2}
			err := backend.AddEntry(ctx, "orphan-1", "m", "a brand new distinct query",
				[]byte("req"), []byte("resp"), -1)

			Expect(errors.Is(err, context.Canceled)).To(BeTrue(),
				"expected context.Canceled, got %v", err)
			Expect(backend.GetStats().TotalEntries).To(Equal(before),
				"cancelled AddEntry must not publish an entry")
		})

		It("AddPendingRequest returns the context error and publishes no pending entry", func() {
			backend := newSeededBackend()
			defer func() { _ = backend.Close() }()
			before := backend.GetStats().TotalEntries

			ctx := &cancelAfterEmbedCtx{Context: context.Background(), errAfter: 2}
			err := backend.AddPendingRequest(ctx, "orphan-2", "m", "another distinct query",
				[]byte("req"), -1)

			Expect(errors.Is(err, context.Canceled)).To(BeTrue(),
				"expected context.Canceled, got %v", err)
			Expect(backend.GetStats().TotalEntries).To(Equal(before),
				"cancelled AddPendingRequest must not publish a pending entry")
		})
	})
})

// cancelAfterEmbedCtx reports no error until the errAfter-th Err() call, then
// context.Canceled. It lets a test deterministically trip the post-embedding
// cancellation guard (Err() call #2) without racing the CGO embed.
type cancelAfterEmbedCtx struct {
	context.Context
	errAfter int
	calls    int
}

func (c *cancelAfterEmbedCtx) Err() error {
	c.calls++
	if c.calls >= c.errAfter {
		return context.Canceled
	}
	return nil
}
