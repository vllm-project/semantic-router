//go:build !windows && cgo

package cache

import (
	"context"
	"errors"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// Coverage for #2473: the request context threads into the embedding work of a
// lookup. Each helper below documents the contract it pins.
var _ = Describe("Cache lookup cancellation and miss contract (#2473)", func() {
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

	specCancelledContextShortCircuits(newSeededBackend, threshold)
	specBelowThresholdMissReturnsZero(newSeededBackend, threshold)
	specCGOEmbedCancellation(newSeededBackend)
})

// specCancelledContextShortCircuits pins the best-effort half of the contract:
// an already-cancelled context short-circuits before the embed starts, and the
// lookup returns the context error instead of a possibly stale hit.
func specCancelledContextShortCircuits(newSeededBackend func() CacheBackend, threshold float32) {
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
}

// specBelowThresholdMissReturnsZero pins that a below-threshold lookup is a miss and
// does not publish the rejected candidate's similarity.
func specBelowThresholdMissReturnsZero(newSeededBackend func() CacheBackend, threshold float32) {
	Context("with a below-threshold query", func() {
		It("returns an empty result with zero similarity", func() {
			backend := newSeededBackend()
			defer func() { _ = backend.Close() }()

			res, err := backend.FindSimilarWithThreshold(
				context.Background(), "m",
				"totally unrelated question about database indexing",
				threshold,
			)

			Expect(err).NotTo(HaveOccurred())
			Expect(res.Found).To(BeFalse())
			Expect(res.Body).To(BeNil())
			Expect(res.Similarity).To(BeZero(),
				"a cache miss must not publish a candidate similarity")
		})
	})
}

// specCGOEmbedCancellation covers the other half: the embed cannot be
// interrupted mid-flight, so cancellation is re-checked after it returns and
// before the entry is published. cancelAfterEmbedCtx trips exactly on that
// second check.
//
// Asserting the bare context error, not errors.Is, is what pins the call site:
// a short-circuit inside the embed would surface it wrapped as "failed to
// generate embedding: ...", so a future ctxErr call added before the embed fails
// these specs instead of silently moving them to another branch.
func specCGOEmbedCancellation(newSeededBackend func() CacheBackend) {
	Context("with a context cancelled during the CGO embedding", func() {
		It("AddEntry returns the context error and publishes no entry", func() {
			backend := newSeededBackend()
			defer func() { _ = backend.Close() }()
			before := backend.GetStats().TotalEntries

			ctx := &cancelAfterEmbedCtx{Context: context.Background(), errAfter: errAfterPostEmbedGuard}
			err := backend.AddEntry(ctx, "orphan-1", "m", "a brand new distinct query",
				[]byte("req"), []byte("resp"), -1)

			Expect(err).To(Equal(context.Canceled),
				"expected the post-embed guard's bare context error, got %v", err)
			Expect(backend.GetStats().TotalEntries).To(Equal(before),
				"cancelled AddEntry must not publish an entry")
		})

		It("AddPendingRequest returns the context error and publishes no pending entry", func() {
			backend := newSeededBackend()
			defer func() { _ = backend.Close() }()
			before := backend.GetStats().TotalEntries

			ctx := &cancelAfterEmbedCtx{Context: context.Background(), errAfter: errAfterPostEmbedGuard}
			err := backend.AddPendingRequest(ctx, "orphan-2", "m", "another distinct query",
				[]byte("req"), -1)

			Expect(err).To(Equal(context.Canceled),
				"expected the post-embed guard's bare context error, got %v", err)
			Expect(backend.GetStats().TotalEntries).To(Equal(before),
				"cancelled AddPendingRequest must not publish a pending entry")
		})
	})
}

// errAfterPostEmbedGuard is the Err() call index of the guard under test: call
// #1 is generateEmbedding's pre-embed short-circuit, call #2 is the write
// method's post-embed re-check.
const errAfterPostEmbedGuard = 2

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
