//go:build !windows && cgo

package cache

import (
	"context"
	"errors"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// Coverage for #2473: the request context now threads into the embedding work
// of a lookup. The specs are registered by the helpers below; see each helper's
// doc comment for the exact contract it pins.
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

// specCancelledContextShortCircuits pins that embedding is a synchronous CGO
// call that cannot be interrupted mid-flight, so the contract is best-effort: an
// already-cancelled context short-circuits BEFORE the expensive embed starts,
// and the lookup returns the context error instead of a (potentially stale) hit.
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

// specCGOEmbedCancellation covers write-path cancellation (#2473): embedding is
// a synchronous CGO call that cannot be interrupted mid-flight, so cancellation
// is re-checked AFTER the embed completes and BEFORE the entry is published. A
// request cancelled in that window must return the context error and leave no
// orphaned state.
//
// cancelAfterEmbedCtx trips on the post-embed guard: generateEmbedding calls
// ctxErr once (sees nil and proceeds), then the write method's guard calls
// ctxErr again and observes cancellation — simulating a context cancelled while
// the CGO embed was running.
//
// The specs assert the *bare* context error rather than errors.Is, which is what
// pins the call site: an embedding-path short-circuit would surface it wrapped
// as "failed to generate embedding: context canceled". So if a future change
// adds another ctxErr call before the embed — making the fake trip at the wrong
// site — these specs fail instead of silently covering a different branch.
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
