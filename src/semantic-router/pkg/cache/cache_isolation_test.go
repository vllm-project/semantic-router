//go:build !windows && cgo

package cache

import (
	"context"
	"sync"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// Regression coverage for #2473: two concurrent lookups against the same cache
// backend must each receive their OWN similarity score. Before #2473 the score
// was published through a shared SimilarityTracker on the backend and read
// after the Find returned, so a concurrent lookup could overwrite the score
// between one caller's Find and its LastSimilarity() read — leaking another
// request's similarity into the reader's response header, debug surface, and
// Replay record.
//
// The fix moves similarity into the return value of Find/FindSimilarWithThreshold
// (a LookupResult), removing the shared-state read entirely. This spec pins
// the new contract.
var _ = Describe("Cache lookup isolation (regression #2473)", func() {
	Context("concurrent lookups on the same in-memory backend", func() {
		It("returns per-request similarity via LookupResult with no cross-request leak", func() {
			const threshold = float32(0.75)
			backend, err := NewCacheBackend(CacheConfig{
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: threshold,
				MaxEntries:          16,
				EmbeddingModel:      "bert",
			})
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = backend.Close() }()

			hitQuery := "what is the capital of france"
			missQuery := "totally unrelated question about database indexing"

			// ttlSeconds=-1 means "use cache default TTL"; ttlSeconds=0 would
			// mark the entry as uncacheable and drop it silently.
			Expect(backend.AddEntry(context.Background(),
				"seed-1", "m", hitQuery,
				[]byte("req"), []byte("cached"), -1,
			)).To(Succeed())

			barrier := make(chan struct{})
			var wg sync.WaitGroup
			var hitRes, missRes LookupResult
			var hitErr, missErr error

			wg.Add(2)
			go func() {
				defer wg.Done()
				<-barrier
				hitRes, hitErr = backend.FindSimilarWithThreshold(
					context.Background(), "m", hitQuery, threshold,
				)
			}()
			go func() {
				defer wg.Done()
				<-barrier
				missRes, missErr = backend.FindSimilarWithThreshold(
					context.Background(), "m", missQuery, threshold,
				)
			}()
			close(barrier)
			wg.Wait()

			Expect(hitErr).NotTo(HaveOccurred())
			Expect(missErr).NotTo(HaveOccurred())

			// Hit lookup: matched entry, similarity is this request's own score.
			Expect(hitRes.Found).To(BeTrue(),
				"hit lookup expected Found=true (similarity=%.4f)", hitRes.Similarity)
			Expect(hitRes.Similarity).To(BeNumerically(">=", threshold),
				"hit lookup similarity below threshold — cross-request leak from miss?")

			// Miss lookup: no match. Similarity must be this request's own best
			// score (well below threshold), never the concurrent hit's score.
			Expect(missRes.Found).To(BeFalse(),
				"miss lookup expected Found=false (similarity=%.4f)", missRes.Similarity)
			Expect(missRes.Similarity).To(BeNumerically("<", threshold),
				"miss lookup similarity >= threshold — cross-request leak from hit")
		})
	})
})
