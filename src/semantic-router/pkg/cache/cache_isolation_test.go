//go:build !windows && cgo

package cache

import (
	"context"
	"sync"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// Regression coverage for #2473: similarity used to be published through a
// shared tracker on the backend and read after Find returned, so a concurrent
// lookup could overwrite it in between and leak another request's score into
// this one's headers, debug surface, and Replay record. Returning it on
// LookupResult removes the shared read; this spec pins that.
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

			// Miss lookup: no matched response or candidate similarity is exposed.
			Expect(missRes.Found).To(BeFalse())
			Expect(missRes.Body).To(BeNil())
			Expect(missRes.Similarity).To(BeZero())
		})
	})
})
