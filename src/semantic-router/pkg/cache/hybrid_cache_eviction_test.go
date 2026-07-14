//go:build !windows && cgo

package cache

import "testing"

// oneHot returns a unit vector of dimension dim with 1.0 at position i%dim.
// Distinct positions give orthogonal vectors (dot product 0), so an exact-match
// query scores 1.0 against its own vector and 0.0 against every other.
func oneHot(i, dim int) []float32 {
	v := make([]float32, dim)
	v[i%dim] = 1.0
	return v
}

// addToMemoryIndexForTest reproduces the in-memory indexing sequence the public
// Add* methods run under h.mu (evict-if-full, append, index), so the eviction
// path can be exercised without a live Milvus backend.
func addToMemoryIndexForTest(h *HybridCache, milvusID string, embedding []float32) {
	if len(h.embeddings) >= h.maxMemoryEntries {
		h.evictOneUnsafe()
	}
	entryIndex := len(h.embeddings)
	h.embeddings = append(h.embeddings, embedding)
	h.idMap[entryIndex] = milvusID
	h.addNodeHybridOrRebuild(entryIndex, embedding)
}

func newTestHybridCache(maxEntries int) *HybridCache {
	return &HybridCache{
		enabled:          true,
		maxMemoryEntries: maxEntries,
		idMap:            make(map[int]string),
		hnswIndex:        newHNSWIndex(16, 200),
	}
}

// TestHybridCacheEvictionRebuildsHNSWIndex is a regression test for the bug where
// evictOneUnsafe wiped the HNSW graph but nothing rebuilt it, leaving only the
// last node and degrading the cache to ~100% misses at capacity. The restored
// invariant: after eviction the graph indexes every retained embedding and
// retained entries stay findable.
func TestHybridCacheEvictionRebuildsHNSWIndex(t *testing.T) {
	const (
		maxEntries = 5
		total      = 20
		dim        = 8
	)
	h := newTestHybridCache(maxEntries)

	for i := 0; i < total; i++ {
		addToMemoryIndexForTest(h, requestIDForTest(i), oneHot(i, dim))
	}

	// The in-memory graph must index every retained embedding. Before the fix
	// this was ~1 (only the last inserted node survived markStale).
	if got := len(h.hnswIndex.nodes); got != len(h.embeddings) {
		t.Fatalf("HNSW graph is out of sync with embeddings after eviction: "+
			"nodes=%d, embeddings=%d (want equal)", got, len(h.embeddings))
	}
	if len(h.embeddings) != maxEntries {
		t.Fatalf("expected %d retained embeddings, got %d", maxEntries, len(h.embeddings))
	}
	if h.hnswNeedsRebuild {
		t.Fatal("hnswNeedsRebuild should be cleared once the index has been rebuilt")
	}

	// Every retained entry must still be findable via the HNSW search path.
	// FIFO retains the last maxEntries insertions (indices total-maxEntries..total-1).
	for i := total - maxEntries; i < total; i++ {
		query := oneHot(i, dim)
		results := h.searchKNNHybridWithThreshold(query, 10, 200, 0.9)
		if !containsSimilarMatch(results, 0.99) {
			t.Fatalf("retained entry %d not found in HNSW search after eviction "+
				"(results=%+v); index likely holds only the last node", i, results)
		}
	}
}

// TestHybridCacheDeferredRebuildMechanics pins the flag handshake directly:
// eviction marks the graph stale and empties it, and the next add rebuilds it
// from all embeddings and clears the flag.
func TestHybridCacheDeferredRebuildMechanics(t *testing.T) {
	const dim = 8
	h := newTestHybridCache(3)

	// Fill to capacity without triggering eviction yet.
	for i := 0; i < 3; i++ {
		addToMemoryIndexForTest(h, requestIDForTest(i), oneHot(i, dim))
	}
	if h.hnswNeedsRebuild {
		t.Fatal("no eviction has happened yet; rebuild flag must be false")
	}
	if len(h.hnswIndex.nodes) != 3 {
		t.Fatalf("expected 3 nodes before eviction, got %d", len(h.hnswIndex.nodes))
	}

	// Evict directly and observe the deferred-rebuild handshake.
	h.evictOneUnsafe()
	if !h.hnswNeedsRebuild {
		t.Fatal("evictOneUnsafe must mark the HNSW graph for rebuild")
	}
	if len(h.hnswIndex.nodes) != 0 {
		t.Fatalf("evictOneUnsafe should have cleared the graph, got %d nodes", len(h.hnswIndex.nodes))
	}

	// The next add appends and must rebuild from all embeddings, not just add one.
	entryIndex := len(h.embeddings)
	h.embeddings = append(h.embeddings, oneHot(99, dim))
	h.idMap[entryIndex] = requestIDForTest(99)
	h.addNodeHybridOrRebuild(entryIndex, h.embeddings[entryIndex])

	if h.hnswNeedsRebuild {
		t.Fatal("rebuild flag must be cleared after addNodeHybridOrRebuild rebuilds")
	}
	if got := len(h.hnswIndex.nodes); got != len(h.embeddings) {
		t.Fatalf("graph out of sync after deferred rebuild: nodes=%d, embeddings=%d", got, len(h.embeddings))
	}
}

func requestIDForTest(i int) string {
	return "req-" + itoaForTest(i)
}

// itoaForTest is a tiny local int->string to avoid importing strconv just for tests.
func itoaForTest(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}

func containsSimilarMatch(results []searchResult, minSimilarity float32) bool {
	for _, r := range results {
		if r.similarity >= minSimilarity {
			return true
		}
	}
	return false
}
