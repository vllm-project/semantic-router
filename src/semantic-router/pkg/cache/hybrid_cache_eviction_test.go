//go:build !windows

package cache

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// oneHot returns a unit vector of dimension dim with 1.0 at position i%dim.
// Distinct positions give orthogonal vectors (dot product 0), so an exact-match
// query scores 1.0 against its own vector and 0.0 against every other.
func oneHot(i, dim int) []float32 {
	v := make([]float32, dim)
	v[i%dim] = 1.0
	return v
}

// addToMemoryIndexForTest runs the in-memory indexing step the public Add*
// methods run under h.mu, so the eviction path can be exercised without a live
// Milvus backend.
func addToMemoryIndexForTest(h *HybridCache, milvusID string, embedding []float32) {
	h.addToMemoryIndexLocked(milvusID, embedding)
}

func newTestHybridCache(maxEntries int) *HybridCache {
	return &HybridCache{
		enabled:          true,
		maxMemoryEntries: maxEntries,
		idMap:            make(map[int]string),
		hnswIndex:        newHNSWIndex(16, 200),
	}
}

// TestHybridCacheEvictionKeepsRetainedEntriesSearchable is a regression test for
// the bug where eviction wiped the HNSW graph and nothing rebuilt it, leaving
// only the last node and degrading the cache to ~100% misses at capacity. The
// invariant: after eviction the graph indexes every retained embedding, and
// retained entries stay findable.
func TestHybridCacheEvictionKeepsRetainedEntriesSearchable(t *testing.T) {
	const (
		maxEntries = 5
		total      = 20
		dim        = 8
	)
	h := newTestHybridCache(maxEntries)

	for i := 0; i < total; i++ {
		addToMemoryIndexForTest(h, requestIDForTest(i), oneHot(i, dim))
	}

	if len(h.embeddings) != maxEntries {
		t.Fatalf("expected %d retained embeddings, got %d", maxEntries, len(h.embeddings))
	}
	assertHybridGraphConsistent(t, h)

	// FIFO retains the last maxEntries insertions.
	for i := total - maxEntries; i < total; i++ {
		results := h.searchKNNHybridWithThreshold(oneHot(i, dim), 10, 200, 0.9)
		if !containsSimilarMatch(results, 0.99) {
			t.Fatalf("retained entry %d not found in HNSW search after eviction (results=%+v)", i, results)
		}
	}
	for i := 0; i < total-maxEntries; i++ {
		if _, ok := hybridIndexOfIDForTest(h, requestIDForTest(i)); ok {
			t.Fatalf("evicted entry %d is still in the index", i)
		}
	}
}

// TestHybridCacheWriteAtCapacityKeepsOtherNodes checks that a write to a full
// index evicts the oldest entry and leaves every other entry's graph node in
// place. Rebuilding the whole graph on each such write held h.mu for seconds at
// a few thousand entries, blocking every cache lookup meanwhile.
func TestHybridCacheWriteAtCapacityKeepsOtherNodes(t *testing.T) {
	for _, capacity := range []int{1, 2, 64} {
		h := newTestHybridCache(capacity)
		r := rand.New(rand.NewPCG(7, 7))
		next := 0
		add := func() {
			addToMemoryIndexForTest(h, requestIDForTest(next), randomUnitVectorForTest(r, 16))
			next++
		}
		for next < capacity {
			add()
		}

		for write := 0; write < 3*capacity; write++ {
			oldest := requestIDForTest(next - capacity)
			nodes := make(map[string]*HNSWNode, capacity)
			for entryIndex, id := range h.idMap {
				nodes[id] = h.hnswIndex.nodeIndex[entryIndex]
			}

			add()

			if _, ok := hybridIndexOfIDForTest(h, oldest); ok {
				t.Fatalf("capacity %d: the oldest entry %s survived a write at capacity", capacity, oldest)
			}
			if len(h.embeddings) != capacity {
				t.Fatalf("capacity %d: %d entries after a write at capacity", capacity, len(h.embeddings))
			}
			for id, node := range nodes {
				if id == oldest {
					continue
				}
				entryIndex, ok := hybridIndexOfIDForTest(h, id)
				if !ok {
					t.Fatalf("capacity %d: entry %s was evicted, want only %s evicted", capacity, id, oldest)
				}
				if h.hnswIndex.nodeIndex[entryIndex] != node {
					t.Fatalf("capacity %d: entry %s got a new graph node on an unrelated write; "+
						"a write at capacity must not rebuild the graph", capacity, id)
				}
			}
			assertHybridGraphConsistent(t, h)
		}
		if got, want := h.evictCount, int64(3*capacity); got != want {
			t.Fatalf("capacity %d: evictCount=%d, want %d", capacity, got, want)
		}
	}
}

// TestHybridCacheChurnKeepsGraphQuality replaces the whole index several times
// over, then compares it with a graph freshly built from the retained entries.
// Eviction must keep the graph about as dense, and every retained entry must
// still be found with the production search settings. Without relinking the
// evicted nodes' neighbors, the graph keeps about half the links of a fresh
// one, and at 10,000 entries lookups missed several percent of them.
func TestHybridCacheChurnKeepsGraphQuality(t *testing.T) {
	const (
		capacity = 500
		dim      = 32
	)
	const total = 6 * capacity
	r := rand.New(rand.NewPCG(42, 42))
	h := newTestHybridCache(capacity)
	for i := 0; i < total; i++ {
		addToMemoryIndexForTest(h, requestIDForTest(i), randomUnitVectorForTest(r, dim))
	}
	assertHybridGraphConsistent(t, h)

	fresh := newTestHybridCache(capacity)
	for i := total - capacity; i < total; i++ {
		entryIndex, ok := hybridIndexOfIDForTest(h, requestIDForTest(i))
		if !ok {
			t.Fatalf("entry %d was evicted, but it is among the last %d written", i, capacity)
		}
		addToMemoryIndexForTest(fresh, requestIDForTest(i), h.embeddings[entryIndex])
	}
	churned, built := meanLayer0DegreeForTest(h), meanLayer0DegreeForTest(fresh)
	if churned < 0.8*built {
		t.Fatalf("mean layer-0 degree after churn is %.1f, want at least 80%% of a fresh build's %.1f", churned, built)
	}

	missed := 0
	for _, embedding := range h.embeddings {
		if !containsSimilarMatch(h.searchKNNHybridWithThreshold(embedding, 1, 20, 0.9999), 0.9999) {
			missed++
		}
	}
	// HNSW levels are random, so allow a little slack.
	if missed > capacity/100 {
		t.Fatalf("%d of %d retained entries not found after churn", missed, capacity)
	}
}

// assertHybridGraphConsistent checks that every entry has exactly one graph
// node, links are symmetric and point only at live nodes, and the entry point
// sits on the highest layer.
func assertHybridGraphConsistent(t *testing.T, h *HybridCache) {
	t.Helper()
	idx := h.hnswIndex
	n := len(h.embeddings)
	if len(idx.nodes) != n || len(idx.nodeIndex) != n || len(h.idMap) != n {
		t.Fatalf("graph out of sync: embeddings=%d nodes=%d nodeIndex=%d idMap=%d",
			n, len(idx.nodes), len(idx.nodeIndex), len(h.idMap))
	}
	topLayer := -1
	for entryIndex := 0; entryIndex < n; entryIndex++ {
		node := idx.nodeIndex[entryIndex]
		if node == nil || node.entryIndex != entryIndex {
			t.Fatalf("entry %d has no graph node", entryIndex)
		}
		topLayer = max(topLayer, node.maxLayer)
		for layer, neighbors := range node.neighbors {
			for _, neighborID := range neighbors {
				neighbor := idx.nodeIndex[neighborID]
				if neighborID == entryIndex || neighbor == nil {
					t.Fatalf("entry %d links to %d on layer %d, which is itself or not in the graph",
						entryIndex, neighborID, layer)
				}
				if !slices.Contains(neighbor.neighbors[layer], entryIndex) {
					t.Fatalf("entry %d links to %d on layer %d without a link back", entryIndex, neighborID, layer)
				}
			}
		}
	}
	if n == 0 {
		return
	}
	if ep := idx.nodeIndex[idx.entryPoint]; ep == nil || ep.maxLayer != topLayer || idx.maxLayer != topLayer {
		t.Fatalf("entry point %d is not on the top layer %d (index maxLayer=%d)", idx.entryPoint, topLayer, idx.maxLayer)
	}
}

func meanLayer0DegreeForTest(h *HybridCache) float64 {
	links := 0
	for _, node := range h.hnswIndex.nodeIndex {
		links += len(node.neighbors[0])
	}
	return float64(links) / float64(len(h.hnswIndex.nodeIndex))
}

func hybridIndexOfIDForTest(h *HybridCache, milvusID string) (int, bool) {
	for entryIndex, id := range h.idMap {
		if id == milvusID {
			return entryIndex, true
		}
	}
	return 0, false
}

func randomUnitVectorForTest(r *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	var norm float64
	for i := range v {
		x := r.NormFloat64()
		v[i] = float32(x)
		norm += x * x
	}
	for i := range v {
		v[i] /= float32(math.Sqrt(norm))
	}
	return v
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
