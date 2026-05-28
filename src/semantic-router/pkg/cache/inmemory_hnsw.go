//go:build !windows && cgo

package cache

import (
	"math"
	"math/rand/v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// HNSWNode represents a node in the HNSW graph
type HNSWNode struct {
	entryIndex int           // Index into InMemoryCache.entries
	neighbors  map[int][]int // Layer -> neighbor indices
	maxLayer   int           // Highest layer this node appears in
}

// HNSWIndex implements Hierarchical Navigable Small World graph for fast ANN search
type HNSWIndex struct {
	nodes          []*HNSWNode
	nodeIndex      map[int]*HNSWNode // entryIndex → node for O(1) lookup (critical for performance!)
	entryPoint     int               // Index of the top-level entry point
	maxLayer       int               // Maximum layer in the graph
	efConstruction int               // Size of dynamic candidate list during construction
	M              int               // Number of bi-directional links per node
	Mmax           int               // Maximum number of connections per node (=M)
	Mmax0          int               // Maximum number of connections for layer 0 (=M*2)
	ml             float64           // Normalization factor for level assignment
}

// newHNSWIndex creates a new HNSW index
func newHNSWIndex(m, efConstruction int) *HNSWIndex {
	return &HNSWIndex{
		nodes:          []*HNSWNode{},
		nodeIndex:      make(map[int]*HNSWNode), // Initialize O(1) lookup map
		entryPoint:     -1,
		maxLayer:       -1,
		efConstruction: efConstruction,
		M:              m,
		Mmax:           m,
		Mmax0:          m * 2,
		ml:             1.0 / math.Log(float64(m)),
	}
}

// markStale marks the index as needing a rebuild
func (h *HNSWIndex) markStale() {
	// Simple approach: clear the index
	h.nodes = []*HNSWNode{}
	h.nodeIndex = make(map[int]*HNSWNode) // Clear O(1) lookup map
	h.entryPoint = -1
	h.maxLayer = -1
}

// selectLevel randomly selects a level for a new node using exponential decay.
func (h *HNSWIndex) selectLevel() int {
	r := rand.Float64()
	if r == 0.0 {
		r = 1e-9
	}
	return int(-math.Log(r) * h.ml)
}

func (h *HNSWIndex) addReverseNeighborLink(entryIndex int, neighborIdx int, lc int, maxNeighbors int, entries []CacheEntry) {
	n := h.nodeIndex[neighborIdx]
	if n == nil {
		return
	}
	if n.neighbors[lc] == nil {
		n.neighbors[lc] = []int{}
	}
	n.neighbors[lc] = append(n.neighbors[lc], entryIndex)
	if len(n.neighbors[lc]) <= maxNeighbors {
		return
	}
	n.neighbors[lc] = h.selectNeighbors(n.neighbors[lc], maxNeighbors, entries[neighborIdx].Embedding, entries)
}

// addNode adds a new node to the HNSW index.
//
// For InMemoryCache, it is called via addEntryToHNSWIndex to keep in sync with entries slice.
func (h *HNSWIndex) addNode(entryIndex int, embedding []float32, entries []CacheEntry) {
	level := h.selectLevel()

	node := &HNSWNode{
		entryIndex: entryIndex,
		neighbors:  make(map[int][]int),
		maxLayer:   level,
	}

	// If this is the first node, make it the entry point
	if h.entryPoint == -1 {
		h.entryPoint = entryIndex
		h.maxLayer = level
		h.nodes = append(h.nodes, node)
		h.nodeIndex[entryIndex] = node // Add to O(1) lookup map
		return
	}

	currentEntryPoint := h.entryPoint
	for lc := h.maxLayer; lc > level; lc-- {
		candidates := h.searchLayer(embedding, currentEntryPoint, 1, lc, entries)
		if len(candidates) > 0 {
			currentEntryPoint = candidates[0]
		}
	}

	// Find nearest neighbors and connect
	for lc := min(level, h.maxLayer); lc >= 0; lc-- {
		candidates := h.searchLayer(embedding, currentEntryPoint, h.efConstruction, lc, entries)

		M := h.Mmax
		if lc == 0 {
			M = h.Mmax0
		}
		neighbors := candidates
		if len(neighbors) > M {
			neighbors = neighbors[:M]
		}

		node.neighbors[lc] = neighbors
		for _, neighborIdx := range neighbors {
			h.addReverseNeighborLink(entryIndex, neighborIdx, lc, M, entries)
		}

		if len(candidates) > 0 {
			currentEntryPoint = candidates[0]
		}
	}

	// Update entry point if this node has a higher level
	if level > h.maxLayer {
		h.maxLayer = level
		h.entryPoint = entryIndex
	}

	h.nodes = append(h.nodes, node)
	h.nodeIndex[entryIndex] = node // Add to O(1) lookup map
}

// searchKNN performs k-nearest neighbor search
func (h *HNSWIndex) searchKNN(queryEmbedding []float32, k, ef int, entries []CacheEntry) []int {
	if h.entryPoint == -1 || len(h.nodes) == 0 {
		return []int{}
	}

	// Search from top layer to layer 1
	currentNearest := h.entryPoint
	for lc := h.maxLayer; lc > 0; lc-- {
		nearest := h.searchLayer(queryEmbedding, currentNearest, 1, lc, entries)
		if len(nearest) > 0 {
			currentNearest = nearest[0]
		}
	}

	// Search at layer 0 with ef
	return h.searchLayer(queryEmbedding, currentNearest, ef, 0, entries)
}

func (h *HNSWIndex) visitNeighborForSearch(
	queryEmbedding []float32,
	neighborIdx int,
	_ int,
	ef int,
	entries []CacheEntry,
	visited map[int]bool,
	candidates *minHeap,
	results *maxHeap,
) {
	if visited[neighborIdx] {
		return
	}
	visited[neighborIdx] = true
	if neighborIdx < 0 || neighborIdx >= len(entries) {
		return
	}
	dist := h.distance(queryEmbedding, entries[neighborIdx].Embedding)
	if results.len() < ef {
		candidates.push(neighborIdx, dist)
		results.push(neighborIdx, dist)
		return
	}
	if dist >= results.peekDist() {
		return
	}
	candidates.push(neighborIdx, dist)
	results.push(neighborIdx, dist)
	if results.len() > ef {
		results.pop()
	}
}

// searchLayer searches for nearest neighbors at a specific layer.
//
// It returns candidate entry indices ordered by ascending distance (nearest first, farthest last).
func (h *HNSWIndex) searchLayer(queryEmbedding []float32, entryPoint, ef, layer int, entries []CacheEntry) []int {
	visited := make(map[int]bool)
	candidates := newMinHeap() // set of candidates, explore closest candidate first
	results := newMaxHeap()    // dynamic list of found nearest neighbors, track current frontier, worst distance on top

	// Calculate distance to entry point
	if entryPoint >= 0 && entryPoint < len(entries) {
		dist := h.distance(queryEmbedding, entries[entryPoint].Embedding)
		candidates.push(entryPoint, dist)
		results.push(entryPoint, dist)
		visited[entryPoint] = true
	}

	for candidates.len() > 0 {
		currentIdx, currentDist := candidates.pop()

		if results.len() > 0 && currentDist > results.peekDist() {
			break
		}

		currentNode := h.nodeIndex[currentIdx]
		if currentNode == nil || currentNode.neighbors[layer] == nil {
			continue
		}

		for _, neighborIdx := range currentNode.neighbors[layer] {
			h.visitNeighborForSearch(queryEmbedding, neighborIdx, layer, ef, entries, visited, candidates, results)
		}
	}

	return results.sortedIndicesAsc()
}

// hnswNeighborDist pairs an entry index with distance-to-query for neighbor selection.
type hnswNeighborDist struct {
	idx  int
	dist float32
}

func (h *HNSWIndex) buildHNSWNeighborDists(candidates []int, queryEmb []float32, entries []CacheEntry) []hnswNeighborDist {
	neighbors := make([]hnswNeighborDist, len(candidates))
	for i, idx := range candidates {
		if idx < 0 || idx >= len(entries) {
			continue
		}
		emb := entries[idx].Embedding
		if len(emb) != len(queryEmb) {
			logging.Errorf(
				"selectNeighbors: dimension mismatch - query has %d dims, candidate %d has %d dims",
				len(queryEmb), idx, len(emb),
			)
			continue
		}
		neighbors[i] = hnswNeighborDist{idx: idx, dist: h.distance(queryEmb, emb)}
	}
	return neighbors
}

func partialSortHNSWNeighborDists(neighbors []hnswNeighborDist, m int) {
	// Selection sort for the m smallest distances (m is typically small, e.g. 16–32).
	for i := 0; i < m && i < len(neighbors); i++ {
		minIdx := i
		for j := i + 1; j < len(neighbors); j++ {
			if neighbors[j].dist < neighbors[minIdx].dist {
				minIdx = j
			}
		}
		if minIdx != i {
			neighbors[i], neighbors[minIdx] = neighbors[minIdx], neighbors[i]
		}
	}
}

func takeHNSWNeighborIndices(neighbors []hnswNeighborDist, m int) []int {
	result := make([]int, m)
	for i := 0; i < m; i++ {
		result[i] = neighbors[i].idx
	}
	return result
}

// selectNeighbors selects the best neighbors by sorting by distance
// This is CRITICAL for HNSW graph quality - must select NEAREST neighbors, not arbitrary ones!
func (h *HNSWIndex) selectNeighbors(candidates []int, m int, queryEmb []float32, entries []CacheEntry) []int {
	// Validate queryEmb: must not be nil or empty to ensure correct distance calculations
	if len(queryEmb) == 0 {
		logging.Errorf("selectNeighbors: queryEmb is empty - cannot compute distances")
		return []int{}
	}

	if len(candidates) <= m {
		return candidates
	}

	neighbors := h.buildHNSWNeighborDists(candidates, queryEmb, entries)
	partialSortHNSWNeighborDists(neighbors, m)
	return takeHNSWNeighborIndices(neighbors, m)
}

// distance calculates cosine similarity (as dot product since embeddings are normalized)
func (h *HNSWIndex) distance(a, b []float32) float32 {
	// We use negative dot product so that larger similarity = smaller distance
	// Use SIMD-optimized dot product (AVX2/AVX512)
	dotProduct := dotProductSIMD(a, b)
	return -dotProduct // Negate so higher similarity = lower distance
}
