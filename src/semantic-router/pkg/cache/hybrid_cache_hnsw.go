//go:build !windows && cgo

package cache

import "math/rand/v2"

// searchResult holds a candidate with its similarity score.
type searchResult struct {
	index      int
	similarity float32
}

// dotProduct calculates the dot product between two vectors.
// Uses SIMD instructions (AVX2/AVX-512) when available for performance.
// Falls back to scalar implementation on non-x86 platforms.
func dotProduct(a, b []float32) float32 {
	return dotProductSIMD(a, b)
}

// hybridHNSWAdapter adapts the HNSW index to work with [][]float32 instead of []CacheEntry.
type hybridHNSWAdapter struct {
	embeddings [][]float32
}

type hybridSimilarityFunc func(query, embedding []float32) float32

func (h *hybridHNSWAdapter) getEmbedding(idx int) []float32 {
	if idx < 0 || idx >= len(h.embeddings) {
		return nil
	}
	return h.embeddings[idx]
}

func (h *hybridHNSWAdapter) distance(idx1, idx2 int) float32 {
	emb1 := h.getEmbedding(idx1)
	emb2 := h.getEmbedding(idx2)
	if emb1 == nil || emb2 == nil {
		return 0
	}
	return dotProduct(emb1, emb2)
}

// addNodeHybrid adds a node to the HNSW index (hybrid version).
func (h *HybridCache) addNodeHybrid(entryIndex int, embedding []float32) {
	level := h.selectLevelHybrid()
	node := newHybridNode(entryIndex, level)
	h.registerHybridNode(entryIndex, node)

	if h.initializeHybridEntryPoint(entryIndex, level) {
		return
	}

	adapter := &hybridHNSWAdapter{embeddings: h.embeddings}
	currNearest := h.findNearestHybridEntry(entryIndex, level, adapter)
	h.connectHybridNodeLayers(node, entryIndex, embedding, level, currNearest)
	h.promoteHybridEntryPoint(entryIndex, level)
}

func newHybridNode(entryIndex, level int) *HNSWNode {
	node := &HNSWNode{
		entryIndex: entryIndex,
		neighbors:  make(map[int][]int),
		maxLayer:   level,
	}
	for i := 0; i <= level; i++ {
		node.neighbors[i] = make([]int, 0)
	}
	return node
}

func (h *HybridCache) registerHybridNode(entryIndex int, node *HNSWNode) {
	h.hnswIndex.nodes = append(h.hnswIndex.nodes, node)
	h.hnswIndex.nodeIndex[entryIndex] = node
}

func (h *HybridCache) initializeHybridEntryPoint(entryIndex, level int) bool {
	if h.hnswIndex.entryPoint != -1 {
		return false
	}
	h.hnswIndex.entryPoint = entryIndex
	h.hnswIndex.maxLayer = level
	return true
}

func (h *HybridCache) findNearestHybridEntry(entryIndex, level int, adapter *hybridHNSWAdapter) int {
	currNearest := h.hnswIndex.entryPoint
	for lc := h.hnswIndex.maxLayer; lc > level; lc-- {
		currNearest = h.bestHybridCandidate(entryIndex, currNearest, lc, adapter)
	}
	return currNearest
}

func (h *HybridCache) bestHybridCandidate(entryIndex, currNearest, layer int, adapter *hybridHNSWAdapter) int {
	bestDist := adapter.distance(entryIndex, currNearest)
	for _, candidate := range h.hybridLayerCandidates(currNearest, layer) {
		dist := adapter.distance(entryIndex, candidate)
		if dist > bestDist {
			bestDist = dist
			currNearest = candidate
		}
	}
	return currNearest
}

func (h *HybridCache) hybridLayerCandidates(entryPoint, layer int) []int {
	candidates := []int{entryPoint}
	node := h.hnswIndex.nodeIndex[entryPoint]
	if node == nil {
		return candidates
	}
	for _, neighbor := range node.neighbors[layer] {
		if h.isValidHybridEmbeddingIndex(neighbor) {
			candidates = append(candidates, neighbor)
		}
	}
	return candidates
}

func (h *HybridCache) connectHybridNodeLayers(node *HNSWNode, entryIndex int, embedding []float32, level, currNearest int) {
	for lc := level; lc >= 0; lc-- {
		neighbors := h.searchLayerHybrid(embedding, h.hnswIndex.efConstruction, lc, []int{currNearest})
		h.linkHybridNeighbors(node, entryIndex, lc, h.selectNeighborsHybrid(neighbors, h.hybridNeighborLimit(lc)))
	}
}

func (h *HybridCache) hybridNeighborLimit(layer int) int {
	if layer == 0 {
		return h.hnswIndex.Mmax0
	}
	return h.hnswIndex.M
}

func (h *HybridCache) linkHybridNeighbors(node *HNSWNode, entryIndex, layer int, neighbors []int) {
	for _, neighborID := range neighbors {
		node.neighbors[layer] = append(node.neighbors[layer], neighborID)
		neighborNode := h.hnswIndex.nodeIndex[neighborID]
		if neighborNode == nil {
			continue
		}
		if neighborNode.neighbors[layer] == nil {
			neighborNode.neighbors[layer] = make([]int, 0)
		}
		neighborNode.neighbors[layer] = append(neighborNode.neighbors[layer], entryIndex)
	}
}

func (h *HybridCache) promoteHybridEntryPoint(entryIndex, level int) {
	if level <= h.hnswIndex.maxLayer {
		return
	}
	h.hnswIndex.maxLayer = level
	h.hnswIndex.entryPoint = entryIndex
}

// selectLevelHybrid randomly selects a level for a new node.
func (h *HybridCache) selectLevelHybrid() int {
	level := 0
	for level < maxHNSWLayers {
		if randFloat() > h.hnswIndex.ml {
			break
		}
		level++
	}
	return level
}

// randFloat returns a random float between 0 and 1.
func randFloat() float64 {
	return rand.Float64()
}

// searchLayerHybrid searches for nearest neighbors at a specific layer.
func (h *HybridCache) searchLayerHybrid(query []float32, ef int, layer int, entryPoints []int) []int {
	return h.searchLayerHybridInternal(query, ef, layer, entryPoints, dotProduct, nil)
}

func (h *HybridCache) searchLayerHybridInternal(
	query []float32,
	ef int,
	layer int,
	entryPoints []int,
	similarityFn hybridSimilarityFunc,
	threshold *float32,
) []int {
	buf := getSearchBuffers()
	defer putSearchBuffers(buf)

	if match, found := h.seedHybridSearch(query, entryPoints, buf, similarityFn, threshold); found {
		return match
	}

	for buf.candidates.len() > 0 {
		currentIdx, currentDist := buf.candidates.pop()
		if shouldStopHybridSearch(currentDist, buf.results) {
			break
		}
		if match, found := h.expandHybridSearchNode(query, ef, layer, currentIdx, buf, similarityFn, threshold); found {
			return match
		}
	}

	// return the indexes of the results sorted by similarity (nearest first)
	return buf.results.sortedIndicesAsc()
}

// selectNeighborsHybrid selects the best neighbors from candidates (hybrid version).
func (h *HybridCache) selectNeighborsHybrid(candidates []int, m int) []int {
	if len(candidates) <= m {
		return candidates
	}
	return candidates[:m]
}

// searchKNNHybridWithThreshold searches for k nearest neighbors with early stopping.
func (h *HybridCache) searchKNNHybridWithThreshold(query []float32, k int, ef int, threshold float32) []searchResult {
	if h.hnswIndex.entryPoint == -1 || len(h.embeddings) == 0 {
		return nil
	}

	currNearest := []int{h.hnswIndex.entryPoint}
	for lc := h.hnswIndex.maxLayer; lc > 0; lc-- {
		currNearest = h.searchLayerHybrid(query, 1, lc, currNearest)
	}

	candidateIndices := h.searchLayerHybridWithEarlyStop(query, ef, 0, currNearest, threshold)
	results := make([]searchResult, 0, len(candidateIndices))
	for _, idx := range candidateIndices {
		if idx < 0 || idx >= len(h.embeddings) {
			continue
		}
		similarity := dotProductSIMD(query, h.embeddings[idx])
		results = append(results, searchResult{index: idx, similarity: similarity})
		if similarity >= threshold {
			return results
		}
	}

	if len(results) > k {
		return results[:k]
	}
	return results
}

// searchLayerHybridWithEarlyStop searches a layer and stops when finding a match above threshold.
func (h *HybridCache) searchLayerHybridWithEarlyStop(query []float32, ef int, layer int, entryPoints []int, threshold float32) []int {
	return h.searchLayerHybridInternal(query, ef, layer, entryPoints, dotProductSIMD, &threshold)
}

func (h *HybridCache) seedHybridSearch(
	query []float32,
	entryPoints []int,
	buf *searchBuffers,
	similarityFn hybridSimilarityFunc,
	threshold *float32,
) ([]int, bool) {
	for _, ep := range entryPoints {
		if !h.isValidHybridEmbeddingIndex(ep) {
			continue
		}
		similarity := similarityFn(query, h.embeddings[ep])
		if hitHybridThreshold(similarity, threshold) {
			return []int{ep}, true
		}
		dist := -similarity
		buf.candidates.push(ep, dist)
		buf.results.push(ep, dist)
		buf.visited[ep] = true
	}
	return nil, false
}

func (h *HybridCache) expandHybridSearchNode(
	query []float32,
	ef int,
	layer int,
	currentIdx int,
	buf *searchBuffers,
	similarityFn hybridSimilarityFunc,
	threshold *float32,
) ([]int, bool) {
	currentNode := h.hnswIndex.nodeIndex[currentIdx]
	if currentNode == nil || currentNode.neighbors[layer] == nil {
		return nil, false
	}
	for _, neighborID := range currentNode.neighbors[layer] {
		if buf.visited[neighborID] || !h.isValidHybridEmbeddingIndex(neighborID) {
			continue
		}
		buf.visited[neighborID] = true
		similarity := similarityFn(query, h.embeddings[neighborID])
		if hitHybridThreshold(similarity, threshold) {
			return []int{neighborID}, true
		}
		h.enqueueHybridSearchResult(neighborID, -similarity, ef, buf)
	}
	return nil, false
}

func (h *HybridCache) enqueueHybridSearchResult(index int, dist float32, ef int, buf *searchBuffers) {
	if buf.results.len() >= ef && dist >= buf.results.peekDist() {
		return
	}
	buf.candidates.push(index, dist)
	buf.results.push(index, dist)
	if buf.results.len() > ef {
		buf.results.pop()
	}
}

func (h *HybridCache) isValidHybridEmbeddingIndex(idx int) bool {
	return idx >= 0 && idx < len(h.embeddings)
}

func shouldStopHybridSearch(currentDist float32, results *maxHeap) bool {
	return results.len() > 0 && currentDist > results.peekDist()
}

func hitHybridThreshold(similarity float32, threshold *float32) bool {
	return threshold != nil && similarity >= *threshold
}
