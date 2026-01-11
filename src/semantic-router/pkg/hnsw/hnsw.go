//go:build !windows && cgo

// Package hnsw provides a Hierarchical Navigable Small World (HNSW) graph implementation
// for fast approximate nearest neighbor search on high-dimensional vectors.
//
// HNSW enables O(log n) similarity search compared to O(n) brute-force search,
// making it ideal for large-scale embedding similarity matching.
//
// Key features:
//   - O(log n) search complexity
//   - Configurable accuracy vs speed tradeoff via M and ef parameters
//   - SIMD-optimized distance calculations (AVX2/AVX-512)
//   - Thread-safe for concurrent reads
package hnsw

import (
	"math"
	"sync"
	"time"
)

// Node represents a node in the HNSW graph
type Node struct {
	ID        int           // Unique identifier for this node
	Embedding []float32     // The embedding vector for this node
	neighbors map[int][]int // Layer -> neighbor IDs
	maxLayer  int           // Highest layer this node appears in
}

// Index implements Hierarchical Navigable Small World graph for fast ANN search
type Index struct {
	mu             sync.RWMutex
	nodes          map[int]*Node // ID -> Node
	entryPoint     int           // ID of the top-level entry point
	maxLayer       int           // Maximum layer in the graph
	efConstruction int           // Size of dynamic candidate list during construction
	efSearch       int           // Size of dynamic candidate list during search
	M              int           // Number of bi-directional links per node
	Mmax           int           // Maximum number of connections per node (=M)
	Mmax0          int           // Maximum number of connections for layer 0 (=M*2)
	ml             float64       // Normalization factor for level assignment
	nodeCount      int           // Total number of nodes
}

// Config contains HNSW index configuration parameters
type Config struct {
	// M is the number of bi-directional links per node (default: 16)
	// Higher values improve recall but increase memory usage
	M int

	// EfConstruction is the size of dynamic candidate list during construction (default: 200)
	// Higher values improve index quality but increase build time
	EfConstruction int

	// EfSearch is the size of dynamic candidate list during search (default: 50)
	// Higher values improve search accuracy but increase latency
	EfSearch int
}

// DefaultConfig returns a Config with sensible default values
func DefaultConfig() Config {
	return Config{
		M:              16,
		EfConstruction: 200,
		EfSearch:       50,
	}
}

// SearchResult represents a single search result
type SearchResult struct {
	ID         int     // Node ID
	Similarity float32 // Cosine similarity score (higher is better)
}

// NewIndex creates a new HNSW index with the given configuration
func NewIndex(cfg Config) *Index {
	// Apply defaults for zero values
	if cfg.M <= 0 {
		cfg.M = 16
	}
	if cfg.EfConstruction <= 0 {
		cfg.EfConstruction = 200
	}
	if cfg.EfSearch <= 0 {
		cfg.EfSearch = 50
	}

	return &Index{
		nodes:          make(map[int]*Node),
		entryPoint:     -1,
		maxLayer:       -1,
		efConstruction: cfg.EfConstruction,
		efSearch:       cfg.EfSearch,
		M:              cfg.M,
		Mmax:           cfg.M,
		Mmax0:          cfg.M * 2,
		ml:             1.0 / math.Log(float64(cfg.M)),
	}
}

// Add inserts a new node with the given ID and embedding into the index
func (h *Index) Add(id int, embedding []float32) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.addNode(id, embedding)
}

// AddBatch inserts multiple nodes into the index
func (h *Index) AddBatch(embeddings map[int][]float32) {
	h.mu.Lock()
	defer h.mu.Unlock()

	for id, embedding := range embeddings {
		h.addNode(id, embedding)
	}
}

// Search finds the k most similar nodes to the query embedding
// Returns results sorted by similarity (highest first)
func (h *Index) Search(queryEmbedding []float32, k int) []SearchResult {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.entryPoint == -1 || len(h.nodes) == 0 {
		return []SearchResult{}
	}

	// Search from top layer to layer 1
	currentNearest := h.entryPoint
	for lc := h.maxLayer; lc > 0; lc-- {
		nearest := h.searchLayer(queryEmbedding, currentNearest, 1, lc)
		if len(nearest) > 0 {
			currentNearest = nearest[0]
		}
	}

	// Search at layer 0 with efSearch
	ef := h.efSearch
	if ef < k {
		ef = k
	}
	candidateIDs := h.searchLayer(queryEmbedding, currentNearest, ef, 0)

	// Convert to SearchResults with similarities
	results := make([]SearchResult, 0, len(candidateIDs))
	for _, id := range candidateIDs {
		if node, ok := h.nodes[id]; ok {
			similarity := dotProductSIMD(queryEmbedding, node.Embedding)
			results = append(results, SearchResult{
				ID:         id,
				Similarity: similarity,
			})
		}
	}

	// Sort by similarity (descending) and return top k
	sortBySimDesc(results)
	if len(results) > k {
		results = results[:k]
	}

	return results
}

// SearchAll performs a brute-force search against all nodes
// Useful for small indices or when exact results are needed
func (h *Index) SearchAll(queryEmbedding []float32, k int) []SearchResult {
	h.mu.RLock()
	defer h.mu.RUnlock()

	results := make([]SearchResult, 0, len(h.nodes))
	for id, node := range h.nodes {
		similarity := dotProductSIMD(queryEmbedding, node.Embedding)
		results = append(results, SearchResult{
			ID:         id,
			Similarity: similarity,
		})
	}

	sortBySimDesc(results)
	if len(results) > k {
		results = results[:k]
	}

	return results
}

// Size returns the number of nodes in the index
func (h *Index) Size() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.nodes)
}

// GetEmbedding returns the embedding for a given node ID
func (h *Index) GetEmbedding(id int) ([]float32, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if node, ok := h.nodes[id]; ok {
		return node.Embedding, true
	}
	return nil, false
}

// Clear removes all nodes from the index
func (h *Index) Clear() {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.nodes = make(map[int]*Node)
	h.entryPoint = -1
	h.maxLayer = -1
	h.nodeCount = 0
}

// ===== Internal Implementation =====

// selectLevel randomly selects a level for a new node
func (h *Index) selectLevel() int {
	// Use exponential decay probability
	r := -math.Log(math.Max(1e-9, 1.0-float64(time.Now().UnixNano()%1000000)/1000000.0))
	return int(r * h.ml)
}

// addNode adds a new node to the HNSW index (caller must hold write lock)
func (h *Index) addNode(id int, embedding []float32) {
	level := h.selectLevel()

	node := &Node{
		ID:        id,
		Embedding: embedding,
		neighbors: make(map[int][]int),
		maxLayer:  level,
	}

	// If this is the first node, make it the entry point
	if h.entryPoint == -1 {
		h.entryPoint = id
		h.maxLayer = level
		h.nodes[id] = node
		h.nodeCount++
		return
	}

	// Find nearest neighbors and connect
	for lc := min(level, h.maxLayer); lc >= 0; lc-- {
		candidates := h.searchLayer(embedding, h.entryPoint, h.efConstruction, lc)

		// Select M nearest neighbors
		M := h.Mmax
		if lc == 0 {
			M = h.Mmax0
		}
		neighbors := h.selectNeighbors(candidates, M, embedding)

		// Add bidirectional links
		node.neighbors[lc] = neighbors
		for _, neighborID := range neighbors {
			if n, ok := h.nodes[neighborID]; ok {
				if n.neighbors[lc] == nil {
					n.neighbors[lc] = []int{}
				}
				n.neighbors[lc] = append(n.neighbors[lc], id)

				// Prune neighbors if needed
				if len(n.neighbors[lc]) > M {
					n.neighbors[lc] = h.selectNeighbors(n.neighbors[lc], M, n.Embedding)
				}
			}
		}
	}

	// Update entry point if this node has a higher level
	if level > h.maxLayer {
		h.maxLayer = level
		h.entryPoint = id
	}

	h.nodes[id] = node
	h.nodeCount++
}

// searchLayer searches for nearest neighbors at a specific layer
func (h *Index) searchLayer(queryEmbedding []float32, entryPoint, ef, layer int) []int {
	visited := make(map[int]bool)
	candidates := newMinHeap()
	results := newMaxHeap()

	// Calculate distance to entry point
	if entryNode, ok := h.nodes[entryPoint]; ok {
		dist := h.distance(queryEmbedding, entryNode.Embedding)
		candidates.push(entryPoint, dist)
		results.push(entryPoint, dist)
		visited[entryPoint] = true
	}

	for candidates.len() > 0 {
		currentID, currentDist := candidates.pop()

		// If we have enough results and current is worse than worst result, stop
		if results.len() > 0 && currentDist > results.peekDist() {
			break
		}

		currentNode, ok := h.nodes[currentID]
		if !ok || currentNode.neighbors[layer] == nil {
			continue
		}

		// Check neighbors
		for _, neighborID := range currentNode.neighbors[layer] {
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			if neighborNode, ok := h.nodes[neighborID]; ok {
				dist := h.distance(queryEmbedding, neighborNode.Embedding)

				if results.len() < ef {
					candidates.push(neighborID, dist)
					results.push(neighborID, dist)
				} else if dist < results.peekDist() {
					candidates.push(neighborID, dist)
					results.push(neighborID, dist)
					if results.len() > ef {
						results.pop()
					}
				}
			}
		}
	}

	return results.items()
}

// selectNeighbors selects the best neighbors by sorting by distance
func (h *Index) selectNeighbors(candidateIDs []int, m int, queryEmb []float32) []int {
	if len(queryEmb) == 0 {
		return []int{}
	}

	if len(candidateIDs) <= m {
		return candidateIDs
	}

	// Create a temporary slice with distances for sorting
	type neighborDist struct {
		id   int
		dist float32
	}

	neighbors := make([]neighborDist, 0, len(candidateIDs))

	// Compute distance from query to each candidate
	for _, id := range candidateIDs {
		if node, ok := h.nodes[id]; ok {
			if len(node.Embedding) != len(queryEmb) {
				continue // Skip dimension mismatch
			}
			neighbors = append(neighbors, neighborDist{
				id:   id,
				dist: h.distance(queryEmb, node.Embedding),
			})
		}
	}

	// Sort by distance (ascending - smallest distance first)
	// Use selection sort since m is typically small (16-32)
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

	// Return the m nearest neighbors
	result := make([]int, 0, m)
	for i := 0; i < m && i < len(neighbors); i++ {
		result = append(result, neighbors[i].id)
	}
	return result
}

// distance calculates cosine distance (lower is more similar)
// Uses negative dot product since embeddings are normalized
func (h *Index) distance(a, b []float32) float32 {
	dotProduct := dotProductSIMD(a, b)
	return -dotProduct // Negate so higher similarity = lower distance
}

// sortBySimDesc sorts SearchResults by similarity in descending order
func sortBySimDesc(results []SearchResult) {
	// Simple insertion sort - efficient for small slices
	for i := 1; i < len(results); i++ {
		for j := i; j > 0 && results[j].Similarity > results[j-1].Similarity; j-- {
			results[j], results[j-1] = results[j-1], results[j]
		}
	}
}

// ===== Priority Queue Implementations =====

type heapItem struct {
	id   int
	dist float32
}

type minHeap struct {
	data []heapItem
}

func newMinHeap() *minHeap {
	return &minHeap{data: []heapItem{}}
}

func (h *minHeap) push(id int, dist float32) {
	h.data = append(h.data, heapItem{id, dist})
	h.bubbleUp(len(h.data) - 1)
}

func (h *minHeap) pop() (int, float32) {
	if len(h.data) == 0 {
		return -1, 0
	}
	result := h.data[0]
	h.data[0] = h.data[len(h.data)-1]
	h.data = h.data[:len(h.data)-1]
	if len(h.data) > 0 {
		h.bubbleDown(0)
	}
	return result.id, result.dist
}

func (h *minHeap) len() int {
	return len(h.data)
}

func (h *minHeap) bubbleUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.data[i].dist >= h.data[parent].dist {
			break
		}
		h.data[i], h.data[parent] = h.data[parent], h.data[i]
		i = parent
	}
}

func (h *minHeap) bubbleDown(i int) {
	for {
		left := 2*i + 1
		right := 2*i + 2
		smallest := i

		if left < len(h.data) && h.data[left].dist < h.data[smallest].dist {
			smallest = left
		}
		if right < len(h.data) && h.data[right].dist < h.data[smallest].dist {
			smallest = right
		}
		if smallest == i {
			break
		}
		h.data[i], h.data[smallest] = h.data[smallest], h.data[i]
		i = smallest
	}
}

type maxHeap struct {
	data []heapItem
}

func newMaxHeap() *maxHeap {
	return &maxHeap{data: []heapItem{}}
}

func (h *maxHeap) push(id int, dist float32) {
	h.data = append(h.data, heapItem{id, dist})
	h.bubbleUp(len(h.data) - 1)
}

func (h *maxHeap) pop() (int, float32) {
	if len(h.data) == 0 {
		return -1, 0
	}
	result := h.data[0]
	h.data[0] = h.data[len(h.data)-1]
	h.data = h.data[:len(h.data)-1]
	if len(h.data) > 0 {
		h.bubbleDown(0)
	}
	return result.id, result.dist
}

func (h *maxHeap) len() int {
	return len(h.data)
}

func (h *maxHeap) peekDist() float32 {
	if len(h.data) == 0 {
		return math.MaxFloat32
	}
	return h.data[0].dist
}

func (h *maxHeap) items() []int {
	result := make([]int, len(h.data))
	for i, item := range h.data {
		result[i] = item.id
	}
	return result
}

func (h *maxHeap) bubbleUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.data[i].dist <= h.data[parent].dist {
			break
		}
		h.data[i], h.data[parent] = h.data[parent], h.data[i]
		i = parent
	}
}

func (h *maxHeap) bubbleDown(i int) {
	for {
		left := 2*i + 1
		right := 2*i + 2
		largest := i

		if left < len(h.data) && h.data[left].dist > h.data[largest].dist {
			largest = left
		}
		if right < len(h.data) && h.data[right].dist > h.data[largest].dist {
			largest = right
		}
		if largest == i {
			break
		}
		h.data[i], h.data[largest] = h.data[largest], h.data[i]
		i = largest
	}
}
