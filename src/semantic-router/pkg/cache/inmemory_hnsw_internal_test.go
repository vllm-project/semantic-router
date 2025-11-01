package cache

import (
	"slices"
	"testing"
)

func TestSearchLayerHeapManagement(t *testing.T) {
	t.Run("retains the closest neighbor when ef is saturated", func(t *testing.T) {
		// Regression fixture: with the previous max-heap candidates/min-heap results
		// mix, trimming to ef would evict the best element instead of the worst.
		queryEmbedding := []float32{1.0}

		entries := []CacheEntry{
			{Embedding: []float32{0.1}}, // entry point has low similarity
			{Embedding: []float32{1.0}}, // neighbor is the true nearest
		}

		entryNode := &HNSWNode{
			entryIndex: 0,
			neighbors: map[int][]int{
				0: {1},
			},
			maxLayer: 0,
		}

		neighborNode := &HNSWNode{
			entryIndex: 1,
			neighbors: map[int][]int{
				0: {0},
			},
			maxLayer: 0,
		}

		index := &HNSWIndex{
			nodes: []*HNSWNode{entryNode, neighborNode},
			nodeIndex: map[int]*HNSWNode{
				0: entryNode,
				1: neighborNode,
			},
			entryPoint:     0,
			maxLayer:       0,
			efConstruction: 2,
			M:              1,
			Mmax:           1,
			Mmax0:          2,
			ml:             1,
		}

		results := index.searchLayer(queryEmbedding, index.entryPoint, 1, 0, entries)

		if !slices.Contains(results, 1) {
			t.Fatalf("expected results to contain best neighbor 1, got %v", results)
		}
		if slices.Contains(results, 0) {
			t.Fatalf("expected results to drop entry point 0 once ef trimmed, got %v", results)
		}
	})

	t.Run("continues exploring even when next candidate looks worse", func(t *testing.T) {
		// Regression fixture: the break condition used the wrong polarity so the
		// search stopped before expanding the intermediate (worse) vertex, making
		// the actual best neighbor unreachable.
		queryEmbedding := []float32{1.0}

		entries := []CacheEntry{
			{Embedding: []float32{0.2}},  // entry point
			{Embedding: []float32{0.05}}, // intermediate node with poor similarity
			{Embedding: []float32{1.0}},  // hidden best match
		}

		entryNode := &HNSWNode{
			entryIndex: 0,
			neighbors: map[int][]int{
				0: {1},
			},
			maxLayer: 0,
		}

		intermediateNode := &HNSWNode{
			entryIndex: 1,
			neighbors: map[int][]int{
				0: {0, 2},
			},
			maxLayer: 0,
		}

		bestNode := &HNSWNode{
			entryIndex: 2,
			neighbors: map[int][]int{
				0: {1},
			},
			maxLayer: 0,
		}

		index := &HNSWIndex{
			nodes: []*HNSWNode{entryNode, intermediateNode, bestNode},
			nodeIndex: map[int]*HNSWNode{
				0: entryNode,
				1: intermediateNode,
				2: bestNode,
			},
			entryPoint:     0,
			maxLayer:       0,
			efConstruction: 2,
			M:              1,
			Mmax:           1,
			Mmax0:          2,
			ml:             1,
		}

		results := index.searchLayer(queryEmbedding, index.entryPoint, 2, 0, entries)

		if !slices.Contains(results, 2) {
			t.Fatalf("expected results to reach best neighbor 2 via intermediate node, got %v", results)
		}
	})
}
