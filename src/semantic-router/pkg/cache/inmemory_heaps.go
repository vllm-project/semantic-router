//go:build !windows && cgo

package cache

import (
	"math"
	"sort"
)

// Helper priority queue implementations for HNSW

type heapItem struct {
	index int
	dist  float32
}

type minHeap struct {
	data []heapItem
}

func newMinHeap() *minHeap {
	return &minHeap{data: []heapItem{}}
}

func (h *minHeap) push(index int, dist float32) {
	h.data = append(h.data, heapItem{index, dist})
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
	return result.index, result.dist
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

func (h *maxHeap) push(index int, dist float32) {
	h.data = append(h.data, heapItem{index, dist})
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
	return result.index, result.dist
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

// sortedIndicesAsc exports the heap contents as entry indices ordered by ascending distance (nearest first).
func (h *maxHeap) sortedIndicesAsc() []int {
	result := make([]heapItem, len(h.data))
	copy(result, h.data)

	sort.Slice(result, func(i, j int) bool {
		return result[i].dist < result[j].dist
	})

	indices := make([]int, len(result))
	for i, item := range result {
		indices[i] = item.index
	}

	return indices
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
