package latency

import "slices"

// window holds a model's most recent observations in arrival order, and the
// same values in ascending order, so a percentile is read without copying or
// sorting under the cache lock. Once full, each observation replaces the
// oldest, so the window never allocates again.
type window struct {
	ring []float64
	// next is the oldest observation's index once the ring is full.
	next   int
	sorted []float64
}

func newWindow(value float64) window {
	return window{ring: []float64{value}, sorted: []float64{value}}
}

// add records value, dropping the oldest observation beyond size.
func (w *window) add(value float64, size int) {
	if len(w.ring) < size {
		w.ring = append(w.ring, value)
		at, _ := slices.BinarySearch(w.sorted, value)
		w.sorted = slices.Insert(w.sorted, at, value)
		return
	}
	replaceSorted(w.sorted, w.ring[w.next], value)
	w.ring[w.next] = value
	w.next = (w.next + 1) % size
}

func (w *window) len() int { return len(w.sorted) }

// replaceSorted replaces one occurrence of old with value in an ascending
// slice, keeping it ascending, with a single shift. The order is the one
// sort.Float64s produces, NaN first.
func replaceSorted(sorted []float64, old, value float64) {
	from, _ := slices.BinarySearch(sorted, old)
	to, _ := slices.BinarySearch(sorted, value)
	if to > from {
		copy(sorted[from:to-1], sorted[from+1:to])
		sorted[to-1] = value
		return
	}
	copy(sorted[to+1:from+1], sorted[to:from])
	sorted[to] = value
}
