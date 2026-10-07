package latency

import (
	"math"
	"math/rand/v2"
	"slices"
	"sort"
	"testing"
)

// The window must hold exactly the last size observations, sorted the way
// sort.Float64s sorts a copy of them, duplicates and NaN included.
func TestWindowMatchesSortedSlidingWindow(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, size := range []int{1, 2, 3, 7, MaxTTFTHistorySize} {
		var w window
		var history []float64
		for i := 0; i < 3*size+5; i++ {
			value := float64(rng.IntN(size+3)) / 4
			switch rng.IntN(50) {
			case 0:
				value = math.NaN()
			case 1:
				value = math.Inf(1)
			}
			if i == 0 {
				w = newWindow(value)
			} else {
				w.add(value, size)
			}
			history = append(history, value)
			want := append([]float64(nil), history[max(0, len(history)-size):]...)
			sort.Float64s(want)
			if !sameFloats(w.sorted, want) {
				t.Fatalf("size %d after %d observations: sorted = %v, want %v", size, i+1, w.sorted, want)
			}
			ring := append([]float64(nil), w.ring...)
			sort.Float64s(ring)
			if !sameFloats(ring, want) {
				t.Fatalf("size %d after %d observations: ring holds %v, want %v", size, i+1, ring, want)
			}
		}
	}
}

func sameFloats(a, b []float64) bool {
	return slices.EqualFunc(a, b, func(x, y float64) bool { return x == y || (math.IsNaN(x) && math.IsNaN(y)) })
}

// Percentiles over a full window equal those of the last
// MaxTTFTHistorySize observations, sorted.
func TestPercentilesReadTheLastObservations(t *testing.T) {
	ResetTTFT()
	ResetTPOT()
	rng := rand.New(rand.NewPCG(3, 4))
	var history []float64
	for i := 0; i < 2*MaxTTFTHistorySize+17; i++ {
		value := 0.05 + rng.Float64()
		history = append(history, value)
		UpdateTTFT("m-window", value)
		UpdateTPOT("m-window", value)
	}
	last := append([]float64(nil), history[len(history)-MaxTTFTHistorySize:]...)
	sort.Float64s(last)
	for _, percentile := range []int{1, 10, 20, 50, 80, 95, 100} {
		want, _ := percentileFromSorted(last, float64(percentile)/100)
		if got, ok := GetTTFTPercentile("m-window", percentile); !ok || got != want {
			t.Errorf("TTFT p%d = %v, %v; want %v", percentile, got, ok, want)
		}
		if got, ok := GetTPOTPercentile("m-window", percentile); !ok || got != want {
			t.Errorf("TPOT p%d = %v, %v; want %v", percentile, got, ok, want)
		}
	}
	snap, ok := getTTFTSnapshot("m-window")
	if !ok {
		t.Fatal("no snapshot")
	}
	for _, c := range []struct {
		got, pct float64
	}{{snap.warm, 0.20}, {snap.ref, 0.50}, {snap.cold, 0.80}} {
		if want, _ := percentileFromSorted(last, c.pct); c.got != want {
			t.Errorf("snapshot p%v = %v, want %v", c.pct*100, c.got, want)
		}
	}
}
