package looper

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Shared fixtures for the ReMoM call-distribution micro-benchmarks. These
// exercise the pure, deterministic core of ReMoM (no network, no LLM calls),
// so they yield stable regression baselines for the routing hot path.
var (
	// benchWeightedRefs drives the weighted strategy's full path:
	// proportional floor + largest-remainder top-up + two stable sorts + shuffle.
	benchWeightedRefs = []config.ModelRef{
		{Model: "model-a", Weight: 5},
		{Model: "model-b", Weight: 3},
		{Model: "model-c", Weight: 2},
		{Model: "model-d", Weight: 1},
	}
	// benchFlatRefs has no weights, for the equal / round-robin / first-only paths.
	benchFlatRefs = []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
		{Model: "model-c"},
		{Model: "model-d"},
	}
)

// benchNumCalls is a representative per-round breadth (K) for the distribution.
const benchNumCalls = 12

// BenchmarkReMoM_DistributeByWeight measures the default (and most expensive)
// strategy: proportional split with remainder handling followed by a shuffle.
func BenchmarkReMoM_DistributeByWeight(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		distributeByWeight(benchNumCalls, benchWeightedRefs, 42)
	}
}

// BenchmarkReMoM_DistributeEqually measures even distribution with a final shuffle.
func BenchmarkReMoM_DistributeEqually(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		distributeEqually(benchNumCalls, benchFlatRefs, 42)
	}
}

// BenchmarkReMoM_DistributeRoundRobin measures the cheapest strategy (no shuffle).
func BenchmarkReMoM_DistributeRoundRobin(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		distributeRoundRobin(benchNumCalls, benchFlatRefs)
	}
}

// BenchmarkReMoM_DistributeFirstOnly measures the single-model (PaCoRe) path.
func BenchmarkReMoM_DistributeFirstOnly(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		distributeFirstOnly(benchNumCalls, benchFlatRefs)
	}
}

// BenchmarkReMoM_ShuffleModelCalls isolates the seeded shuffle (rand source
// construction + Fisher-Yates) that weighted and equal distribution apply.
func BenchmarkReMoM_ShuffleModelCalls(b *testing.B) {
	calls := make([]ModelCall, benchNumCalls)
	for i := range calls {
		calls[i] = ModelCall{Model: "model-a"}
	}
	b.ReportAllocs()
	for b.Loop() {
		shuffleModelCalls(calls, 42)
	}
}

// BenchmarkReMoM_DistributeCallsToModels measures the strategy-dispatch wrapper
// that callers actually invoke, using the default weighted strategy.
func BenchmarkReMoM_DistributeCallsToModels(b *testing.B) {
	looper := NewReMoMLooper(&config.LooperConfig{})
	cfg := &config.ReMoMAlgorithmConfig{ShuffleSeed: 42}
	b.ReportAllocs()
	for b.Loop() {
		looper.distributeCallsToModels(cfg, benchNumCalls, benchWeightedRefs)
	}
}
