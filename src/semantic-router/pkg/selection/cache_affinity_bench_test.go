package selection

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Baseline scenarios used across all benchmarks.
var (
	benchCandidates = []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
		{Model: "model-c"},
	}
	benchWindows = map[string]int{
		"model-a": 128000,
		"model-b": 32000,
		"model-c": 8192,
	}
	benchBaseScores = map[string]float64{
		"model-a": 0.52,
		"model-b": 0.50,
		"model-c": 0.48,
	}
)

// BenchmarkComputeCacheAffinityAdjustments_NoOp measures the fast-path cost
// (first turn — all early exits fire, zero allocations expected).
func BenchmarkComputeCacheAffinityAdjustments_NoOp(b *testing.B) {
	affCtx := &CacheAffinityContext{
		TurnIndex:           0,
		HistoryTokens:       0,
		PreviousResponseID:  "",
		ContextTokens:       500,
		ModelContextWindows: benchWindows,
	}
	b.ReportAllocs()
	for b.Loop() {
		ComputeCacheAffinityAdjustments(affCtx, benchCandidates, benchBaseScores)
	}
}

// BenchmarkComputeCacheAffinityAdjustments_Active measures the full-path cost
// for a typical multi-turn session with 3 candidates and known windows.
func BenchmarkComputeCacheAffinityAdjustments_Active(b *testing.B) {
	affCtx := &CacheAffinityContext{
		TurnIndex:           3,
		PreviousModel:       "model-a",
		HistoryTokens:       2048,
		ContextTokens:       3000,
		ModelContextWindows: benchWindows,
	}
	b.ReportAllocs()
	for b.Loop() {
		ComputeCacheAffinityAdjustments(affCtx, benchCandidates, benchBaseScores)
	}
}

// BenchmarkComputeCacheAffinityAdjustments_StrongWinner measures cost when
// gap12 collapses lambda to zero (ambiguity gate fires early).
func BenchmarkComputeCacheAffinityAdjustments_StrongWinner(b *testing.B) {
	affCtx := &CacheAffinityContext{
		TurnIndex:           3,
		PreviousModel:       "model-a",
		HistoryTokens:       2048,
		ContextTokens:       3000,
		ModelContextWindows: benchWindows,
	}
	strongScores := map[string]float64{
		"model-a": 0.9,
		"model-b": 0.1,
		"model-c": 0.1,
	}
	b.ReportAllocs()
	for b.Loop() {
		ComputeCacheAffinityAdjustments(affCtx, benchCandidates, strongScores)
	}
}

// BenchmarkComputeGap12 isolates the gap computation used for the ambiguity gate.
func BenchmarkComputeGap12(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		computeGap12(benchBaseScores, benchCandidates)
	}
}

// BenchmarkComputeFitM covers the context-fit switch statement.
func BenchmarkComputeFitM(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		computeFitM(3000, 32000)
	}
}
