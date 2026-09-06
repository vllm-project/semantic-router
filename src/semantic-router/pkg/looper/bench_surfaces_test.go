package looper

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestLooperBenchmarkSurfaces guards the Tier-1 benchmarks against silently
// measuring nothing. It drives the SAME fixtures and functions the *_bench_test.go
// files use and asserts they still produce meaningful output. If a fixture goes
// empty or a benched function becomes a no-op, the benchmark would still "pass"
// (fast, no error) yet measure nothing; this test fails in the normal unit gate
// instead, keeping the perf gate honest (cf. the silently-green gate in #2398).
func TestLooperBenchmarkSurfaces(t *testing.T) {
	t.Run("ReMoM", func(t *testing.T) {
		require.Len(t, distributeByWeight(benchNumCalls, benchWeightedRefs, 42), benchNumCalls)
		require.Len(t, distributeRoundRobin(benchNumCalls, benchFlatRefs), benchNumCalls)

		looper := NewReMoMLooper(&config.LooperConfig{})
		calls := looper.distributeCallsToModels(&config.ReMoMAlgorithmConfig{ShuffleSeed: 42}, benchNumCalls, benchWeightedRefs)
		require.Len(t, calls, benchNumCalls)
	})

	t.Run("Base", func(t *testing.T) {
		chunks := splitIntoChunks(benchChunkText, 50)
		require.NotEmpty(t, chunks)
		require.Equal(t, benchChunkText, strings.Join(chunks, ""), "chunks must reconstruct the input")

		name, args, ok := parseTaggedToolCall(benchTaggedToolCall)
		require.True(t, ok)
		require.Equal(t, "get_weather", name)
		require.NotEmpty(t, args)
	})

	t.Run("Fusion", func(t *testing.T) {
		looper := NewFusionLooper(&config.LooperConfig{})
		cfg := looper.resolveFusionExecutionConfig(&Request{ModelRefs: benchFusionRefs})
		require.Len(t, cfg.AnalysisModels, len(benchFusionRefs))

		require.Contains(t, formatPanelResponses(benchFusionPanel), "Paris")

		prompt := buildFusionFinalPrompt(fusionExecutionConfig{Model: "judge"}, "q", "", benchFusionPanel, benchFusionAnalysis)
		require.Contains(t, prompt, "Paris")

		analysis, err := parseFusionAnalysis(benchFusionAnalysisJSON)
		require.NoError(t, err)
		require.NotEmpty(t, analysis.Consensus)
	})

	t.Run("Flow", func(t *testing.T) {
		plan, err := buildStaticWorkflowPlan(benchWorkflowStaticCfg)
		require.NoError(t, err)
		require.Len(t, plan.Steps, len(benchWorkflowRoles))

		require.NotEmpty(t, formatWorkflowStepResults(benchWorkflowStepResults))
		require.NotEmpty(t, buildWorkflowFinalPrompt(benchWorkflowPlan, "q", "", benchWorkflowStepResults))
	})
}
