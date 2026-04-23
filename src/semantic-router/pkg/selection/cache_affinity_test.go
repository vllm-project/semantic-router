package selection

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func makeAffCtx(turnIndex, historyTokens, contextTokens int, previousModel, previousResponseID string, windows map[string]int) *CacheAffinityContext {
	return &CacheAffinityContext{
		TurnIndex:           turnIndex,
		PreviousModel:       previousModel,
		PreviousResponseID:  previousResponseID,
		HistoryTokens:       historyTokens,
		ContextTokens:       contextTokens,
		ModelContextWindows: windows,
	}
}

func makeCandidates(names ...string) []config.ModelRef {
	refs := make([]config.ModelRef, len(names))
	for i, n := range names {
		refs[i] = config.ModelRef{Model: n}
	}
	return refs
}

func equalBase(candidates ...string) map[string]float64 {
	m := make(map[string]float64, len(candidates))
	for _, c := range candidates {
		m[c] = 0.5
	}
	return m
}

func TestCacheAffinity_FirstTurn_ZeroAdjustment(t *testing.T) {
	affCtx := makeAffCtx(0, 0, 500, "", "", nil)
	result := ComputeCacheAffinityAdjustments(affCtx, makeCandidates("a", "b"), equalBase("a", "b"))

	assert.Equal(t, 0.0, result.WReq)
	assert.Equal(t, 0.0, result.LambdaReq)
	for _, adj := range result.Adjustments {
		assert.Equal(t, 0.0, adj)
	}
}

func TestCacheAffinity_PreviousResponseID_TriggersContinuation(t *testing.T) {
	affCtx := makeAffCtx(0, 0, 500, "model-a", "resp-xyz", nil)
	result := ComputeCacheAffinityAdjustments(affCtx, makeCandidates("model-a", "model-b"), equalBase("model-a", "model-b"))

	assert.Greater(t, result.LambdaReq, 0.0)
	assert.GreaterOrEqual(t, result.WReq, wrPreviousResponseFloor)
	assert.Greater(t, result.Adjustments["model-a"], 0.0)
}

func TestCacheAffinity_SingleCandidate_ZeroAdjustment(t *testing.T) {
	affCtx := makeAffCtx(3, 1000, 1500, "model-a", "", nil)
	result := ComputeCacheAffinityAdjustments(affCtx, makeCandidates("model-a"), map[string]float64{"model-a": 0.8})

	assert.Equal(t, 0.0, result.LambdaReq)
	assert.Equal(t, 0.0, result.Adjustments["model-a"])
}

func TestCacheAffinity_StrongBaseWinner_CollapseToZero(t *testing.T) {
	affCtx := makeAffCtx(4, 2000, 3000, "model-a", "", nil)
	baseScores := map[string]float64{"model-a": 0.9, "model-b": 0.1}
	result := ComputeCacheAffinityAdjustments(affCtx, makeCandidates("model-a", "model-b"), baseScores)

	assert.Equal(t, 0.0, result.LambdaReq)
	for _, adj := range result.Adjustments {
		assert.Equal(t, 0.0, adj)
	}
}

func TestCacheAffinity_SameModelHighReuse_PositiveAdjustment(t *testing.T) {
	affCtx := makeAffCtx(3, 3200, 4000, "model-a", "", nil)
	result := ComputeCacheAffinityAdjustments(affCtx, makeCandidates("model-a", "model-b"), equalBase("model-a", "model-b"))

	require.Greater(t, result.LambdaReq, 0.0)
	assert.Greater(t, result.Adjustments["model-a"], 0.0)
	assert.Less(t, result.Adjustments["model-b"], 0.0)
}

func TestCacheAffinity_DifferentModel_NegativeWhenHeavyContinuation(t *testing.T) {
	affCtx := makeAffCtx(4, 3000, 3500, "model-a", "", nil)
	result := ComputeCacheAffinityAdjustments(affCtx, makeCandidates("model-a", "model-b"), equalBase("model-a", "model-b"))

	require.Greater(t, result.WReq, 0.5)
	assert.Less(t, result.Adjustments["model-b"], 0.0)
}

func TestCacheAffinity_ContextOverflow_ZeroFit(t *testing.T) {
	windows := map[string]int{
		"small-model": 2048,
		"large-model": 128000,
	}
	overflowCtx := makeAffCtx(2, 2000, 3000, "small-model", "", windows)
	neutralCtx := makeAffCtx(2, 2000, 3000, "small-model", "", nil)
	overflow := ComputeCacheAffinityAdjustments(
		overflowCtx,
		makeCandidates("small-model", "large-model"),
		equalBase("small-model", "large-model"),
	)
	neutral := ComputeCacheAffinityAdjustments(
		neutralCtx,
		makeCandidates("small-model", "large-model"),
		equalBase("small-model", "large-model"),
	)

	require.Greater(t, overflow.LambdaReq, 0.0)
	assert.Less(t, overflow.Adjustments["small-model"], neutral.Adjustments["small-model"])
	assert.Greater(t, overflow.Adjustments["large-model"], neutral.Adjustments["large-model"])
}

func TestCacheAffinity_UnknownWindow_NeutralFit(t *testing.T) {
	affCtx := makeAffCtx(2, 1000, 2000, "model-a", "", nil)
	result := ComputeCacheAffinityAdjustments(affCtx, makeCandidates("model-a", "model-b"), equalBase("model-a", "model-b"))

	require.Greater(t, result.LambdaReq, 0.0)
	assert.Greater(t, result.Adjustments["model-a"], 0.0)
}

func TestCacheAffinity_AdjustmentBounded(t *testing.T) {
	windows := map[string]int{"model-a": 128000, "model-b": 128000}
	affCtx := makeAffCtx(10, 8000, 8000, "model-a", "resp-1", windows)
	result := ComputeCacheAffinityAdjustments(affCtx, makeCandidates("model-a", "model-b"), equalBase("model-a", "model-b"))

	for _, adj := range result.Adjustments {
		assert.LessOrEqual(t, adj, affinityMaxLambda)
		assert.GreaterOrEqual(t, adj, -affinityMaxLambda)
	}
}

func TestComputeFitM(t *testing.T) {
	cases := []struct {
		ctx    int
		window int
		want   float64
	}{
		{500, 0, 0.5},
		{1000, 4000, 1.0},
		{2200, 4000, 0.7},
		{3200, 4000, 0.3},
		{5000, 4000, 0.0},
		{2000, 4000, 1.0},
		{3000, 4000, 0.7},
		{4000, 4000, 0.3},
	}
	for _, tc := range cases {
		got := computeFitM(tc.ctx, tc.window)
		assert.Equal(t, tc.want, got)
	}
}

func TestComputeGap12(t *testing.T) {
	candidates := []config.ModelRef{{Model: "a"}, {Model: "b"}, {Model: "c"}}

	t.Run("standard gap", func(t *testing.T) {
		scores := map[string]float64{"a": 0.9, "b": 0.6, "c": 0.3}
		assert.InDelta(t, 0.3, computeGap12(scores, candidates), 1e-9)
	})
	t.Run("tied scores", func(t *testing.T) {
		scores := map[string]float64{"a": 0.5, "b": 0.5, "c": 0.5}
		assert.Equal(t, 0.0, computeGap12(scores, candidates))
	})
	t.Run("single candidate", func(t *testing.T) {
		single := []config.ModelRef{{Model: "a"}}
		assert.Equal(t, 0.0, computeGap12(map[string]float64{"a": 0.8}, single))
	})
}
