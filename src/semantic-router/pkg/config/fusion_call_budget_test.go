package config

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEstimatedFusionUpstreamCalls(t *testing.T) {
	assert.Equal(t, 32, EstimatedFusionUpstreamCalls(30, FusionAnalysisModeSeparate))
	assert.Equal(t, 31, EstimatedFusionUpstreamCalls(30, FusionAnalysisModeOneCall))
	assert.Equal(t, 31, EstimatedFusionUpstreamCalls(30, FusionAnalysisModeNone))
	// An unset mode defaults to separate.
	assert.Equal(t, 34, EstimatedFusionUpstreamCalls(32, ""))
}

func TestValidateFusionCallBudget(t *testing.T) {
	refs := func(n int) []ModelRef {
		out := make([]ModelRef, 0, n)
		for i := 0; i < n; i++ {
			out = append(out, ModelRef{Model: fmt.Sprintf("ref-%d", i)})
		}
		return out
	}
	names := func(n int) []string {
		out := make([]string, 0, n)
		for i := 0; i < n; i++ {
			out = append(out, fmt.Sprintf("panel-%d", i))
		}
		return out
	}

	// Exact boundary: 30 panel calls + 2 separate-mode judge stages = budget.
	require.NoError(t, validateDecisionFusionAlgorithm("fuse", refs(30), &FusionAlgorithmConfig{}))
	require.NoError(t, validateDecisionFusionAlgorithm("fuse", refs(30), &FusionAlgorithmConfig{
		AnalysisMode: FusionAnalysisModeSeparate,
	}))
	// one_call and none dispatch a single judge stage, so 31 panel calls fit.
	require.NoError(t, validateDecisionFusionAlgorithm("fuse", refs(31), &FusionAlgorithmConfig{
		AnalysisMode: FusionAnalysisModeOneCall,
	}))
	require.NoError(t, validateDecisionFusionAlgorithm("fuse", refs(31), &FusionAlgorithmConfig{
		AnalysisMode: FusionAnalysisModeNone,
	}))

	// 32 separate-mode panel calls would make 34 upstream calls.
	err := validateDecisionFusionAlgorithm("fuse", refs(32), &FusionAlgorithmConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "requests 34 upstream calls")
	assert.Contains(t, err.Error(), "exceeding the per-request limit of 32")

	// 32 none-mode panel calls would make 33 upstream calls.
	err = validateDecisionFusionAlgorithm("fuse", refs(32), &FusionAlgorithmConfig{
		AnalysisMode: FusionAnalysisModeNone,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "requests 33 upstream calls")

	// An explicit analysis_models panel overrides modelRefs, so the panel size
	// is what the estimator must use.
	err = validateDecisionFusionAlgorithm("fuse", refs(1), &FusionAlgorithmConfig{
		AnalysisModels: names(33),
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "fusion panel of 33 analysis models")
	assert.Contains(t, err.Error(), "requests 35 upstream calls")
}
