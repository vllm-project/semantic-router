package config

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateReMoMAlgorithmConfigCompletionLimit(t *testing.T) {
	positive := 512
	valid := &ReMoMAlgorithmConfig{
		BreadthSchedule:     []int{1},
		MaxCompletionTokens: &positive,
	}
	require.NoError(t, ValidateReMoMAlgorithmConfig(valid))
	require.NoError(t, ValidateReMoMAlgorithmConfig(&ReMoMAlgorithmConfig{
		BreadthSchedule: []int{1},
	}))

	for _, value := range []int{0, -1} {
		invalid := &ReMoMAlgorithmConfig{
			BreadthSchedule:     []int{1},
			MaxCompletionTokens: &value,
		}
		err := ValidateReMoMAlgorithmConfig(invalid)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "max_completion_tokens must be >= 1 when set")
	}
}

func TestEstimatedReMoMUpstreamCalls(t *testing.T) {
	assert.Equal(t, 1, EstimatedReMoMUpstreamCalls(nil))
	assert.Equal(t, 6, EstimatedReMoMUpstreamCalls([]int{3, 2}))
	assert.Equal(t, 101, EstimatedReMoMUpstreamCalls([]int{100}))

	// The sum saturates at MaxInt rather than overflowing into a small or
	// negative value that would slip past the budget comparison.
	assert.Equal(t, math.MaxInt, EstimatedReMoMUpstreamCalls([]int{math.MaxInt}))
	assert.Equal(t, math.MaxInt, EstimatedReMoMUpstreamCalls([]int{math.MaxInt - 1, 2}))
	assert.Equal(t, math.MaxInt, EstimatedReMoMUpstreamCalls([]int{math.MaxInt, math.MaxInt}))
}

func TestValidateReMoMAlgorithmConfigCallBudget(t *testing.T) {
	// Exact boundary: the last parallel round plus the final synthesis equals
	// the per-request budget.
	exact := &ReMoMAlgorithmConfig{BreadthSchedule: []int{MaxUpstreamCallsPerRequest - 1}}
	require.NoError(t, ValidateReMoMAlgorithmConfig(exact))

	// One parallel call over the boundary fails with the observed count.
	over := &ReMoMAlgorithmConfig{BreadthSchedule: []int{MaxUpstreamCallsPerRequest}}
	err := ValidateReMoMAlgorithmConfig(over)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "requests 33 upstream calls")
	assert.Contains(t, err.Error(), "exceeding the per-request limit of 32")

	// The audited amplification example in the issue must be rejected.
	err = ValidateReMoMAlgorithmConfig(&ReMoMAlgorithmConfig{BreadthSchedule: []int{100}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "requests 101 upstream calls")

	// A maintained multi-round schedule keeps its existing behaviour.
	require.NoError(t, ValidateReMoMAlgorithmConfig(&ReMoMAlgorithmConfig{BreadthSchedule: []int{3, 2}}))

	// Integer-boundary schedules must fail closed rather than overflow the sum.
	for _, schedule := range [][]int{
		{math.MaxInt},
		{math.MaxInt, math.MaxInt},
		{math.MaxInt - 1},
		{MaxUpstreamCallsPerRequest - 1, math.MaxInt},
	} {
		err := ValidateReMoMAlgorithmConfig(&ReMoMAlgorithmConfig{BreadthSchedule: schedule})
		require.Error(t, err, "schedule %v must be rejected", schedule)
		assert.Contains(t, err.Error(), "exceeding the per-request limit of 32")
	}
}
