package config

import (
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
