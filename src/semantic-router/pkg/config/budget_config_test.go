package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateBudgetConfigNilIsAlwaysValid(t *testing.T) {
	require.NoError(t, ValidateBudgetConfig("rl_driven", nil))
	require.NoError(t, ValidateBudgetConfig("confidence", nil))
}

func TestValidateBudgetConfigRejectsUnsupportedAlgorithmTypes(t *testing.T) {
	for _, algorithmType := range []string{"rl_driven", "base", "elo", "hybrid"} {
		err := ValidateBudgetConfig(algorithmType, &BudgetConfig{MaxPromptTokens: 100})
		require.Error(t, err, "type %q should reject a configured budget", algorithmType)
		assert.Contains(t, err.Error(), "only supported for algorithm types")
	}
}

func TestValidateBudgetConfigAcceptsLooperExecutedAlgorithmTypes(t *testing.T) {
	for _, algorithmType := range SupportedLooperAlgorithmTypes() {
		err := ValidateBudgetConfig(algorithmType, &BudgetConfig{MaxPromptTokens: 100})
		require.NoError(t, err, "type %q should accept a valid budget", algorithmType)
	}
}

func TestValidateBudgetConfigRejectsNegativeLimits(t *testing.T) {
	cases := []BudgetConfig{
		{MaxPromptTokens: -1},
		{MaxCompletionTokens: -1},
		{MaxTotalTokens: -1},
		{MaxEstimatedCost: -0.01},
		{MaxWallTimeMs: -1},
	}
	for _, cfg := range cases {
		cfg := cfg
		err := ValidateBudgetConfig("confidence", &cfg)
		require.Error(t, err, "%+v should be rejected", cfg)
		assert.Contains(t, err.Error(), "cannot be negative")
	}
}

func TestValidateBudgetConfigRejectsInconsistentTokenLimits(t *testing.T) {
	err := ValidateBudgetConfig("confidence", &BudgetConfig{MaxPromptTokens: 100, MaxTotalTokens: 50})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "max_prompt_tokens")

	err = ValidateBudgetConfig("confidence", &BudgetConfig{MaxCompletionTokens: 100, MaxTotalTokens: 50})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "max_completion_tokens")
}

func TestValidateBudgetConfigAcceptsConsistentLimits(t *testing.T) {
	err := ValidateBudgetConfig("fusion", &BudgetConfig{
		MaxPromptTokens:     100,
		MaxCompletionTokens: 100,
		MaxTotalTokens:      200,
		MaxEstimatedCost:    1.5,
		MaxWallTimeMs:       30000,
	})
	require.NoError(t, err)
}
