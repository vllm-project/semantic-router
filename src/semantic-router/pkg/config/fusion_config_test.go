package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateFusionAlgorithmConfigAnalysisOverrides(t *testing.T) {
	cfg := &FusionAlgorithmConfig{
		AnalysisOverrides: []FusionModelOverride{
			{Model: "panel-a", Temperature: floatPtr(0.0)},
			{Model: "panel-b", Temperature: floatPtr(0.8), MaxCompletionTokens: 256},
		},
	}

	require.NoError(t, ValidateFusionAlgorithmConfig(cfg))
}

func TestValidateFusionAlgorithmConfigRejectsInvalidAnalysisOverrides(t *testing.T) {
	cfg := &FusionAlgorithmConfig{
		AnalysisOverrides: []FusionModelOverride{
			{Model: "panel-a", Temperature: floatPtr(0.2)},
			{Model: "panel-a", Temperature: floatPtr(0.4)},
		},
	}

	err := ValidateFusionAlgorithmConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "duplicated")
}

func TestFusionAnalysisModeDefaultsAndValidation(t *testing.T) {
	assert.Equal(t, FusionAnalysisModeSeparate, EffectiveFusionAnalysisMode(""))
	assert.Equal(t, FusionAnalysisModeSeparate, EffectiveFusionAnalysisMode("  separate  "))

	for _, mode := range []string{
		FusionAnalysisModeSeparate,
		FusionAnalysisModeOneCall,
		FusionAnalysisModeNone,
	} {
		t.Run(mode, func(t *testing.T) {
			require.NoError(t, ValidateFusionAlgorithmConfig(&FusionAlgorithmConfig{AnalysisMode: mode}))
		})
	}

	err := ValidateFusionAlgorithmConfig(&FusionAlgorithmConfig{AnalysisMode: "automatic"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "analysis_mode must be one of")
}

func TestFusionAnalysisModeRejectsUnusedAnalysisTemplate(t *testing.T) {
	for _, mode := range []string{FusionAnalysisModeOneCall, FusionAnalysisModeNone} {
		t.Run(mode, func(t *testing.T) {
			err := ValidateFusionAlgorithmConfig(&FusionAlgorithmConfig{
				AnalysisMode:     mode,
				AnalysisTemplate: "compare {{responses}}",
			})
			require.Error(t, err)
			assert.Contains(t, err.Error(), "analysis_template requires analysis_mode=\"separate\"")
		})
	}
}

func TestFusionRequestConfigValidateAnalysisOverrides(t *testing.T) {
	cfg := &FusionRequestConfig{
		AnalysisOverrides: []FusionModelOverride{
			{Model: "panel-a", Temperature: floatPtr(0.2)},
		},
	}

	require.NoError(t, cfg.Validate())
}

func floatPtr(v float64) *float64 {
	return &v
}
