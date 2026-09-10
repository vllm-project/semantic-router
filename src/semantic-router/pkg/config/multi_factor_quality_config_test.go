package config

import (
	"math"
	"strings"
	"testing"
)

func TestValidateDecisionMultiFactorQualityEvidence(t *testing.T) {
	qualityFloor := 40.0
	nan := math.NaN()
	tests := []struct {
		name       string
		quality    *QualityEvidenceConfig
		wantErrSub string
	}{
		{name: "valid strict", quality: &QualityEvidenceConfig{Index: "vllm-sr/coding@1.0.0", OnMissing: "exclude"}},
		{name: "valid inclusive", quality: &QualityEvidenceConfig{Index: "vllm-sr/coding@1.0.0", OnMissing: "disable_quality"}},
		{name: "valid coverage and floor", quality: &QualityEvidenceConfig{Index: "vllm-sr/coding@1.0.0", OnMissing: "exclude", MinCoverage: 0.8, MinScore: &qualityFloor}},
		{name: "missing index", quality: &QualityEvidenceConfig{OnMissing: "exclude"}, wantErrSub: "index is required"},
		{name: "surrounding whitespace", quality: &QualityEvidenceConfig{Index: " vllm-sr/coding@1.0.0 "}, wantErrSub: "surrounding whitespace"},
		{name: "unknown policy", quality: &QualityEvidenceConfig{Index: "vllm-sr/coding@1.0.0", OnMissing: "impute"}, wantErrSub: "on_missing must be"},
		{name: "invalid coverage", quality: &QualityEvidenceConfig{Index: "vllm-sr/coding@1.0.0", MinCoverage: 1.1}, wantErrSub: "min_coverage"},
		{name: "non-finite floor", quality: &QualityEvidenceConfig{Index: "vllm-sr/coding@1.0.0", MinScore: &nan}, wantErrSub: "min_score must be finite"},
		{name: "floor cannot disable quality", quality: &QualityEvidenceConfig{Index: "vllm-sr/coding@1.0.0", OnMissing: "disable_quality", MinScore: &qualityFloor}, wantErrSub: "min_score requires"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateDecisionAlgorithmConfig("quality-route", nil, &AlgorithmConfig{
				Type: DecisionAlgorithmMultiFactor,
				MultiFactor: &MultiFactorSelectionConfig{
					Quality: test.quality,
				},
			})
			if test.wantErrSub == "" && err != nil {
				t.Fatalf("valid quality evidence config rejected: %v", err)
			}
			if test.wantErrSub != "" && (err == nil || !strings.Contains(err.Error(), test.wantErrSub)) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErrSub)
			}
		})
	}
}

func TestValidateDecisionMultiFactorObjective(t *testing.T) {
	tests := []struct {
		name       string
		config     *MultiFactorSelectionConfig
		wantErrSub string
	}{
		{
			name: "valid weighted default",
			config: &MultiFactorSelectionConfig{
				Weights: &MultiFactorWeightsConfig{Quality: 0.7, Cost: 0.3},
			},
		},
		{
			name: "valid accuracy first",
			config: &MultiFactorSelectionConfig{Objective: &MultiFactorObjectiveConfig{
				Strategy: MultiFactorObjectiveLexicographic,
				Priorities: []MultiFactorPriorityConfig{
					{Factor: MultiFactorFactorQuality, Tolerance: 0.02},
					{Factor: MultiFactorFactorCost},
				},
			}},
		},
		{
			name: "lexicographic cannot use weights",
			config: &MultiFactorSelectionConfig{
				Objective: &MultiFactorObjectiveConfig{
					Strategy:   MultiFactorObjectiveLexicographic,
					Priorities: []MultiFactorPriorityConfig{{Factor: MultiFactorFactorQuality}},
				},
				Weights: &MultiFactorWeightsConfig{Quality: 1},
			},
			wantErrSub: "weights cannot be combined",
		},
		{
			name: "duplicate priority",
			config: &MultiFactorSelectionConfig{Objective: &MultiFactorObjectiveConfig{
				Strategy: MultiFactorObjectiveLexicographic,
				Priorities: []MultiFactorPriorityConfig{
					{Factor: MultiFactorFactorCost},
					{Factor: MultiFactorFactorCost},
				},
			}},
			wantErrSub: "duplicate factor",
		},
		{
			name: "invalid priority factor",
			config: &MultiFactorSelectionConfig{Objective: &MultiFactorObjectiveConfig{
				Strategy:   MultiFactorObjectiveLexicographic,
				Priorities: []MultiFactorPriorityConfig{{Factor: "energy"}},
			}},
			wantErrSub: "unsupported",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateDecisionMultiFactorAlgorithm("objective-route", test.config)
			if test.wantErrSub == "" && err != nil {
				t.Fatalf("valid objective rejected: %v", err)
			}
			if test.wantErrSub != "" && (err == nil || !strings.Contains(err.Error(), test.wantErrSub)) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErrSub)
			}
		})
	}
}
