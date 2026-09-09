package config

import (
	"strings"
	"testing"
)

func TestValidateDecisionMultiFactorQualityEvidence(t *testing.T) {
	tests := []struct {
		name       string
		quality    *QualityEvidenceConfig
		wantErrSub string
	}{
		{name: "valid strict", quality: &QualityEvidenceConfig{Index: "vllm-sr/coding@1.0.0", OnMissing: "exclude"}},
		{name: "valid inclusive", quality: &QualityEvidenceConfig{Index: "vllm-sr/coding@1.0.0", OnMissing: "disable_quality"}},
		{name: "missing index", quality: &QualityEvidenceConfig{OnMissing: "exclude"}, wantErrSub: "index is required"},
		{name: "surrounding whitespace", quality: &QualityEvidenceConfig{Index: " vllm-sr/coding@1.0.0 "}, wantErrSub: "surrounding whitespace"},
		{name: "unknown policy", quality: &QualityEvidenceConfig{Index: "vllm-sr/coding@1.0.0", OnMissing: "impute"}, wantErrSub: "on_missing must be"},
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
