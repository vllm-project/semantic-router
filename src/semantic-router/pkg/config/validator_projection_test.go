package config

import (
	"strings"
	"testing"
)

func TestParseRoutingYAMLBytesRejectsUnknownProjectionDecisionReference(t *testing.T) {
	yaml := []byte(`
routing:
  signals:
    keywords:
      - name: reasoning_markers
        operator: OR
        keywords: ["reason carefully"]
  projections:
    scores:
      - name: difficulty_score
        method: weighted_sum
        inputs:
          - type: keyword
            name: reasoning_markers
            weight: 0.6
    mappings:
      - name: difficulty_band
        source: difficulty_score
        method: threshold_bands
        outputs:
          - name: balance_medium
            lt: 0.7
          - name: balance_reasoning
            gte: 0.7
  decisions:
    - name: bad_route
      rules:
        operator: AND
        conditions:
          - type: projection
            name: missing_output
      modelRefs:
        - model: qwen3-8b
`)

	_, err := ParseRoutingYAMLBytes(yaml)
	if err == nil {
		t.Fatal("expected error for unknown projection output reference")
	}
	if !strings.Contains(err.Error(), `references projection "missing_output"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestParseRoutingYAMLBytesRejectsUnknownProjectionScoreInput(t *testing.T) {
	yaml := []byte(`
routing:
  signals:
    keywords:
      - name: reasoning_markers
        operator: OR
        keywords: ["reason carefully"]
  projections:
    scores:
      - name: difficulty_score
        method: weighted_sum
        inputs:
          - type: keyword
            name: missing_signal
            weight: 0.6
    mappings:
      - name: difficulty_band
        source: difficulty_score
        method: threshold_bands
        outputs:
          - name: balance_medium
            lt: 0.7
          - name: balance_reasoning
            gte: 0.7
  decisions:
    - name: reasoning_route
      rules:
        operator: AND
        conditions:
          - type: projection
            name: balance_reasoning
      modelRefs:
        - model: qwen3-8b
`)

	_, err := ParseRoutingYAMLBytes(yaml)
	if err == nil {
		t.Fatal("expected error for unknown projection score input")
	}
	if !strings.Contains(err.Error(), `input keyword("missing_signal") is not declared`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateProjectionMappingAcceptsNewMethods(t *testing.T) {
	tests := []struct {
		name   string
		method string
		valid  bool
	}{
		{"threshold_bands", "threshold_bands", true},
		{"multi_emit", "multi_emit", true},
		{"top_k", "top_k", true},
		{"hysteresis", "hysteresis", true},
		{"unknown", "unknown_method", false},
	}

	scoreNames := map[string]struct{}{"test_score": {}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputNames := make(map[string]struct{}) // Fresh map for each subtest

			mapping := ProjectionMapping{
				Name:   "test_mapping",
				Source: "test_score",
				Method: tt.method,
				TopK:   2,
				Hysteresis: &HysteresisConfig{
					UpThreshold:   0.7,
					DownThreshold: 0.3,
				},
				Outputs: []ProjectionMappingOutput{
					{Name: "output1", LT: float64Ptr(0.5)},
					{Name: "output2", GTE: float64Ptr(0.5)},
				},
			}

			err := validateProjectionMapping(mapping, scoreNames, outputNames)
			if tt.valid && err != nil {
				t.Errorf("expected valid, got error: %v", err)
			}
			if !tt.valid && err == nil {
				t.Errorf("expected error for invalid method")
			}
		})
	}
}

func TestValidateProjectionMappingTopKRequiresTopKField(t *testing.T) {
	scoreNames := map[string]struct{}{"test_score": {}}
	outputNames := make(map[string]struct{})

	mapping := ProjectionMapping{
		Name:   "test_mapping",
		Source: "test_score",
		Method: "top_k",
		TopK:   0, // Invalid: should be > 0
		Outputs: []ProjectionMappingOutput{
			{Name: "output1", LT: float64Ptr(0.5)},
		},
	}

	err := validateProjectionMapping(mapping, scoreNames, outputNames)
	if err == nil {
		t.Fatal("expected error for top_k without valid TopK field")
	}
	if !strings.Contains(err.Error(), "top_k > 0") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateProjectionMappingTopKExceedsOutputs(t *testing.T) {
	scoreNames := map[string]struct{}{"test_score": {}}
	outputNames := make(map[string]struct{})

	mapping := ProjectionMapping{
		Name:   "test_mapping",
		Source: "test_score",
		Method: "top_k",
		TopK:   5, // More than outputs
		Outputs: []ProjectionMappingOutput{
			{Name: "output1", LT: float64Ptr(0.5)},
			{Name: "output2", GTE: float64Ptr(0.5)},
		},
	}

	err := validateProjectionMapping(mapping, scoreNames, outputNames)
	if err == nil {
		t.Fatal("expected error for TopK exceeding outputs")
	}
	if !strings.Contains(err.Error(), "cannot exceed number of outputs") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateProjectionMappingHysteresisRequiresConfig(t *testing.T) {
	scoreNames := map[string]struct{}{"test_score": {}}
	outputNames := make(map[string]struct{})

	mapping := ProjectionMapping{
		Name:       "test_mapping",
		Source:     "test_score",
		Method:     "hysteresis",
		Hysteresis: nil, // Invalid: should be non-nil
		Outputs: []ProjectionMappingOutput{
			{Name: "output1", LT: float64Ptr(0.5)},
		},
	}

	err := validateProjectionMapping(mapping, scoreNames, outputNames)
	if err == nil {
		t.Fatal("expected error for hysteresis without config")
	}
	if !strings.Contains(err.Error(), "requires hysteresis config") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateProjectionMappingHysteresisInvalidThresholds(t *testing.T) {
	scoreNames := map[string]struct{}{"test_score": {}}
	outputNames := make(map[string]struct{})

	mapping := ProjectionMapping{
		Name:   "test_mapping",
		Source: "test_score",
		Method: "hysteresis",
		Hysteresis: &HysteresisConfig{
			UpThreshold:   0.3, // Invalid: should be > DownThreshold
			DownThreshold: 0.7,
		},
		Outputs: []ProjectionMappingOutput{
			{Name: "output1", LT: float64Ptr(0.5)},
		},
	}

	err := validateProjectionMapping(mapping, scoreNames, outputNames)
	if err == nil {
		t.Fatal("expected error for invalid hysteresis thresholds")
	}
	if !strings.Contains(err.Error(), "must be less than up_threshold") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func float64Ptr(v float64) *float64 {
	return &v
}
