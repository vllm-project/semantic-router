package config

import (
	"fmt"
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
		{"unknown", "unknown_method", false},
	}

	scoreNames := map[string]struct{}{"test_score": {}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputNames := make(map[string]struct{})

			mapping := ProjectionMapping{
				Name:   "test_mapping",
				Source: "test_score",
				Method: tt.method,
				TopK:   2,
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
		TopK:   0,
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
		TopK:   5,
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

func TestValidateProjectionInputValueSourceAcceptsAllModes(t *testing.T) {
	for _, vs := range []string{"", "binary", "confidence", "raw"} {
		t.Run(fmt.Sprintf("value_source=%q", vs), func(t *testing.T) {
			err := validateProjectionInputValueSource("test_score", ProjectionScoreInput{
				Type:        "keyword",
				Name:        "sig",
				Weight:      1.0,
				ValueSource: vs,
			})
			if err != nil {
				t.Fatalf("expected no error for value_source %q, got: %v", vs, err)
			}
		})
	}
}

func TestValidateProjectionInputValueSourceRejectsUnknown(t *testing.T) {
	err := validateProjectionInputValueSource("test_score", ProjectionScoreInput{
		Type:        "keyword",
		Name:        "sig",
		Weight:      1.0,
		ValueSource: "unknown_mode",
	})
	if err == nil {
		t.Fatal("expected error for unsupported value_source")
	}
	if !strings.Contains(err.Error(), "unsupported value_source") {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(err.Error(), "raw") {
		t.Fatalf("error message should list 'raw' as a supported mode: %v", err)
	}
}

func TestParseRoutingYAMLBytesAcceptsRawValueSource(t *testing.T) {
	yaml := []byte(`
routing:
  signals:
    structure:
      - name: many_questions
        operator: OR
        patterns: ["\\?"]
  projections:
    scores:
      - name: workload_pressure
        method: weighted_sum
        inputs:
          - type: structure
            name: many_questions
            weight: 0.5
            value_source: raw
    mappings:
      - name: pressure_band
        source: workload_pressure
        method: threshold_bands
        outputs:
          - name: high_pressure
            gte: 2.0
  decisions:
    - name: heavy_route
      rules:
        operator: AND
        conditions:
          - type: projection
            name: high_pressure
      modelRefs:
        - model: qwen3-8b
`)
	cfg, err := ParseRoutingYAMLBytes(yaml)
	if err != nil {
		t.Fatalf("expected valid config with value_source: raw, got: %v", err)
	}
	if len(cfg.Projections.Scores) != 1 {
		t.Fatalf("expected 1 projection score, got %d", len(cfg.Projections.Scores))
	}
	if cfg.Projections.Scores[0].Inputs[0].ValueSource != "raw" {
		t.Fatalf("value_source = %q, want %q", cfg.Projections.Scores[0].Inputs[0].ValueSource, "raw")
	}
}

func float64Ptr(v float64) *float64 {
	return &v
}
