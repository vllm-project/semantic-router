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
