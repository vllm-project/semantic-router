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

func TestParseRoutingYAMLBytesAcceptsSessionProjectionScoreInput(t *testing.T) {
	yaml := []byte(`
routing:
  signals:
    session:
      - name: session_present
        fact: session_present
        predicate:
          gte: 1
  projections:
    scores:
      - name: continuity_score
        method: weighted_sum
        inputs:
          - type: session
            name: session_present
            weight: 0.6
    mappings:
      - name: continuity_band
        source: continuity_score
        method: threshold_bands
        outputs:
          - name: continue_session
            gte: 0.5
  decisions:
    - name: session_route
      rules:
        operator: AND
        conditions:
          - type: projection
            name: continue_session
      modelRefs:
        - model: qwen3-8b
`)

	_, err := ParseRoutingYAMLBytes(yaml)
	if err != nil {
		fatalf := t.Fatalf
		fatalf("expected session projection input to validate, got: %v", err)
	}
}
