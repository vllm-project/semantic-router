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

func TestValidateProjectionScoreDependencyOrder_ValidChain(t *testing.T) {
	scores := []ProjectionScore{
		{
			Name:   "base_score",
			Method: "weighted_sum",
			Inputs: []ProjectionScoreInput{{Type: "keyword", Name: "k1", Weight: 1.0}},
		},
		{
			Name:   "derived_score",
			Method: "weighted_sum",
			Inputs: []ProjectionScoreInput{
				{Type: SignalTypeProjection, Name: "base_score", Weight: 0.8, ValueSource: "score"},
			},
		},
	}
	if err := validateProjectionScoreDependencyOrder(scores, nil); err != nil {
		t.Fatalf("expected no error for valid chain, got: %v", err)
	}
}

func TestValidateProjectionScoreDependencyOrder_RejectsCycle(t *testing.T) {
	scores := []ProjectionScore{
		{
			Name:   "alpha",
			Method: "weighted_sum",
			Inputs: []ProjectionScoreInput{
				{Type: SignalTypeProjection, Name: "beta", Weight: 1.0},
			},
		},
		{
			Name:   "beta",
			Method: "weighted_sum",
			Inputs: []ProjectionScoreInput{
				{Type: SignalTypeProjection, Name: "alpha", Weight: 1.0},
			},
		},
	}
	err := validateProjectionScoreDependencyOrder(scores, nil)
	if err == nil {
		t.Fatal("expected cycle detection error")
	}
	if !strings.Contains(err.Error(), "dependency cycle") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateProjectionScoreDependencyOrder_RejectsSelfRef(t *testing.T) {
	scores := []ProjectionScore{
		{
			Name:   "self_loop",
			Method: "weighted_sum",
			Inputs: []ProjectionScoreInput{
				{Type: SignalTypeProjection, Name: "self_loop", Weight: 1.0},
			},
		},
	}
	err := validateProjectionScoreDependencyOrder(scores, nil)
	if err == nil {
		t.Fatal("expected self-reference cycle error")
	}
	if !strings.Contains(err.Error(), "dependency cycle") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateProjectionScoreDependencyOrder_RejectsUndefined(t *testing.T) {
	scores := []ProjectionScore{
		{
			Name:   "uses_missing",
			Method: "weighted_sum",
			Inputs: []ProjectionScoreInput{
				{Type: SignalTypeProjection, Name: "nonexistent", Weight: 1.0},
			},
		},
	}
	err := validateProjectionScoreDependencyOrder(scores, nil)
	if err == nil {
		t.Fatal("expected undefined reference error")
	}
	if !strings.Contains(err.Error(), "undefined score") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateProjectionScoreDependencyOrder_ThreeNodeCycle(t *testing.T) {
	scores := []ProjectionScore{
		{Name: "a", Method: "weighted_sum", Inputs: []ProjectionScoreInput{
			{Type: SignalTypeProjection, Name: "b", Weight: 1.0},
		}},
		{Name: "b", Method: "weighted_sum", Inputs: []ProjectionScoreInput{
			{Type: SignalTypeProjection, Name: "c", Weight: 1.0},
		}},
		{Name: "c", Method: "weighted_sum", Inputs: []ProjectionScoreInput{
			{Type: SignalTypeProjection, Name: "a", Weight: 1.0},
		}},
	}
	err := validateProjectionScoreDependencyOrder(scores, nil)
	if err == nil {
		t.Fatal("expected cycle detection error for 3-node cycle")
	}
	if !strings.Contains(err.Error(), "dependency cycle") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateProjectionInputProjectionRef_ValidScoreSource(t *testing.T) {
	err := validateProjectionInputProjectionRef("test_score", ProjectionScoreInput{
		Type: SignalTypeProjection, Name: "other_score", Weight: 0.5, ValueSource: "score",
	}, nil)
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
}

func TestValidateProjectionInputProjectionRef_ValidConfidenceSource(t *testing.T) {
	outputToSource := map[string]string{"other_output": "some_score"}
	err := validateProjectionInputProjectionRef("test_score", ProjectionScoreInput{
		Type: SignalTypeProjection, Name: "other_output", Weight: 0.5, ValueSource: "confidence",
	}, outputToSource)
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
}

func TestValidateProjectionInputProjectionRef_ConfidenceRejectsUndefinedOutput(t *testing.T) {
	outputToSource := map[string]string{"known_output": "some_score"}
	err := validateProjectionInputProjectionRef("test_score", ProjectionScoreInput{
		Type: SignalTypeProjection, Name: "unknown_output", Weight: 0.5, ValueSource: "confidence",
	}, outputToSource)
	if err == nil {
		t.Fatal("expected error for undefined mapping output")
	}
	if !strings.Contains(err.Error(), "undefined mapping output") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateProjectionInputProjectionRef_EmptyName(t *testing.T) {
	err := validateProjectionInputProjectionRef("test_score", ProjectionScoreInput{
		Type: SignalTypeProjection, Name: "", Weight: 0.5,
	}, nil)
	if err == nil {
		t.Fatal("expected error for empty projection input name")
	}
}

func TestValidateProjectionInputProjectionRef_InvalidValueSource(t *testing.T) {
	err := validateProjectionInputProjectionRef("test_score", ProjectionScoreInput{
		Type: SignalTypeProjection, Name: "other", Weight: 0.5, ValueSource: "invalid",
	}, nil)
	if err == nil {
		t.Fatal("expected error for invalid value_source")
	}
	if !strings.Contains(err.Error(), "unsupported value_source") {
		t.Fatalf("unexpected error: %v", err)
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

func TestIsProjectionInputTypeSupported_ConversationAndProjection(t *testing.T) {
	for _, typ := range []string{SignalTypeConversation, SignalTypeProjection} {
		if !isProjectionInputTypeSupported(typ) {
			t.Fatalf("expected supported input type %q", typ)
		}
	}
}

func TestParseRoutingYAMLBytesAcceptsProjectionToProjectionRef(t *testing.T) {
	yaml := []byte(fmt.Sprintf(`
routing:
  signals:
    keywords:
      - name: reasoning_markers
        operator: OR
        keywords: ["reason carefully"]
  projections:
    scores:
      - name: base_score
        method: weighted_sum
        inputs:
          - type: keyword
            name: reasoning_markers
            weight: 0.6
      - name: derived_score
        method: weighted_sum
        inputs:
          - type: %s
            name: base_score
            value_source: score
            weight: 0.8
    mappings:
      - name: derived_band
        source: derived_score
        method: threshold_bands
        outputs:
          - name: high_derived
            gte: 0.5
  decisions:
    - name: test_route
      rules:
        operator: AND
        conditions:
          - type: projection
            name: high_derived
      modelRefs:
        - model: test-model
`, SignalTypeProjection))

	cfg, err := ParseRoutingYAMLBytes(yaml)
	if err != nil {
		t.Fatalf("expected valid config, got error: %v", err)
	}
	if len(cfg.Projections.Scores) != 2 {
		t.Fatalf("expected 2 scores, got %d", len(cfg.Projections.Scores))
	}
}

func TestParseRoutingYAMLBytesAcceptsConfidenceProjectionRef(t *testing.T) {
	yaml := []byte(`
routing:
  signals:
    keywords:
      - name: reasoning_markers
        operator: OR
        keywords: ["reason carefully"]
  projections:
    scores:
      - name: base_score
        method: weighted_sum
        inputs:
          - type: keyword
            name: reasoning_markers
            weight: 0.6
      - name: composite_score
        method: weighted_sum
        inputs:
          - type: projection
            name: high_base
            value_source: confidence
            weight: 0.7
    mappings:
      - name: base_band
        source: base_score
        method: threshold_bands
        outputs:
          - name: high_base
            gte: 0.5
      - name: composite_band
        source: composite_score
        method: threshold_bands
        outputs:
          - name: high_composite
            gte: 0.3
  decisions:
    - name: test_route
      rules:
        operator: AND
        conditions:
          - type: projection
            name: high_composite
      modelRefs:
        - model: test-model
`)

	cfg, err := ParseRoutingYAMLBytes(yaml)
	if err != nil {
		t.Fatalf("expected valid config with confidence projection ref, got error: %v", err)
	}
	if len(cfg.Projections.Scores) != 2 {
		t.Fatalf("expected 2 scores, got %d", len(cfg.Projections.Scores))
	}
}

func TestParseRoutingYAMLBytesRejectsUndefinedConfidenceOutput(t *testing.T) {
	yaml := []byte(`
routing:
  signals:
    keywords:
      - name: reasoning_markers
        operator: OR
        keywords: ["reason carefully"]
  projections:
    scores:
      - name: base_score
        method: weighted_sum
        inputs:
          - type: keyword
            name: reasoning_markers
            weight: 0.6
      - name: composite_score
        method: weighted_sum
        inputs:
          - type: projection
            name: nonexistent_output
            value_source: confidence
            weight: 0.7
    mappings:
      - name: base_band
        source: base_score
        method: threshold_bands
        outputs:
          - name: high_base
            gte: 0.5
  decisions:
    - name: test_route
      rules:
        operator: AND
        conditions:
          - type: projection
            name: high_base
      modelRefs:
        - model: test-model
`)

	_, err := ParseRoutingYAMLBytes(yaml)
	if err == nil {
		t.Fatal("expected error for undefined confidence output reference")
	}
	if !strings.Contains(err.Error(), "undefined mapping output") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateProjectionScoreDependencyOrder_ConfidenceResolvesToSourceScore(t *testing.T) {
	scores := []ProjectionScore{
		{
			Name:   "base_score",
			Method: "weighted_sum",
			Inputs: []ProjectionScoreInput{{Type: "keyword", Name: "k1", Weight: 1.0}},
		},
		{
			Name:   "derived_score",
			Method: "weighted_sum",
			Inputs: []ProjectionScoreInput{
				{Type: SignalTypeProjection, Name: "high_band", Weight: 0.8, ValueSource: "confidence"},
			},
		},
	}
	mappings := []ProjectionMapping{
		{
			Name:   "base_mapping",
			Source: "base_score",
			Method: "threshold_bands",
			Outputs: []ProjectionMappingOutput{
				{Name: "high_band", GTE: float64Ptr(0.5)},
			},
		},
	}
	if err := validateProjectionScoreDependencyOrder(scores, mappings); err != nil {
		t.Fatalf("expected no error for confidence ref resolving to valid source score, got: %v", err)
	}
}

func float64Ptr(v float64) *float64 { return &v }
