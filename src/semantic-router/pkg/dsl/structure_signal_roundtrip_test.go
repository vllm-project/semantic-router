package dsl

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestValidateProjectionScoreAcceptsStructureSignals(t *testing.T) {
	input := `
SIGNAL structure ordered_workflow {
  feature: {
    type: "sequence",
    source: {
      type: "sequence",
      sequences: [["first", "then"], ["先", "再"]]
    }
  }
}

PROJECTION score difficulty_score {
  method: "weighted_sum"
  inputs: [
    { type: "structure", name: "ordered_workflow", weight: 0.2 }
  ]
}
`

	diags, errs := Validate(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	for _, diag := range diags {
		if strings.Contains(diag.Message, `unsupported type "structure"`) {
			t.Fatalf("unexpected structure projection validation failure: %s", diag.Message)
		}
	}
}

func TestDecompileRoutingPreservesStructureSequenceLists(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				StructureRules: []config.StructureRule{
					{
						Name:        "ordered_workflow",
						Description: "Prompts that express an ordered workflow.",
						Feature: config.StructureFeature{
							Type: "sequence",
							Source: config.StructureSource{
								Type: "sequence",
								Sequences: [][]string{
									{"first", "then"},
									{"first", "next", "finally"},
									{"先", "再"},
									{"首先", "然后"},
								},
							},
						},
					},
				},
			},
			Projections: config.Projections{
				Scores: []config.ProjectionScore{
					{
						Name:   "difficulty_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{
							{Type: config.SignalTypeStructure, Name: "ordered_workflow", Weight: 0.2},
						},
					},
				},
			},
		},
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}
	assertDecompiledStructureSnippets(t, dslText)
	assertValidStructureDSL(t, dslText)
	assertStructureRoundTrip(t, cfg, dslText)
}

func TestDecompileRoutingOmitsRemovedStructureNormalizer(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				StructureRules: []config.StructureRule{
					{
						Name: "format_directive_dense",
						Feature: config.StructureFeature{
							Type: "density",
							Source: config.StructureSource{
								Type:     "keyword_set",
								Keywords: []string{"table", "表格", "json"},
							},
						},
						Predicate: &config.NumericPredicate{GT: floatPtr(0.08)},
					},
				},
			},
		},
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}
	if strings.Contains(dslText, "normalize_by") {
		t.Fatalf("decompiled DSL should not contain normalize_by:\n%s", dslText)
	}

	roundTripped, compileErrs := Compile(dslText)
	if len(compileErrs) > 0 {
		t.Fatalf("Compile errors: %v", compileErrs)
	}
	if !reflect.DeepEqual(roundTripped.StructureRules, cfg.StructureRules) {
		t.Fatalf("round-trip structure rules mismatch\nwant: %#v\ngot:  %#v", cfg.StructureRules, roundTripped.StructureRules)
	}
}

func floatPtr(v float64) *float64 {
	return &v
}

func TestValidateRejectsRemovedStructureGapFeature(t *testing.T) {
	input := `
SIGNAL structure tight_first_then_gap {
  feature: {
    type: "span_distance",
    source: {
      type: "marker_pair",
      start: ["first"],
      end: ["then"]
    }
  }
  predicate: { lte: 6 }
}
`

	diags, errs := Validate(input)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	for _, diag := range diags {
		if strings.Contains(diag.Message, `unsupported feature.type "span_distance"`) {
			return
		}
	}
	t.Fatalf("expected Validate to reject removed structure gap feature, got diagnostics: %+v", diags)
}

func TestCompileRejectsRemovedStructureGapSource(t *testing.T) {
	input := `
SIGNAL structure tight_first_then_gap {
  feature: {
    type: "count",
    source: {
      type: "marker_pair",
      start: ["first"],
      end: ["then"]
    }
  }
}
`

	_, errs := Compile(input)
	if len(errs) == 0 {
		t.Fatal("expected Compile to reject removed structure gap source")
	}
	for _, err := range errs {
		if strings.Contains(err.Error(), `unsupported feature.source.type "marker_pair"`) {
			return
		}
	}
	t.Fatalf("expected marker_pair rejection, got errors: %v", errs)
}

func assertDecompiledStructureSnippets(t *testing.T, dslText string) {
	t.Helper()

	for _, snippet := range []string{
		`["first", "then"]`,
		`["first", "next", "finally"]`,
		`["先", "再"]`,
		`["首先", "然后"]`,
	} {
		if !strings.Contains(dslText, snippet) {
			t.Fatalf("decompiled DSL missing quoted structure sequence %s:\n%s", snippet, dslText)
		}
	}
}

func assertValidStructureDSL(t *testing.T, dslText string) {
	t.Helper()

	diags, errs := Validate(dslText)
	if len(errs) > 0 {
		t.Fatalf("Validate parse errors: %v", errs)
	}
	for _, diag := range diags {
		if diag.Level == DiagError || diag.Level == DiagConstraint {
			t.Fatalf("unexpected DSL diagnostic after structure decompile: %s", diag.String())
		}
	}
}

func assertStructureRoundTrip(t *testing.T, cfg *config.RouterConfig, dslText string) {
	t.Helper()

	roundTripped, compileErrs := Compile(dslText)
	if len(compileErrs) > 0 {
		t.Fatalf("Compile errors: %v", compileErrs)
	}
	if len(roundTripped.StructureRules) != 1 {
		t.Fatalf("expected 1 round-tripped structure rule, got %d", len(roundTripped.StructureRules))
	}
	if !reflect.DeepEqual(
		roundTripped.StructureRules[0].Feature.Source.Sequences,
		cfg.StructureRules[0].Feature.Source.Sequences,
	) {
		t.Fatalf(
			"round-trip structure sequences mismatch\nwant: %#v\ngot:  %#v",
			cfg.StructureRules[0].Feature.Source.Sequences,
			roundTripped.StructureRules[0].Feature.Source.Sequences,
		)
	}
	if len(roundTripped.Projections.Scores) != 1 || len(roundTripped.Projections.Scores[0].Inputs) != 1 {
		t.Fatalf("expected 1 round-tripped projection score input, got %+v", roundTripped.Projections.Scores)
	}
	if got := roundTripped.Projections.Scores[0].Inputs[0].Type; got != config.SignalTypeStructure {
		t.Fatalf("round-trip projection input type = %q, want %q", got, config.SignalTypeStructure)
	}
}
