package dsl

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func floatPtr(v float64) *float64 {
	return &v
}

// A keyword rule's description, count and density survive decompile and
// compile, and a projection reads the rule's value as a keyword input.
func TestDecompileRoutingRoundTripsKeywordCountAndDensity(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				KeywordRules: []config.KeywordRule{
					{
						Name:        "many_questions",
						Description: "Prompts that contain many explicit questions.",
						Operator:    "OR",
						Method:      "regex",
						Keywords:    []string{`[?？]`},
						Count:       &config.NumericPredicate{GTE: floatPtr(4)},
					},
					{
						Name:     "constraint_dense",
						Operator: "OR",
						Keywords: []string{"at most", "至少"},
						Density:  &config.NumericPredicate{GT: floatPtr(16), LTE: floatPtr(400)},
					},
				},
			},
			Projections: config.Projections{
				Scores: []config.ProjectionScore{{
					Name:   "shape",
					Method: "weighted_sum",
					Inputs: []config.ProjectionScoreInput{{Type: config.SignalTypeKeyword, Name: "many_questions", Weight: 0.2, ValueSource: "raw"}},
				}},
			},
		},
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}
	for _, snippet := range []string{`description: "Prompts that contain many explicit questions."`, "count: {", "density: {"} {
		if !strings.Contains(dslText, snippet) {
			t.Fatalf("decompiled DSL is missing %s:\n%s", snippet, dslText)
		}
	}
	diags, errs := Validate(dslText)
	if len(errs) > 0 {
		t.Fatalf("Validate parse errors: %v", errs)
	}
	for _, diag := range diags {
		if diag.Level == DiagError || diag.Level == DiagConstraint {
			t.Fatalf("unexpected DSL diagnostic: %s", diag.String())
		}
	}
	roundTripped, compileErrs := Compile(dslText)
	if len(compileErrs) > 0 {
		t.Fatalf("Compile errors: %v", compileErrs)
	}
	if !reflect.DeepEqual(roundTripped.KeywordRules, cfg.KeywordRules) {
		t.Fatalf("round-trip keyword rules mismatch\nwant: %#v\ngot:  %#v", cfg.KeywordRules, roundTripped.KeywordRules)
	}
	if got := roundTripped.Projections.Scores[0].Inputs[0]; got.Type != config.SignalTypeKeyword || got.ValueSource != "raw" {
		t.Fatalf("round-trip projection input = %+v", got)
	}
}

// Validate and Compile refuse a count that the rule cannot locate, and the
// removed structure and event signal types.
func TestKeywordCountContractAndRemovedHeuristicTypes(t *testing.T) {
	counted := `
SIGNAL keyword all_markers {
  operator: "AND"
  keywords: ["first", "then"]
  count: { gte: 2 }
}
`
	diags, errs := Validate(counted)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	refused := false
	for _, diag := range diags {
		refused = refused || strings.Contains(diag.Message, `operator "AND"`)
	}
	if !refused {
		t.Fatalf("Validate accepted a count on an AND rule: %+v", diags)
	}
	if _, errs := Compile(counted); len(errs) == 0 {
		t.Fatal("Compile accepted a count on an AND rule")
	}
	for _, removed := range []string{"structure", "event"} {
		_, errs := Compile("SIGNAL " + removed + " old { description: \"removed\" }\n")
		if len(errs) == 0 || !strings.Contains(errs[0].Error(), `unknown signal type "`+removed+`"`) {
			t.Errorf("SIGNAL %s: %v", removed, errs)
		}
	}
}
