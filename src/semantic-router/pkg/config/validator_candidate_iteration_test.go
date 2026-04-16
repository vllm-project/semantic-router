package config

import (
	"strings"
	"testing"
)

// minimalDecision returns a Decision with one ModelRef pre-filled so tests
// only need to set the fields under test.
func minimalDecision(name string) Decision {
	return Decision{
		Name:      name,
		ModelRefs: []ModelRef{{Model: "base-model"}},
	}
}

func TestValidateCandidateIterationAcceptsDecisionCandidatesSource(t *testing.T) {
	d := minimalDecision("route")
	d.CandidateIterations = []CandidateIterationConfig{
		{
			Variable: "candidate",
			Source:   "decision.candidates",
			Outputs:  []CandidateIterationOutputConfig{{Type: "model", Value: "candidate"}},
		},
	}
	if err := validateDecisionCandidateIterations(d); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateCandidateIterationAcceptsExplicitModelSource(t *testing.T) {
	d := minimalDecision("route")
	d.CandidateIterations = []CandidateIterationConfig{
		{
			Variable: "c",
			Source:   "models",
			Models:   []ModelRef{{Model: "small-model"}, {Model: "large-model"}},
			Outputs:  []CandidateIterationOutputConfig{{Type: "model", Value: "c"}},
		},
	}
	if err := validateDecisionCandidateIterations(d); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateCandidateIterationRejectsEmptyVariable(t *testing.T) {
	d := minimalDecision("route")
	d.CandidateIterations = []CandidateIterationConfig{
		{Variable: "", Source: "decision.candidates"},
	}
	err := validateDecisionCandidateIterations(d)
	if err == nil {
		t.Fatal("expected error for empty variable, got nil")
	}
	if !strings.Contains(err.Error(), "variable cannot be empty") {
		t.Fatalf("error message = %q, want to contain 'variable cannot be empty'", err.Error())
	}
}

func TestValidateCandidateIterationRejectsUnsupportedSource(t *testing.T) {
	d := minimalDecision("route")
	d.CandidateIterations = []CandidateIterationConfig{
		{Variable: "c", Source: "runtime.models"},
	}
	err := validateDecisionCandidateIterations(d)
	if err == nil {
		t.Fatal("expected error for unsupported source, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported source") {
		t.Fatalf("error message = %q, want to contain 'unsupported source'", err.Error())
	}
}

func TestValidateCandidateIterationRejectsDecisionCandidatesWithNoModelRefs(t *testing.T) {
	d := Decision{
		Name: "route",
		CandidateIterations: []CandidateIterationConfig{
			{Variable: "c", Source: "decision.candidates"},
		},
	}
	err := validateDecisionCandidateIterations(d)
	if err == nil {
		t.Fatal("expected error when decision.candidates used without modelRefs, got nil")
	}
	if !strings.Contains(err.Error(), "non-empty modelRefs") {
		t.Fatalf("error message = %q, want to contain 'non-empty modelRefs'", err.Error())
	}
}

func TestValidateCandidateIterationRejectsEmptyModelList(t *testing.T) {
	d := minimalDecision("route")
	d.CandidateIterations = []CandidateIterationConfig{
		{Variable: "c", Source: "models", Models: nil},
	}
	err := validateDecisionCandidateIterations(d)
	if err == nil {
		t.Fatal("expected error for empty models list, got nil")
	}
	if !strings.Contains(err.Error(), "at least one model") {
		t.Fatalf("error message = %q, want to contain 'at least one model'", err.Error())
	}
}

func TestValidateCandidateIterationRejectsBlankModelName(t *testing.T) {
	d := minimalDecision("route")
	d.CandidateIterations = []CandidateIterationConfig{
		{Variable: "c", Source: "models", Models: []ModelRef{{Model: ""}}},
	}
	err := validateDecisionCandidateIterations(d)
	if err == nil {
		t.Fatal("expected error for blank model name, got nil")
	}
	if !strings.Contains(err.Error(), "model name cannot be empty") {
		t.Fatalf("error message = %q, want to contain 'model name cannot be empty'", err.Error())
	}
}

func TestValidateCandidateIterationRejectsUnsupportedOutputType(t *testing.T) {
	d := minimalDecision("route")
	d.CandidateIterations = []CandidateIterationConfig{
		{
			Variable: "c",
			Source:   "decision.candidates",
			Outputs:  []CandidateIterationOutputConfig{{Type: "policy", Value: "c"}},
		},
	}
	err := validateDecisionCandidateIterations(d)
	if err == nil {
		t.Fatal("expected error for unsupported output type, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported output type") {
		t.Fatalf("error message = %q, want to contain 'unsupported output type'", err.Error())
	}
}

func TestValidateCandidateIterationRejectsOutputNotReferencingVariable(t *testing.T) {
	d := minimalDecision("route")
	d.CandidateIterations = []CandidateIterationConfig{
		{
			Variable: "c",
			Source:   "decision.candidates",
			Outputs:  []CandidateIterationOutputConfig{{Type: "model", Value: "other"}},
		},
	}
	err := validateDecisionCandidateIterations(d)
	if err == nil {
		t.Fatal("expected error when output value does not match iterator variable, got nil")
	}
	if !strings.Contains(err.Error(), "reference variable") {
		t.Fatalf("error message = %q, want to contain 'reference variable'", err.Error())
	}
}
