package dsl

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Every systemone rule field survives decompiling to the DSL and compiling
// it again, and the DSL writes only the systemone signal and algorithm.
func TestSystemOneSignalAndAlgorithmRoundTrip(t *testing.T) {
	threshold := 0.7
	cfg := &config.RouterConfig{}
	cfg.SystemOneRules = []config.SystemOneRule{
		{
			Name: "needs_reasoning", Description: "Needs reasoning", Deployment: "decision-kai",
			Question:  config.SystemOneQuestion{Type: "noul", Instructions: "Does this need reasoning?"},
			Predicate: &config.NumericPredicate{GTE: &threshold}, TimeoutMs: 800,
			Input: config.SystemOneInput{History: true, PreviousTurns: 4, ToolResults: true, Image: true},
		},
		{
			Name: "request_kind", Deployment: "decision-kai", Labels: []string{"code"},
			Question: config.SystemOneQuestion{Type: "choice", Instructions: "Which kind?", Choices: []config.SystemOneOption{
				{Key: "code", Description: "Code"}, {Key: "math"},
			}},
		},
		{
			Name: "difficulty", Deployment: "decision-kai", Predicate: &config.NumericPredicate{GTE: &threshold},
			Question: config.SystemOneQuestion{Type: "score", Instructions: "How hard?", Levels: []string{"easy", "medium", "hard"}},
		},
		{
			Name: "topics", Deployment: "vela2", Predicate: &config.NumericPredicate{GTE: &threshold},
			Labels: []string{"billing", "shipping"}, IgnoreLabels: []string{"other"},
			Question: config.SystemOneQuestion{Type: "set", Instructions: "Which topics?", Threshold: &threshold, Labels: []config.SystemOneOption{
				{Key: "billing", Description: "Payments"}, {Key: "shipping"}, {Key: "other"},
			}},
		},
		{
			Name: "places", Deployment: "vela2", Coverage: "full", OnUnscanned: "match",
			Question: config.SystemOneQuestion{Type: "span", Instructions: "Which spans name a place?", Head: "broad", Labels: []config.SystemOneOption{{Key: "city"}}},
		},
		{
			Name: "unsafe_answer", Stage: "response", Input: config.SystemOneInput{Answer: true, Context: true},
			Question: config.SystemOneQuestion{Type: "noul", Instructions: "Is the answer unsafe?"},
		},
	}
	cfg.Decisions = []config.Decision{{
		Name:      "systemone-route",
		Priority:  10,
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmSystemOne, SystemOne: &config.SystemOneSelectionConfig{
			Deployment: "decision-kai", Instructions: "Which model?", TimeoutMs: 900,
			Candidates: map[string]string{"model-a": "Fast", "model-b": "Strong"},
		}},
	}}
	source, err := Decompile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(source, "SIGNAL decision") || strings.Contains(source, "ALGORITHM decision") || !strings.Contains(source, "SIGNAL systemone") || !strings.Contains(source, "ALGORITHM systemone") {
		t.Fatalf("the DSL writes the systemone signal and algorithm:\n%s", source)
	}
	compiled, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v\n%s", errs, source)
	}
	if !reflect.DeepEqual(compiled.SystemOneRules, cfg.SystemOneRules) {
		t.Fatalf("systemone signals changed in the round trip:\n got %#v\nwant %#v\n%s", compiled.SystemOneRules, cfg.SystemOneRules, source)
	}
	if !reflect.DeepEqual(compiled.Decisions[0].Algorithm.SystemOne, cfg.Decisions[0].Algorithm.SystemOne) {
		t.Fatalf("systemone algorithm changed in the round trip: %#v", compiled.Decisions[0].Algorithm.SystemOne)
	}
}

// A rule field the contract refuses fails the DSL compile with the load's
// message.
func TestSystemOneSignalCompileRefusesFieldErrors(t *testing.T) {
	source := `SIGNAL systemone kind {
  question: {type: "choice", instructions: "Which kind?", choices: [{key: "code"}, {key: "math"}]}
  ignore_labels: ["code"]
}
`
	if _, errs := Compile(source); len(errs) == 0 || !strings.Contains(errs[0].Error(), "ignore_labels apply only to set and span rules") {
		t.Fatalf("errors: %v", errs)
	}
}

func TestProjectionOnASystemOneOptionValidatesAndRoundTrips(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.SystemOneRules = []config.SystemOneRule{{
		Name: "needs",
		Question: config.SystemOneQuestion{Type: "set", Instructions: "What does a good answer need?", Labels: []config.SystemOneOption{
			{Key: "deliberation"}, {Key: "tools"},
		}},
	}}
	gte := 0.5
	cfg.Projections = config.Projections{
		Scores: []config.ProjectionScore{{Name: "effort", Method: "weighted_sum", Inputs: []config.ProjectionScoreInput{
			{Type: "systemone", Name: "needs:deliberation", Weight: 0.4, ValueSource: "raw"},
		}}},
		Mappings: []config.ProjectionMapping{{
			Name: "effort_band", Source: "effort", Method: "threshold_bands",
			Outputs: []config.ProjectionMappingOutput{{Name: "effort_high", GTE: &gte}},
		}},
	}
	cfg.Decisions = []config.Decision{{
		Name: "hard", Priority: 10, ModelRefs: []config.ModelRef{{Model: "model-a"}},
		Rules: config.RuleCombination{Type: "projection", Name: "effort_high"},
	}}
	source, err := Decompile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	diagnostics, errs := Validate(source)
	if len(errs) > 0 || len(diagnostics) > 0 {
		t.Fatalf("a projection on a declared question's option must validate cleanly: %v %v\n%s", errs, diagnostics, source)
	}
	compiled, compileErrs := Compile(source)
	if len(compileErrs) > 0 {
		t.Fatalf("compile errors: %v\n%s", compileErrs, source)
	}
	if !reflect.DeepEqual(compiled.Projections.Scores, cfg.Projections.Scores) {
		t.Fatalf("projection scores changed in the round trip: %#v", compiled.Projections.Scores)
	}
}

func TestSystemOneAlgorithmWithoutDeploymentRoundTrip(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{{
		Name:      "frontier",
		Priority:  10,
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmSystemOne, SystemOne: &config.SystemOneSelectionConfig{
			Instructions: "Which model should answer?",
			Candidates:   map[string]string{"model-a": "Fast", "model-b": "Strong"},
		}},
	}}
	source, err := Decompile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(source, "deployment") {
		t.Fatalf("a selector that asks the decision model names no deployment:\n%s", source)
	}
	compiled, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v\n%s", errs, source)
	}
	if !reflect.DeepEqual(compiled.Decisions[0].Algorithm.SystemOne, cfg.Decisions[0].Algorithm.SystemOne) {
		t.Fatalf("systemone algorithm changed in the round trip: %#v", compiled.Decisions[0].Algorithm.SystemOne)
	}
}
