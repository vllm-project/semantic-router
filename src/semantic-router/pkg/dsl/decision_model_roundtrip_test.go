package dsl

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionModelSignalAndAlgorithmRoundTrip(t *testing.T) {
	threshold := 0.7
	cfg := &config.RouterConfig{}
	cfg.DecisionRules = []config.DecisionSignalRule{
		{
			Name: "needs_reasoning", Description: "Needs reasoning", Deployment: "decision-kai",
			Question:  config.DecisionQuestion{Type: "noul", Instructions: "Does this need reasoning?"},
			Predicate: &config.NumericPredicate{GTE: &threshold}, TimeoutMs: 800,
		},
		{
			Name: "request_kind", Deployment: "decision-kai",
			Question: config.DecisionQuestion{Type: "choice", Instructions: "Which kind?", Choices: []config.DecisionChoice{
				{Key: "code", Description: "Code"}, {Key: "math"},
			}},
		},
		{
			Name: "difficulty", Deployment: "decision-kai", Predicate: &config.NumericPredicate{GTE: &threshold},
			Question: config.DecisionQuestion{Type: "score", Instructions: "How hard?", Levels: []string{"easy", "medium", "hard"}},
		},
		{
			Name: "topics", Deployment: "vela2", Predicate: &config.NumericPredicate{GTE: &threshold},
			Question: config.DecisionQuestion{Type: "set", Instructions: "Which topics?", Threshold: &threshold, Labels: []config.DecisionChoice{
				{Key: "billing", Description: "Payments"}, {Key: "shipping"},
			}},
		},
		{
			Name: "places", Deployment: "vela2",
			Question: config.DecisionQuestion{Type: "span", Instructions: "Which spans name a place?", Head: "broad", Labels: []config.DecisionChoice{{Key: "city"}}},
		},
	}
	cfg.Decisions = []config.Decision{{
		Name:      "decision-route",
		Priority:  10,
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		Algorithm: &config.AlgorithmConfig{Type: "decision", Decision: &config.DecisionSelectionConfig{
			Deployment: "decision-kai", Instructions: "Which model?", TimeoutMs: 900,
			Candidates: map[string]string{"model-a": "Fast", "model-b": "Strong"},
		}},
	}}
	source, err := Decompile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	compiled, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v\n%s", errs, source)
	}
	if !reflect.DeepEqual(compiled.DecisionRules, cfg.DecisionRules) {
		t.Fatalf("decision signals changed in the round trip:\n got %#v\nwant %#v\n%s", compiled.DecisionRules, cfg.DecisionRules, source)
	}
	if !reflect.DeepEqual(compiled.Decisions[0].Algorithm.Decision, cfg.Decisions[0].Algorithm.Decision) {
		t.Fatalf("decision algorithm changed in the round trip: %#v", compiled.Decisions[0].Algorithm.Decision)
	}
}

func TestProjectionOnADecisionOptionValidatesAndRoundTrips(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.DecisionRules = []config.DecisionSignalRule{{
		Name: "needs",
		Question: config.DecisionQuestion{Type: "set", Instructions: "What does a good answer need?", Labels: []config.DecisionChoice{
			{Key: "deliberation"}, {Key: "tools"},
		}},
	}}
	gte := 0.5
	cfg.Projections = config.Projections{
		Scores: []config.ProjectionScore{{Name: "effort", Method: "weighted_sum", Inputs: []config.ProjectionScoreInput{
			{Type: "decision", Name: "needs:deliberation", Weight: 0.4, ValueSource: "raw"},
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

func TestDecisionAlgorithmWithoutDeploymentRoundTrip(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{{
		Name:      "frontier",
		Priority:  10,
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		Algorithm: &config.AlgorithmConfig{Type: "decision", Decision: &config.DecisionSelectionConfig{
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
	if !reflect.DeepEqual(compiled.Decisions[0].Algorithm.Decision, cfg.Decisions[0].Algorithm.Decision) {
		t.Fatalf("decision algorithm changed in the round trip: %#v", compiled.Decisions[0].Algorithm.Decision)
	}
}
