package dsl

import (
	"reflect"
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
