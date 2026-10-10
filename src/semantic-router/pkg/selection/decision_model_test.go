package selection

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func decisionSelectionContext() *SelectionContext {
	return &SelectionContext{
		Query:           "Prove that there are infinitely many primes.",
		DecisionName:    "math",
		CandidateModels: []config.ModelRef{{Model: "large"}, {Model: "small", LoRAName: "fast"}},
	}
}

func TestDecisionModelSelectorChoosesTheAnswer(t *testing.T) {
	var asked []DecisionModelChoice
	var state, instructions string
	selector := NewDecisionModelSelector(
		config.DecisionSelectionConfig{Deployment: "kai", Instructions: "Which model?", Candidates: map[string]string{"large": "Best for proofs"}},
		func(_ context.Context, gotInstructions, gotState string, choices []DecisionModelChoice) (DecisionModelAnswer, error) {
			asked, state, instructions = choices, gotState, gotInstructions
			return DecisionModelAnswer{Choice: "small", Probabilities: map[string]float64{"large": 0.3, "small": 0.7}, Confidence: 0.12}, nil
		},
		map[string]string{"small": "Fast and cheap"},
	)
	result, err := selector.Select(context.Background(), decisionSelectionContext())
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedModel != "small" || result.LoRAName != "fast" || result.Method != MethodDecision {
		t.Fatalf("result = %+v", result)
	}
	if result.Score != 0.7 || result.AllScores["large"] != 0.3 || len(result.CandidateScores) != 2 {
		t.Fatalf("scores = %+v", result)
	}
	if state != "Prove that there are infinitely many primes." || instructions != "Which model?" {
		t.Fatalf("question = %q / %q", instructions, state)
	}
	if asked[0].Description != "Best for proofs" || asked[1].Description != "Fast and cheap" {
		t.Fatalf("candidate descriptions = %+v", asked)
	}
}

func TestDecisionModelSelectorErrorsFallBack(t *testing.T) {
	failing := NewDecisionModelSelector(config.DecisionSelectionConfig{Deployment: "kai", Instructions: "?"},
		func(context.Context, string, string, []DecisionModelChoice) (DecisionModelAnswer, error) {
			return DecisionModelAnswer{}, errors.New("runtime is not ready")
		}, nil)
	if _, err := failing.Select(context.Background(), decisionSelectionContext()); !errors.Is(err, ErrDecisionModelInvocation) {
		t.Fatalf("invocation error = %v", err)
	}
	undeclared := NewDecisionModelSelector(config.DecisionSelectionConfig{Deployment: "kai", Instructions: "?"},
		func(context.Context, string, string, []DecisionModelChoice) (DecisionModelAnswer, error) {
			return DecisionModelAnswer{Choice: "other", Probabilities: map[string]float64{"other": 1}}, nil
		}, nil)
	if _, err := undeclared.Select(context.Background(), decisionSelectionContext()); !errors.Is(err, ErrDecisionModelAnswer) {
		t.Fatalf("undeclared answer error = %v", err)
	}
}
