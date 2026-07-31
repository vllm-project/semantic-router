package selection

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPromptSelectorChoosesDeclaredCandidate(t *testing.T) {
	selector := NewPromptSelector(
		config.PromptSelectionConfig{
			Model:        "router-small",
			Instructions: "Choose by difficulty.",
		},
		func(_ context.Context, model, systemPrompt, input string) (string, error) {
			if model != "router-small" {
				t.Fatalf("model = %q", model)
			}
			if !strings.Contains(systemPrompt, "reasoning-large: Hard reasoning") {
				t.Fatalf("system prompt = %q", systemPrompt)
			}
			if input != "solve this proof" {
				t.Fatalf("input = %q", input)
			}
			return `{"selected_model":"reasoning-large","rationale":"Requires multi-step reasoning."}`, nil
		},
		map[string]string{"reasoning-large": "Hard reasoning"},
	)

	result, err := selector.Select(context.Background(), &SelectionContext{
		Query: "solve this proof",
		CandidateModels: []config.ModelRef{
			{Model: "general-small"},
			{Model: "reasoning-large"},
		},
	})
	if err != nil {
		t.Fatalf("Select() error = %v", err)
	}
	if result.SelectedModel != "reasoning-large" {
		t.Fatalf("SelectedModel = %q", result.SelectedModel)
	}
}

func TestPromptSelectorRejectsUndeclaredCandidate(t *testing.T) {
	selector := NewPromptSelector(
		config.PromptSelectionConfig{Model: "router-small", Instructions: "Choose."},
		func(context.Context, string, string, string) (string, error) {
			return `{"selected_model":"invented","rationale":"No reason."}`, nil
		},
		nil,
	)
	_, err := selector.Select(context.Background(), &SelectionContext{
		Query:           "hello",
		CandidateModels: []config.ModelRef{{Model: "general-small"}},
	})
	if err == nil {
		t.Fatal("Select() expected undeclared candidate error")
	}
}
