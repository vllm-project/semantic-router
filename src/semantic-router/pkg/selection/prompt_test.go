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
		func(_ context.Context, model, systemPrompt, input string) (PromptInvocationResult, error) {
			if model != "router-small" {
				t.Fatalf("model = %q", model)
			}
			if !strings.Contains(systemPrompt, "reasoning-large: Hard reasoning") {
				t.Fatalf("system prompt = %q", systemPrompt)
			}
			if input != "solve this proof" {
				t.Fatalf("input = %q", input)
			}
			return PromptInvocationResult{
				Content:     `{"selected_model":"reasoning-large","rationale":"Requires multi-step reasoning."}`,
				Model:       model,
				TotalTokens: 12,
				LatencyMs:   25,
			}, nil
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
	if result.HelperModel != "router-small" ||
		result.HelperTotalTokens != 12 ||
		result.HelperLatencyMs != 25 {
		t.Fatalf("helper telemetry = %#v", result)
	}
}

func TestPromptSelectorRejectsUndeclaredCandidate(t *testing.T) {
	selector := NewPromptSelector(
		config.PromptSelectionConfig{Model: "router-small", Instructions: "Choose."},
		func(context.Context, string, string, string) (PromptInvocationResult, error) {
			return PromptInvocationResult{
				Content: `{"selected_model":"invented","rationale":"No reason."}`,
			}, nil
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
	if strings.Contains(err.Error(), "invented") {
		t.Fatal("undeclared model name leaked into selector error")
	}
}

func TestPromptSelectorRejectsDuplicateBaseModels(t *testing.T) {
	selector := NewPromptSelector(
		config.PromptSelectionConfig{Model: "router-small", Instructions: "Choose."},
		func(context.Context, string, string, string) (PromptInvocationResult, error) {
			t.Fatal("duplicate candidates must fail before invoking the helper model")
			return PromptInvocationResult{}, nil
		},
		nil,
	)
	_, err := selector.Select(context.Background(), &SelectionContext{
		Query: "hello",
		CandidateModels: []config.ModelRef{
			{Model: "general-small"},
			{Model: "general-small", LoRAName: "specialist"},
		},
	})
	if err == nil {
		t.Fatal("Select() expected duplicate candidate error")
	}
}

func TestPromptSelectorDeclaresRequiredHelperDependency(t *testing.T) {
	selector := NewPromptSelector(
		config.PromptSelectionConfig{},
		nil,
		nil,
	)
	dependencies := selector.ExternalDependencies()
	if len(dependencies) != 1 ||
		dependencies[0].Type != DependencyExternalService ||
		!dependencies[0].Required {
		t.Fatalf("dependencies = %#v", dependencies)
	}
}
