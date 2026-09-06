package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCompilePromptAlgorithm(t *testing.T) {
	input := `
ROUTE prompt_route {
  PRIORITY 10
  MODEL "model-a", "model-b"
  ALGORITHM prompt {
    prompt: {
      model: "router-small",
      instructions: "Choose the exact candidate.",
      timeout_seconds: 5
    }
    on_error: "fallback"
  }
}`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	algorithm := cfg.Decisions[0].Algorithm
	if algorithm == nil || algorithm.Prompt == nil {
		t.Fatal("prompt algorithm was not compiled")
	}
	if algorithm.Prompt.Model != "router-small" ||
		algorithm.Prompt.Instructions != "Choose the exact candidate." ||
		algorithm.Prompt.TimeoutSeconds != 5 ||
		algorithm.OnError != "fallback" {
		t.Fatalf("prompt algorithm = %#v", algorithm)
	}
}

func TestDecompilePromptAlgorithmRoundTrip(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:     "prompt-route",
				Priority: 10,
				ModelRefs: []config.ModelRef{
					{Model: "model-a"},
					{Model: "model-b"},
				},
				Algorithm: &config.AlgorithmConfig{
					Type:    "prompt",
					OnError: "fallback",
					Prompt: &config.PromptSelectionConfig{
						Model:          "router-small",
						Instructions:   "Choose the exact candidate.",
						TimeoutSeconds: 5,
					},
				},
			}},
		},
	}

	source, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}
	for _, expected := range []string{
		"ALGORITHM prompt",
		`model: "router-small"`,
		`instructions: "Choose the exact candidate."`,
		"timeout_seconds: 5",
		`on_error: "fallback"`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("decompiled source missing %q:\n%s", expected, source)
		}
	}
	roundTrip, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatalf("round-trip compile errors: %v", errs)
	}
	prompt := roundTrip.Decisions[0].Algorithm.Prompt
	if prompt == nil || prompt.Model != "router-small" ||
		prompt.Instructions != "Choose the exact candidate." ||
		prompt.TimeoutSeconds != 5 {
		t.Fatalf("round-trip prompt = %#v", prompt)
	}
}
