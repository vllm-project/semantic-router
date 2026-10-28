package dsl

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const cliNamedRecipeConfig = `version: v0.3
providers:
  models:
    - name: local-model
      reasoning:
        family: qwen3
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: localhost:8000
          protocol: http
routing:
  modelCards:
    - name: local-model
entrypoints:
  - model_names: [my/balance]
    recipe: balance
recipes:
  - name: balance
    routing:
      decisions:
        - name: default-route
          priority: 100
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: local-model
              use_reasoning: true
              reasoning_mode: enabled
`

func TestCLIDecompilePreservesNamedRecipeAndEntrypoint(t *testing.T) {
	dir := t.TempDir()
	input, output := filepath.Join(dir, "config.yaml"), filepath.Join(dir, "recipe.dsl")
	if err := os.WriteFile(input, []byte(cliNamedRecipeConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := CLIDecompile(input, output); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	compiled, errors := Compile(string(data))
	if len(errors) != 0 {
		t.Fatalf("roundtrip compile: %v", errors)
	}
	if len(compiled.Entrypoints) != 1 || compiled.Entrypoints[0].Recipe != "balance" ||
		len(compiled.Entrypoints[0].ModelNames) != 1 || compiled.Entrypoints[0].ModelNames[0] != "my/balance" {
		t.Fatalf("entrypoint lost: %+v", compiled.Entrypoints)
	}
	for _, recipe := range compiled.Recipes {
		if recipe.Name == "balance" {
			if len(recipe.Profile.Decisions) != 1 || recipe.Profile.Decisions[0].Name != "default-route" {
				t.Fatalf("recipe decisions lost: %+v", recipe)
			}
			return
		}
	}
	t.Fatalf("named recipe lost: %s", data)
}

func TestCLIDecompileRejectsInvalidNamedRecipeWithoutLosingScope(t *testing.T) {
	dir := t.TempDir()
	input, output := filepath.Join(dir, "config.yaml"), filepath.Join(dir, "recipe.dsl")
	invalid := strings.Replace(cliNamedRecipeConfig, "reasoning_mode: enabled", "reasoning_effort: high", 1)
	if err := os.WriteFile(input, []byte(invalid), 0o600); err != nil {
		t.Fatal(err)
	}
	err := CLIDecompile(input, output)
	if err == nil || !strings.Contains(err.Error(), "mode-only reasoning family") {
		t.Fatalf("expected source error instead of empty DSL, got %v", err)
	}
	if _, err := os.Stat(output); !os.IsNotExist(err) {
		t.Fatalf("invalid config produced an output file: %v", err)
	}
}
