package dsl

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCLIDecompileUnboundRecipeOnlyDocument(t *testing.T) {
	var source strings.Builder
	source.WriteString("version: v0.3\nrecipes:\n")
	for _, name := range []string{"alpha", "beta", "gamma", "delta", "epsilon"} {
		fmt.Fprintf(&source, `  - name: %s
    routing:
      candidate_requirements: {capabilities: declared, context: known_limits}
      data_policy: {replay: false}
      decisions:
        - name: primary
          priority: 1
          rules: {operator: AND}
          algorithm:
            type: multi_factor
            minimum_candidates: 1
            multi_factor:
              objective:
                strategy: lexicographic
                priorities: [{factor: latency}, {factor: cost}]
              latency_metric: ttft
              quality: {index: "vllm-sr/general@1.0.0", on_missing: exclude, min_coverage: 0.8, min_score: 0}
          plugins:
            - type: request_params
              configuration: {default_max_tokens: 4096}
`, name)
	}
	input := []byte(source.String())
	want, err := config.ParseYAMLBytes(input)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	yamlPath, dslPath := filepath.Join(dir, "recipes.yaml"), filepath.Join(dir, "recipes.dsl")
	if writeErr := os.WriteFile(yamlPath, input, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	if decompileErr := CLIDecompile(yamlPath, dslPath); decompileErr != nil {
		t.Fatal(decompileErr)
	}
	data, err := os.ReadFile(dslPath)
	if err != nil {
		t.Fatal(err)
	}
	got, errs := Compile(string(data))
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	original := config.CanonicalConfigFromRouterConfig(want).Recipes
	compiled := config.CanonicalConfigFromRouterConfig(got).Recipes
	originalYAML, err := yaml.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	compiledYAML, err := yaml.Marshal(compiled)
	if err != nil {
		t.Fatal(err)
	}
	if len(original) != 5 || !bytes.Equal(originalYAML, compiledYAML) {
		t.Fatal("unbound named recipes changed through CLI decompile/compile")
	}
	if len(got.ModelConfig) != 0 || len(got.VLLMEndpoints) != 0 {
		t.Fatal("unbound recipe roundtrip invented model assignments")
	}
	again, err := Decompile(got)
	if err != nil || string(data) != again {
		t.Fatalf("DSL is not byte-stable: %v", err)
	}
}

func TestCLIDecompilePreservesCanonicalValidationError(t *testing.T) {
	dir := t.TempDir()
	input, output := filepath.Join(dir, "invalid.yaml"), filepath.Join(dir, "output.dsl")
	if err := os.WriteFile(input, []byte("version: invalid\nrecipes: []\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	err := CLIDecompile(input, output)
	if err == nil || !strings.Contains(err.Error(), "unsupported config version") {
		t.Fatalf("canonical error lost: %v", err)
	}
	if _, err := os.Stat(output); !os.IsNotExist(err) {
		t.Fatal("invalid document emitted DSL")
	}
}
