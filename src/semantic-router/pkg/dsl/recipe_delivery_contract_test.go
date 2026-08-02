package dsl

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMaintainedRecipeDSLMatchesEveryRuntimeConfig(t *testing.T) {
	root := filepath.Join("..", "..", "..", "..", "config", "recipes")
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		t.Run(entry.Name(), func(t *testing.T) {
			assertRecipeDSLContract(t, filepath.Join(root, entry.Name()))
		})
	}
}

func assertRecipeDSLContract(t *testing.T, directory string) {
	t.Helper()
	yamlPath := filepath.Join(directory, "config.yaml")
	dslPath := filepath.Join(directory, "recipe.dsl")
	runtimeConfig, err := config.Parse(yamlPath)
	if err != nil {
		t.Fatalf("parse %s: %v", yamlPath, err)
	}
	dslBytes, err := os.ReadFile(dslPath)
	if err != nil {
		t.Fatalf("read %s: %v", dslPath, err)
	}

	diagnostics, parseErrs := Validate(string(dslBytes))
	if len(parseErrs) > 0 {
		t.Fatalf("parse DSL %s: %v", dslPath, parseErrs)
	}
	if len(diagnostics) > 0 {
		messages := make([]string, 0, len(diagnostics))
		for _, diagnostic := range diagnostics {
			messages = append(messages, diagnostic.String())
		}
		t.Fatalf("%s must validate without diagnostics:\n%s", dslPath, strings.Join(messages, "\n"))
	}

	// The committed DSL must be the canonical projection of the runnable
	// config. This avoids comparing raw structs where semantically equivalent
	// forms (for example, a single-condition OR versus AND) differ only in
	// normalization details.
	canonicalDSL, err := Decompile(runtimeConfig)
	if err != nil {
		t.Fatalf("decompile %s: %v", yamlPath, err)
	}
	if canonicalDSL != string(dslBytes) {
		t.Fatalf("%s is not the canonical DSL generated from %s", dslPath, yamlPath)
	}

	compiled, compileErrs := Compile(string(dslBytes))
	if len(compileErrs) > 0 {
		t.Fatalf("compile %s: %v", dslPath, compileErrs)
	}

	stableDSL, err := Decompile(compiled)
	if err != nil {
		t.Fatalf("decompile compiled %s: %v", dslPath, err)
	}
	if stableDSL != string(dslBytes) {
		t.Fatalf("%s is not byte-stable after compile/decompile", dslPath)
	}

	// Compiling the DSL is only useful if it can be overlaid on the scenario's
	// infrastructure config and accepted by the real runtime parser.
	baseYAML, err := os.ReadFile(yamlPath)
	if err != nil {
		t.Fatalf("read base config %s: %v", yamlPath, err)
	}
	merged, err := MergeRoutingIntoBase(compiled, baseYAML)
	if err != nil {
		t.Fatalf("merge %s over %s: %v", dslPath, yamlPath, err)
	}
	mergedConfig, err := config.ParseYAMLBytes(merged)
	if err != nil {
		t.Fatalf("runtime parse of DSL-generated %s: %v", dslPath, err)
	}
	mergedDSL, err := Decompile(mergedConfig)
	if err != nil {
		t.Fatalf("decompile DSL-generated %s: %v", dslPath, err)
	}
	if mergedDSL != string(dslBytes) {
		t.Fatalf("%s changes after compile, base merge, and runtime parse", dslPath)
	}
}
