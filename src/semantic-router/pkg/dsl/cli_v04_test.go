package dsl

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCLIDecompileFullManifestUsesInjectedProviderParser(t *testing.T) {
	manifestPath := filepath.Join("..", "..", "..", "..", "config", "recipes", "multi-objective", "config.yaml")
	outputPath := filepath.Join(t.TempDir(), "recipe.dsl")
	parser := maintainedConfigParser(t)
	if err := CLIDecompileWithParser(manifestPath, outputPath, parser.ParseYAMLBytes); err != nil {
		t.Fatalf("CLIDecompileWithParser() error = %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	for _, forbidden := range []string{
		"connections:", "credential:", "catalog_revision:", "recipe_id:",
		"decision_id:", "model_id:", "model_names:", "revision:",
	} {
		if strings.Contains(source, forbidden) {
			t.Fatalf("decompiled human DSL contains %q", forbidden)
		}
	}
	recompiled, compileErrs := Compile(source)
	if len(compileErrs) > 0 {
		t.Fatalf("compile decompiled full manifest: %v", compileErrs)
	}
	if len(recompiled.ModelConfig) == 0 || len(recompiled.Recipes) == 0 || len(recompiled.Entrypoints) == 0 {
		t.Fatalf("full manifest round trip lost v0.4 scopes: Models=%d Recipes=%d Entrypoints=%d", len(recompiled.ModelConfig), len(recompiled.Recipes), len(recompiled.Entrypoints))
	}
}

func TestCLIDecompileFullManifestFailsWithoutApplicationParser(t *testing.T) {
	manifestPath := filepath.Join("..", "..", "..", "..", "config", "recipes", "multi-objective", "config.yaml")
	if err := CLIDecompile(manifestPath, filepath.Join(t.TempDir(), "recipe.dsl")); err == nil ||
		!strings.Contains(err.Error(), "Recipe document") {
		t.Fatalf("provider-neutral CLIDecompile() error = %v", err)
	}
}

func TestDecompileYAMLRecipeBundleIsProviderNeutralAndExact(t *testing.T) {
	source := []byte(`
version: v0.4
recipes:
  - name: first
    description: First reusable policy.
    document:
      strategy: confidence
      decisions:
        - name: direct
          rules: {}
`)
	decompiled, err := DecompileYAML(source, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(decompiled, "# RECIPE first") ||
		!strings.Contains(decompiled, "strategy: confidence") ||
		strings.HasPrefix(decompiled, "# =============================================================================\n# SIGNALS") {
		t.Fatalf("Recipe bundle decompile contains the wrong scopes:\n%s", decompiled)
	}
	compiled, compileErrs := Compile(decompiled)
	if len(compileErrs) != 0 {
		t.Fatalf("compile decompiled Recipe bundle: %v", compileErrs)
	}
	roundTrip, err := Decompile(compiled)
	if err != nil {
		t.Fatal(err)
	}
	if roundTrip != decompiled {
		t.Fatalf("Recipe bundle round-trip is not byte exact")
	}
}
