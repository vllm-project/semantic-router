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
		t.Fatalf("full manifest round trip lost v0.3 scopes: Models=%d Recipes=%d Entrypoints=%d", len(recompiled.ModelConfig), len(recompiled.Recipes), len(recompiled.Entrypoints))
	}
}

func TestCLIDecompileFullManifestFailsWithoutApplicationParser(t *testing.T) {
	manifestPath := filepath.Join("..", "..", "..", "..", "config", "recipes", "multi-objective", "config.yaml")
	if err := CLIDecompile(manifestPath, filepath.Join(t.TempDir(), "recipe.dsl")); err == nil ||
		!strings.Contains(err.Error(), "Recipe authoring YAML") {
		t.Fatalf("provider-neutral CLIDecompile() error = %v", err)
	}
}
