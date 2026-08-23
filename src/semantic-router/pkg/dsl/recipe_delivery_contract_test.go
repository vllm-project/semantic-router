package dsl

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const builtInRecipeCatalogDirectory = "built-in"

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
		if entry.Name() == builtInRecipeCatalogDirectory {
			continue
		}
		t.Run(entry.Name(), func(t *testing.T) {
			assertRecipeDSLContract(t, filepath.Join(root, entry.Name()))
		})
	}
	t.Run("built-in/mom-v1", func(t *testing.T) {
		assertBuiltInRecipeDSLContract(t, filepath.Join(root, "built-in", "latest", "mom-v1"))
	})
}

func assertBuiltInRecipeDSLContract(t *testing.T, directory string) {
	t.Helper()
	yamlPath := filepath.Join(directory, "config.yaml")
	dslPath := filepath.Join(directory, "recipe.dsl")
	payload, err := os.ReadFile(yamlPath)
	if err != nil {
		t.Fatalf("read %s: %v", yamlPath, err)
	}
	runtimeConfig, err := parseRecipeBundleYAML(payload)
	if err != nil {
		t.Fatalf("parse Recipe bundle %s: %v", yamlPath, err)
	}
	dslBytes, err := os.ReadFile(dslPath)
	if err != nil {
		t.Fatalf("read %s: %v", dslPath, err)
	}
	dsl := string(dslBytes)
	if _, parseErrs := Parse(dsl); len(parseErrs) > 0 {
		t.Fatalf("parse DSL %s: %v", dslPath, parseErrs)
	}
	assertCanonicalRuntimeDSL(t, yamlPath, dslPath, dsl, runtimeConfig)
	compileStableRecipeDSL(t, dslPath, dsl)
}

func assertRecipeDSLContract(t *testing.T, directory string) {
	t.Helper()
	yamlPath := filepath.Join(directory, "config.yaml")
	dslPath := filepath.Join(directory, "recipe.dsl")
	runtimeConfig := parseMaintainedConfig(t, yamlPath)
	dslBytes, err := os.ReadFile(dslPath)
	if err != nil {
		t.Fatalf("read %s: %v", dslPath, err)
	}
	dsl := string(dslBytes)

	assertRecipeDSLValid(t, dslPath, dsl)
	assertCanonicalRuntimeDSL(t, yamlPath, dslPath, dsl, runtimeConfig)
	compiled := compileStableRecipeDSL(t, dslPath, dsl)
	assertMergedRecipeDSL(t, yamlPath, dslPath, dsl, runtimeConfig, compiled)
}

func assertRecipeDSLValid(t *testing.T, dslPath, dsl string) {
	t.Helper()
	diagnostics, parseErrs := Validate(dsl)
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
}

func assertCanonicalRuntimeDSL(
	t *testing.T,
	yamlPath string,
	dslPath string,
	dsl string,
	runtimeConfig *config.RouterConfig,
) {
	t.Helper()
	canonicalDSL, err := Decompile(runtimeConfig)
	if err != nil {
		t.Fatalf("decompile %s: %v", yamlPath, err)
	}
	if canonicalDSL != dsl {
		t.Fatalf("%s is not the canonical DSL generated from %s (-asset +generated):\n%s", dslPath, yamlPath, cmp.Diff(dsl, canonicalDSL))
	}
}

func compileStableRecipeDSL(t *testing.T, dslPath, dsl string) *config.RouterConfig {
	t.Helper()
	compiled, compileErrs := Compile(dsl)
	if len(compileErrs) > 0 {
		t.Fatalf("compile %s: %v", dslPath, compileErrs)
	}

	stableDSL, err := Decompile(compiled)
	if err != nil {
		t.Fatalf("decompile compiled %s: %v", dslPath, err)
	}
	if stableDSL != dsl {
		t.Fatalf("%s is not byte-stable after compile/decompile", dslPath)
	}
	return compiled
}

func assertMergedRecipeDSL(
	t *testing.T,
	yamlPath string,
	dslPath string,
	dsl string,
	baseConfig *config.RouterConfig,
	compiled *config.RouterConfig,
) {
	t.Helper()
	baseYAML, err := os.ReadFile(yamlPath)
	if err != nil {
		t.Fatalf("read base config %s: %v", yamlPath, err)
	}
	if len(baseConfig.Recipes) != 1 {
		if _, mergeErr := MergeRoutingIntoBase(compiled, baseYAML); mergeErr == nil {
			t.Fatalf("merging %s must reject a base with %d Recipes", dslPath, len(baseConfig.Recipes))
		}
		return
	}
	merged, err := MergeRoutingIntoBase(compiled, baseYAML)
	if err != nil {
		t.Fatalf("merge %s over %s: %v", dslPath, yamlPath, err)
	}
	mergedConfig := parseMaintainedConfigBytes(t, "DSL-generated "+dslPath, merged)
	mergedDSL, err := Decompile(mergedConfig)
	if err != nil {
		t.Fatalf("decompile DSL-generated %s: %v", dslPath, err)
	}
	if mergedDSL != dsl {
		t.Fatalf("%s changes after compile, base merge, and runtime parse", dslPath)
	}
	if len(mergedConfig.Recipes) != 1 {
		t.Fatalf("runtime parse of merged %s produced %d Recipes", dslPath, len(mergedConfig.Recipes))
	}
	assertDecisionAdaptationsPreserved(
		t,
		yamlPath,
		baseConfig.Recipes[0].Profile.Decisions,
		mergedConfig.Recipes[0].Profile.Decisions,
	)
}

func assertDecisionAdaptationsPreserved(
	t *testing.T,
	yamlPath string,
	baseDecisions []config.Decision,
	mergedDecisions []config.Decision,
) {
	t.Helper()
	mergedByName := make(map[string]config.Decision, len(mergedDecisions))
	for _, decision := range mergedDecisions {
		mergedByName[decision.Name] = decision
	}
	for _, decision := range baseDecisions {
		merged, ok := mergedByName[decision.Name]
		if !ok {
			t.Fatalf("%s lost decision %q while merging DSL", yamlPath, decision.Name)
		}
		if !reflect.DeepEqual(merged.Adaptations, decision.Adaptations) {
			t.Fatalf(
				"%s decision %q adaptations changed after DSL merge: got %#v want %#v",
				yamlPath,
				decision.Name,
				merged.Adaptations,
				decision.Adaptations,
			)
		}
	}
}
