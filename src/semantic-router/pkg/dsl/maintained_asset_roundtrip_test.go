package dsl

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMaintainedBalanceRecipeHasNoUndefinedComplexitySignals(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "balance.yaml")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Fatalf("failed to read deploy/recipes/balance.yaml: %v", err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}

	diags, errs := Validate(dslText)
	if len(errs) > 0 {
		t.Fatalf("Validate parse errors: %v", errs)
	}

	for _, diag := range diags {
		if diag.Level == DiagWarning && strings.Contains(diag.Message, "Signal 'complexity'(") && strings.Contains(diag.Message, "is not defined") {
			t.Fatalf("unexpected undefined complexity signal warning: %s", diag.Message)
		}
	}
}

func TestMaintainedBalanceRecipeUsesSignalGroupsAndTieredDecisions(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "balance.yaml")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Fatalf("failed to read deploy/recipes/balance.yaml: %v", err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}
	if len(cfg.SignalGroups) == 0 {
		t.Fatal("expected deploy/recipes/balance.yaml to include at least one signal_group")
	}

	for _, decision := range cfg.Decisions {
		if decision.Tier <= 0 {
			t.Fatalf("expected decision %q to define a positive tier", decision.Name)
		}
	}
}

func TestMaintainedConflictFreeRoutingExamplesStayInSync(t *testing.T) {
	dslPath := filepath.Join("..", "..", "..", "..", "deploy", "examples", "runtime", "routing", "conflict-free-routing.dsl")
	yamlPath := filepath.Join("..", "..", "..", "..", "deploy", "examples", "runtime", "routing", "conflict-free-routing.yaml")

	prog := mustLoadMaintainedConflictFreeDSLProgram(t, dslPath)
	want := mustCompileMaintainedConflictFreeRouting(t, prog)
	got := mustLoadMaintainedConflictFreeRoutingYAML(t, yamlPath)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("maintained DSL/YAML examples diverged\nwant: %+v\ngot: %+v", want, got)
	}
}

func mustLoadMaintainedConflictFreeDSLProgram(t *testing.T, dslPath string) *Program {
	t.Helper()

	dslData, err := os.ReadFile(dslPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", dslPath, err)
	}
	assertMaintainedConflictFreeDSLMarkers(t, dslPath, string(dslData))

	prog, errs := Parse(string(dslData))
	if len(errs) > 0 {
		t.Fatalf("Parse errors: %v", errs)
	}
	if len(prog.SignalGroups) != 1 {
		t.Fatalf("expected 1 signal group in maintained DSL example, got %d", len(prog.SignalGroups))
	}
	if len(prog.TestBlocks) != 1 {
		t.Fatalf("expected 1 test block in maintained DSL example, got %d", len(prog.TestBlocks))
	}
	assertMaintainedConflictFreeDSLDiagnostics(t, prog)
	return prog
}

func assertMaintainedConflictFreeDSLMarkers(t *testing.T, dslPath, dslText string) {
	t.Helper()
	if !strings.Contains(dslText, "DECISION_TREE") {
		t.Fatalf("%s must demonstrate DECISION_TREE authoring", dslPath)
	}
	if !strings.Contains(dslText, "TEST routing_intent") {
		t.Fatalf("%s must demonstrate TEST blocks", dslPath)
	}
}

func assertMaintainedConflictFreeDSLDiagnostics(t *testing.T, prog *Program) {
	t.Helper()
	for _, diag := range ValidateAST(prog) {
		if diag.Level == DiagError || diag.Level == DiagConstraint {
			t.Fatalf("unexpected maintained DSL diagnostic: %s", diag.String())
		}
	}
}

func mustCompileMaintainedConflictFreeRouting(t *testing.T, prog *Program) config.CanonicalRouting {
	t.Helper()

	compiledCfg, compileErrs := CompileAST(prog)
	if len(compileErrs) > 0 {
		t.Fatalf("CompileAST errors: %v", compileErrs)
	}
	compiledYAML, err := EmitRoutingYAMLFromConfig(compiledCfg)
	if err != nil {
		t.Fatalf("EmitRoutingYAMLFromConfig error: %v", err)
	}
	compiledParsedCfg, err := config.ParseRoutingYAMLBytes(compiledYAML)
	if err != nil {
		t.Fatalf("ParseRoutingYAMLBytes(compiledYAML) error: %v", err)
	}
	return config.CanonicalRoutingFromRouterConfig(compiledParsedCfg)
}

func mustLoadMaintainedConflictFreeRoutingYAML(t *testing.T, yamlPath string) config.CanonicalRouting {
	t.Helper()

	yamlData, err := os.ReadFile(yamlPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", yamlPath, err)
	}
	parsedCfg, err := config.ParseRoutingYAMLBytes(yamlData)
	if err != nil {
		t.Fatalf("ParseRoutingYAMLBytes error: %v", err)
	}
	return config.CanonicalRoutingFromRouterConfig(parsedCfg)
}
