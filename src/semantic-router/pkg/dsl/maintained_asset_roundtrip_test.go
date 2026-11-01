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

func maintainedBalanceAsset(name string) string {
	return filepath.Join("..", "..", "..", "..", "config", "recipes", "balance", name)
}

func TestMaintainedBalanceRecipeHasNoUndefinedComplexitySignals(t *testing.T) {
	cfg := mustLoadMaintainedBalanceRouterConfig(t, maintainedBalanceAsset("config.yaml"))

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

// Every specialist and follow-up lane reads a named decision-model question;
// only the image lane and the final fallback route without one.
func TestMaintainedBalanceLanesReadNamedQuestions(t *testing.T) {
	cfg := mustLoadMaintainedBalanceRouterConfig(t, maintainedBalanceAsset("config.yaml"))

	questions := make(map[string]bool, len(cfg.DecisionRules))
	for _, rule := range cfg.DecisionRules {
		questions[rule.Name] = true
	}
	for _, name := range []string{"task", "verification", "correction", "clarify"} {
		if !questions[name] {
			t.Fatalf("expected balance to ask the %q question, got %v", name, questions)
		}
	}
	if len(cfg.Projections.Scores) != 0 || len(cfg.Projections.Partitions) != 0 {
		t.Fatalf("expected balance to route on questions without projections, got %+v", cfg.Projections)
	}

	assertMaintainedBalanceDecisionTiers(t, cfg.Decisions)
	for _, decision := range cfg.Decisions {
		switch decision.Name {
		case "omni", "casual_chat":
			continue
		}
		if !ruleTreeReadsSignalType(&decision.Rules, config.SignalTypeDecision) {
			t.Fatalf("expected %s to read a decision-model question", decision.Name)
		}
	}
	if fallback := mustFindMaintainedBalanceDecision(t, cfg.Decisions, "casual_chat"); len(fallback.Rules.Conditions) != 0 || fallback.Rules.Type != "" {
		t.Fatalf("expected casual_chat to be the unconditional fallback, got %+v", fallback.Rules)
	}
}

func TestMaintainedBalanceRoutingAssetsStayInSync(t *testing.T) {
	prog := mustLoadMaintainedBalanceDSLProgram(t, maintainedBalanceAsset("recipe.dsl"))
	want := mustCompileMaintainedRoutingDSL(t, prog)
	got := mustLoadMaintainedBalanceRoutingYAML(t, maintainedBalanceAsset("config.yaml"))
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("maintained DSL/YAML examples diverged (-want +got):\n%s", cmp.Diff(want, got))
	}
}

func TestMaintainedBalanceWarningBudgetStaysBelowCeiling(t *testing.T) {
	cfg := mustLoadMaintainedBalanceRouterConfig(t, maintainedBalanceAsset("config.yaml"))

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}
	diags, errs := Validate(dslText)
	if len(errs) > 0 {
		t.Fatalf("Validate parse errors: %v", errs)
	}

	warnings := 0
	for _, diag := range diags {
		if diag.Level == DiagError || diag.Level == DiagConstraint {
			t.Fatalf("unexpected maintained balance diagnostic: %s", diag.String())
		}
		if diag.Level == DiagWarning {
			if !strings.Contains(diag.Message, "no mutual exclusion guard") {
				t.Fatalf("unexpected maintained balance warning category: %s", diag.Message)
			}
			warnings++
		}
	}

	const maxWarnings = 0
	if warnings > maxWarnings {
		t.Fatalf("expected maintained balance warning count <= %d after recipe guard tightening, got %d", maxWarnings, warnings)
	}
}

func mustLoadMaintainedBalanceDSLProgram(t *testing.T, dslPath string) *Program {
	t.Helper()

	dslData, err := os.ReadFile(dslPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", dslPath, err)
	}
	assertMaintainedBalanceDSLMarkers(t, dslPath, string(dslData))

	prog, errs := Parse(string(dslData))
	if len(errs) > 0 {
		t.Fatalf("Parse errors: %v", errs)
	}
	assertMaintainedBalanceDSLDiagnostics(t, prog)
	return prog
}

func assertMaintainedBalanceDecisionTiers(t *testing.T, decisions []config.Decision) {
	t.Helper()

	for _, decision := range decisions {
		if decision.Tier <= 0 {
			t.Fatalf("expected decision %q to define a positive tier", decision.Name)
		}
	}
}

func assertMaintainedBalanceDSLMarkers(t *testing.T, dslPath, dslText string) {
	t.Helper()
	for _, marker := range []string{
		"SIGNAL decision task",
		"SIGNAL decision verification",
		"SIGNAL decision correction",
		"SIGNAL decision clarify",
		"ROUTE formal_math_proof",
		"ROUTE reasoning_deep",
		"ROUTE complex_specialist",
		"ROUTE fast_qa",
		"ROUTE casual_chat",
	} {
		if !strings.Contains(dslText, marker) {
			t.Fatalf("%s must contain %q", dslPath, marker)
		}
	}
}

func assertMaintainedBalanceDSLDiagnostics(t *testing.T, prog *Program) {
	t.Helper()
	for _, diag := range ValidateAST(prog) {
		if diag.Level == DiagError || diag.Level == DiagConstraint {
			t.Fatalf("unexpected maintained DSL diagnostic: %s", diag.String())
		}
	}
}

func mustCompileMaintainedRoutingDSL(t *testing.T, prog *Program) config.CanonicalRouting {
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

func mustLoadMaintainedBalanceRoutingYAML(t *testing.T, yamlPath string) config.CanonicalRouting {
	t.Helper()

	parsedCfg := mustLoadMaintainedBalanceRouterConfig(t, yamlPath)
	return config.CanonicalRoutingFromRouterConfig(parsedCfg)
}

func mustLoadMaintainedBalanceRouterConfig(t *testing.T, yamlPath string) *config.RouterConfig {
	t.Helper()

	yamlData, err := os.ReadFile(yamlPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", yamlPath, err)
	}
	parsedCfg, err := config.ParseYAMLBytes(yamlData)
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}
	return parsedCfg
}

func mustFindMaintainedBalanceDecision(t *testing.T, decisions []config.Decision, name string) *config.Decision {
	t.Helper()

	for i := range decisions {
		if decisions[i].Name == name {
			return &decisions[i]
		}
	}
	t.Fatalf("expected config/recipes/balance/config.yaml to include %s route", name)
	return nil
}

func ruleTreeReadsSignalType(node *config.RuleCombination, signalType string) bool {
	if node == nil {
		return false
	}
	if node.Type != "" {
		return node.Type == signalType
	}
	for i := range node.Conditions {
		if ruleTreeReadsSignalType(&node.Conditions[i], signalType) {
			return true
		}
	}
	return false
}
