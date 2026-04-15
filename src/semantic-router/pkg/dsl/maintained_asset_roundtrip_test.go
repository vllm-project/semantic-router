package dsl

import (
	"os"
	"path/filepath"
	"reflect"
	"slices"
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

func TestMaintainedBalanceRecipeUsesProjectionPartitionsAndTieredDecisions(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "balance.yaml")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Fatalf("failed to read deploy/recipes/balance.yaml: %v", err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}
	if len(cfg.Projections.Partitions) == 0 {
		t.Fatal("expected deploy/recipes/balance.yaml to include at least one projection partition")
	}
	if len(cfg.Projections.Scores) == 0 || len(cfg.Projections.Mappings) == 0 {
		t.Fatalf("expected deploy/recipes/balance.yaml to include derived projection scores and mappings, got %+v", cfg.Projections)
	}
	assertMaintainedBalanceDomainPartition(t, cfg.Projections.Partitions)
	assertMaintainedBalanceIntentPartition(t, cfg.Projections.Partitions)
	assertMaintainedBalanceProjectionBands(t, cfg.Projections)
	assertMaintainedBalanceDecisionTiers(t, cfg.Decisions)
	assertMaintainedBalanceRoute(t, cfg.Decisions, "formal_math_proof")
	assertMaintainedBalanceRoute(t, cfg.Decisions, "complex_specialist")
	assertMaintainedBalanceRoute(t, cfg.Decisions, "reasoning_deep")
	assertMaintainedBalanceRoute(t, cfg.Decisions, "verified_health")
	assertMaintainedBalanceRoute(t, cfg.Decisions, "fast_qa")
	assertMaintainedBalanceRoute(t, cfg.Decisions, "casual_chat")
}

func TestMaintainedBalanceRoutingAssetsStayInSync(t *testing.T) {
	dslPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "balance.dsl")
	yamlPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "balance.yaml")

	prog := mustLoadMaintainedBalanceDSLProgram(t, dslPath)
	want := mustCompileMaintainedRoutingDSL(t, prog)
	got := mustLoadMaintainedBalanceRoutingYAML(t, yamlPath)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("maintained DSL/YAML examples diverged\nwant: %+v\ngot: %+v", want, got)
	}
}

func TestMaintainedBalanceBaseRoutesExplicitlyExcludeVerifiedOverlay(t *testing.T) {
	yamlPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "balance.yaml")
	cfg := mustLoadMaintainedBalanceRouterConfig(t, yamlPath)

	for _, routeName := range []string{
		"medium_explainer",
		"simple_general",
	} {
		decision := mustFindMaintainedBalanceDecision(t, cfg.Decisions, routeName)
		if !ruleTreeContainsNegatedSignal(&decision.Rules, config.SignalTypeProjection, "verification_required") {
			t.Fatalf("expected %s to negate projection(%q) so verified overlays stay explicit siblings", routeName, "verification_required")
		}
	}
}

func TestMaintainedBalanceWarningBudgetStaysBelowCeiling(t *testing.T) {
	yamlPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "balance.yaml")
	cfg := mustLoadMaintainedBalanceRouterConfig(t, yamlPath)

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
	if len(prog.ProjectionPartitions) < 2 {
		t.Fatalf("expected maintained balance DSL to declare at least 2 projection partitions, got %d", len(prog.ProjectionPartitions))
	}
	assertMaintainedBalanceDSLDiagnostics(t, prog)
	return prog
}

func assertMaintainedBalanceDomainPartition(t *testing.T, groups []config.ProjectionPartition) {
	t.Helper()

	var domainPartition *config.ProjectionPartition
	for i := range groups {
		if groups[i].Name == "balance_domain_partition" {
			domainPartition = &groups[i]
			break
		}
	}
	if domainPartition == nil {
		t.Fatal("expected deploy/recipes/balance.yaml to define balance_domain_partition")
	}
	gotMembers := append([]string(nil), domainPartition.Members...)
	wantMembers := config.SupportedRoutingDomainNames()
	slices.Sort(gotMembers)
	slices.Sort(wantMembers)
	if !slices.Equal(gotMembers, wantMembers) {
		t.Fatalf("balance_domain_partition members mismatch\nwant: %v\ngot:  %v", wantMembers, gotMembers)
	}
	if domainPartition.Default != "other" {
		t.Fatalf("expected balance_domain_partition default to be %q, got %q", "other", domainPartition.Default)
	}
}

func assertMaintainedBalanceIntentPartition(t *testing.T, groups []config.ProjectionPartition) {
	t.Helper()

	var intentPartition *config.ProjectionPartition
	for i := range groups {
		if groups[i].Name == "balance_intent_partition" {
			intentPartition = &groups[i]
			break
		}
	}
	if intentPartition == nil {
		t.Fatal("expected deploy/recipes/balance.yaml to define balance_intent_partition")
	}

	wantMembers := []string{
		"agentic_workflows",
		"architecture_design",
		"business_analysis",
		"code_general",
		"complex_stem",
		"creative_tasks",
		"fast_qa_en",
		"fast_qa_zh",
		"general_chat_fallback",
		"health_guidance",
		"history_explainer",
		"premium_legal_analysis",
		"psychology_support",
		"reasoning_general_en",
		"reasoning_general_zh",
		"research_synthesis",
	}
	gotMembers := append([]string(nil), intentPartition.Members...)
	slices.Sort(gotMembers)
	slices.Sort(wantMembers)
	if !slices.Equal(gotMembers, wantMembers) {
		t.Fatalf("balance_intent_partition members mismatch\nwant: %v\ngot:  %v", wantMembers, gotMembers)
	}
	if intentPartition.Default != "general_chat_fallback" {
		t.Fatalf("expected balance_intent_partition default to be %q, got %q", "general_chat_fallback", intentPartition.Default)
	}
}

func assertMaintainedBalanceDecisionTiers(t *testing.T, decisions []config.Decision) {
	t.Helper()

	for _, decision := range decisions {
		if decision.Tier <= 0 {
			t.Fatalf("expected decision %q to define a positive tier", decision.Name)
		}
	}
}

func assertMaintainedBalanceProjectionBands(t *testing.T, projections config.Projections) {
	t.Helper()

	scoreNames := make(map[string]bool, len(projections.Scores))
	for _, score := range projections.Scores {
		scoreNames[score.Name] = true
	}
	for _, name := range []string{
		"difficulty_score",
		"verification_pressure",
		"feedback_correction_pressure",
		"feedback_clarification_pressure",
	} {
		if !scoreNames[name] {
			t.Fatalf("expected balance projections to include %q, got %v", name, scoreNames)
		}
	}

	outputNames := make(map[string]bool)
	for _, mapping := range projections.Mappings {
		for _, output := range mapping.Outputs {
			outputNames[output.Name] = true
		}
	}
	for _, name := range []string{
		"balance_simple",
		"balance_medium",
		"balance_complex",
		"balance_reasoning",
		"verification_required",
		"feedback_correction_verified",
		"feedback_clarification_overlay",
	} {
		if !outputNames[name] {
			t.Fatalf("expected balance projections to emit %q, got %v", name, outputNames)
		}
	}
}

func assertMaintainedBalanceRoute(t *testing.T, decisions []config.Decision, name string) {
	t.Helper()

	for _, decision := range decisions {
		if decision.Name == name {
			return
		}
	}
	t.Fatalf("expected deploy/recipes/balance.yaml to include %s route", name)
}

func assertMaintainedBalanceDSLMarkers(t *testing.T, dslPath, dslText string) {
	t.Helper()
	if !strings.Contains(dslText, "PROJECTION partition balance_intent_partition") {
		t.Fatalf("%s must define the learned intent projection partition", dslPath)
	}
	if !strings.Contains(dslText, "PROJECTION score difficulty_score") {
		t.Fatalf("%s must define the derived difficulty score", dslPath)
	}
	if !strings.Contains(dslText, "PROJECTION mapping verification_band") {
		t.Fatalf("%s must define the verification mapping", dslPath)
	}
	if !strings.Contains(dslText, "PROJECTION mapping feedback_correction_band") {
		t.Fatalf("%s must define the feedback correction mapping", dslPath)
	}
	if !strings.Contains(dslText, "PROJECTION mapping feedback_clarification_band") {
		t.Fatalf("%s must define the feedback clarification mapping", dslPath)
	}
	if !strings.Contains(dslText, "SIGNAL reask likely_dissatisfied") {
		t.Fatalf("%s must define the maintained reask helper for clarification routing", dslPath)
	}
	if !strings.Contains(dslText, "ROUTE formal_math_proof") {
		t.Fatalf("%s must include the narrow formal_math_proof route", dslPath)
	}
	if !strings.Contains(dslText, "ROUTE reasoning_deep") {
		t.Fatalf("%s must include the merged reasoning_deep route", dslPath)
	}
	if !strings.Contains(dslText, "ROUTE complex_specialist") {
		t.Fatalf("%s must include the merged complex_specialist route", dslPath)
	}
	if !strings.Contains(dslText, "ROUTE fast_qa") {
		t.Fatalf("%s must include the merged fast_qa route", dslPath)
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
	t.Fatalf("expected deploy/recipes/balance.yaml to include %s route", name)
	return nil
}

func ruleTreeContainsNegatedSignal(node *config.RuleCombination, signalType string, name string) bool {
	return ruleTreeContainsSignal(node, false, signalType, name)
}

func TestMaintainedSessionStateRecipeAssetsStayInSync(t *testing.T) {
	dslPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "session-state.dsl")
	yamlPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "session-state.yaml")

	dslData, err := os.ReadFile(dslPath)
	if err != nil {
		t.Fatalf("failed to read session-state.dsl: %v", err)
	}
	want, errs := Compile(string(dslData))
	if len(errs) > 0 {
		t.Fatalf("Compile errors for session-state.dsl: %v", errs)
	}

	got, err := config.Parse(yamlPath)
	if err != nil {
		t.Fatalf("failed to parse session-state.yaml: %v", err)
	}

	if !reflect.DeepEqual(got.SessionStates, want.SessionStates) {
		t.Fatalf("session-state DSL/YAML assets diverged\nwant: %+v\ngot:  %+v", want.SessionStates, got.SessionStates)
	}
}

func ruleTreeContainsSignal(node *config.RuleCombination, negated bool, signalType string, name string) bool {
	if node == nil {
		return false
	}
	if node.Type != "" {
		return negated && node.Type == signalType && node.Name == name
	}

	childNegated := negated
	if node.Operator == "NOT" {
		childNegated = !negated
	}
	for i := range node.Conditions {
		if ruleTreeContainsSignal(&node.Conditions[i], childNegated, signalType, name) {
			return true
		}
	}
	return false
}
