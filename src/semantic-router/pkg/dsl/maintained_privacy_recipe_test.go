package dsl

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMaintainedPrivacyRecipeRoutingAssetsStayInSync(t *testing.T) {
	dslPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "privacy", "privacy-router.dsl")
	yamlPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "privacy", "privacy-router.yaml")

	dslData, err := os.ReadFile(dslPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", dslPath, err)
	}

	prog, errs := Parse(string(dslData))
	if len(errs) > 0 {
		t.Fatalf("Parse errors: %v", errs)
	}

	want := mustCompileMaintainedPrivacyRoutingYAML(t, prog)
	got := mustEmitMaintainedPrivacyRoutingYAML(t, yamlPath)
	if want != got {
		t.Fatalf("maintained privacy DSL/YAML examples diverged\nwant:\n%s\ngot:\n%s", want, got)
	}
}

func TestMaintainedPrivacyRecipePolicyBandsAndProviders(t *testing.T) {
	yamlPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "privacy", "privacy-router.yaml")
	cfg := mustLoadMaintainedPrivacyRouterConfig(t, yamlPath)

	if len(cfg.PreferenceRules) != 0 {
		t.Fatalf("expected privacy recipe to avoid preference signals, got %d", len(cfg.PreferenceRules))
	}

	assertMaintainedPrivacyRoute(t, cfg.Decisions, "local_security_containment")
	assertMaintainedPrivacyRoute(t, cfg.Decisions, "local_privacy_policy")
	assertMaintainedPrivacyRoute(t, cfg.Decisions, "cloud_frontier_reasoning")
	assertMaintainedPrivacyRoute(t, cfg.Decisions, "local_standard")

	assertMaintainedPrivacyProjectionOutputs(t, cfg.Projections)
	assertMaintainedPrivacyReplayCoverage(t, cfg.Decisions)
	assertMaintainedPrivacySecurityRoute(t, cfg.Decisions)
	assertMaintainedPrivacyProviders(t, cfg)
}

func TestMaintainedPrivacyRecipeWarningBudgetStaysBelowCeiling(t *testing.T) {
	yamlPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "privacy", "privacy-router.yaml")
	cfg := mustLoadMaintainedPrivacyRouterConfig(t, yamlPath)

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
			t.Fatalf("unexpected maintained privacy diagnostic: %s", diag.String())
		}
		if diag.Level == DiagWarning {
			warnings++
		}
	}

	const maxWarnings = 0
	if warnings > maxWarnings {
		t.Fatalf("expected maintained privacy warning count <= %d, got %d", maxWarnings, warnings)
	}
}

func mustLoadMaintainedPrivacyRouterConfig(t *testing.T, yamlPath string) *config.RouterConfig {
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

func assertMaintainedPrivacyRoute(t *testing.T, decisions []config.Decision, name string) {
	t.Helper()

	for _, decision := range decisions {
		if decision.Name == name {
			return
		}
	}
	t.Fatalf("expected deploy/recipes/privacy/privacy-router.yaml to include %s route", name)
}

func assertMaintainedPrivacyProjectionOutputs(t *testing.T, projections config.Projections) {
	t.Helper()

	scoreNames := make(map[string]bool, len(projections.Scores))
	for _, score := range projections.Scores {
		scoreNames[score.Name] = true
	}
	for _, name := range []string{"security_risk_score", "privacy_risk_score", "reasoning_pressure"} {
		if !scoreNames[name] {
			t.Fatalf("expected privacy recipe to include projection score %q, got %v", name, scoreNames)
		}
	}

	outputNames := make(map[string]bool)
	for _, mapping := range projections.Mappings {
		for _, output := range mapping.Outputs {
			outputNames[output.Name] = true
		}
	}
	for _, name := range []string{
		"policy_security_standard",
		"policy_security_local_only",
		"policy_privacy_cloud_allowed",
		"policy_privacy_local_only",
		"policy_local_reasoning",
		"policy_frontier_reasoning",
	} {
		if !outputNames[name] {
			t.Fatalf("expected privacy recipe to emit %q, got %v", name, outputNames)
		}
	}
}

func assertMaintainedPrivacyReplayCoverage(t *testing.T, decisions []config.Decision) {
	t.Helper()

	for _, decision := range decisions {
		replayCfg := decision.GetRouterReplayConfig()
		if replayCfg == nil || !replayCfg.Enabled {
			t.Fatalf("expected %s to enable router_replay for audit", decision.Name)
		}
		if replayCfg.CaptureRequestBody || replayCfg.CaptureResponseBody {
			t.Fatalf("expected %s replay capture to avoid storing request/response bodies", decision.Name)
		}
	}
}

func assertMaintainedPrivacySecurityRoute(t *testing.T, decisions []config.Decision) {
	t.Helper()

	for _, decision := range decisions {
		if decision.Name != "local_security_containment" {
			continue
		}
		if !ruleTreeContainsPositiveSignal(&decision.Rules, config.SignalTypeProjection, "policy_security_local_only") {
			t.Fatal("expected local_security_containment to depend on policy_security_local_only")
		}
		return
	}
	t.Fatal("expected local_security_containment route")
}

func assertMaintainedPrivacyProviders(t *testing.T, cfg *config.RouterConfig) {
	t.Helper()

	if cfg.DefaultModel != "local/private-qwen" {
		t.Fatalf("expected default model to stay local, got %q", cfg.DefaultModel)
	}

	if len(cfg.VLLMEndpoints) != 2 {
		t.Fatalf("expected 2 normalized backend refs, got %d", len(cfg.VLLMEndpoints))
	}

	var endpointNames []string
	for _, ep := range cfg.VLLMEndpoints {
		endpointNames = append(endpointNames, ep.Name)
	}
	slices.Sort(endpointNames)
	wantNames := []string{
		"cloud/frontier-reasoning_openai-frontier",
		"local/private-qwen_local-vllm",
	}
	slices.Sort(wantNames)
	if !slices.Equal(endpointNames, wantNames) {
		t.Fatalf("provider endpoint names mismatch\nwant: %v\ngot:  %v", wantNames, endpointNames)
	}

	localEndpoint := mustFindEndpointByName(t, cfg.VLLMEndpoints, "local/private-qwen_local-vllm")
	if localEndpoint.Address != "vllm" || localEndpoint.Port != 8000 {
		t.Fatalf("expected local endpoint vllm:8000, got %s:%d", localEndpoint.Address, localEndpoint.Port)
	}
	if localEndpoint.ProviderProfileName != "" {
		t.Fatalf("expected local endpoint to avoid provider profile, got %q", localEndpoint.ProviderProfileName)
	}

	cloudEndpoint := mustFindEndpointByName(t, cfg.VLLMEndpoints, "cloud/frontier-reasoning_openai-frontier")
	if cloudEndpoint.ProviderProfileName == "" {
		t.Fatal("expected cloud endpoint to normalize through a provider profile")
	}
	profile, ok := cfg.ProviderProfiles[cloudEndpoint.ProviderProfileName]
	if !ok {
		t.Fatalf("missing provider profile %q", cloudEndpoint.ProviderProfileName)
	}
	if profile.Type != "openai" || !strings.Contains(profile.BaseURL, "api.openai.com/v1") {
		t.Fatalf("expected openai provider profile, got %#v", profile)
	}
}

func mustFindEndpointByName(t *testing.T, endpoints []config.VLLMEndpoint, name string) config.VLLMEndpoint {
	t.Helper()

	for _, endpoint := range endpoints {
		if endpoint.Name == name {
			return endpoint
		}
	}
	t.Fatalf("expected endpoint %q", name)
	return config.VLLMEndpoint{}
}

func ruleTreeContainsPositiveSignal(node *config.RuleCombination, signalType string, name string) bool {
	if node == nil {
		return false
	}
	if node.Type != "" {
		return node.Type == signalType && node.Name == name
	}
	for i := range node.Conditions {
		if ruleTreeContainsPositiveSignal(&node.Conditions[i], signalType, name) {
			return true
		}
	}
	return false
}

func mustCompileMaintainedPrivacyRoutingYAML(t *testing.T, prog *Program) string {
	t.Helper()

	compiledCfg, compileErrs := CompileAST(prog)
	if len(compileErrs) > 0 {
		t.Fatalf("CompileAST errors: %v", compileErrs)
	}
	rendered, err := EmitRoutingYAMLFromConfig(compiledCfg)
	if err != nil {
		t.Fatalf("EmitRoutingYAMLFromConfig error: %v", err)
	}
	return string(rendered)
}

func mustEmitMaintainedPrivacyRoutingYAML(t *testing.T, yamlPath string) string {
	t.Helper()

	parsedCfg := mustLoadMaintainedPrivacyRouterConfig(t, yamlPath)
	rendered, err := EmitRoutingYAMLFromConfig(parsedCfg)
	if err != nil {
		t.Fatalf("EmitRoutingYAMLFromConfig error: %v", err)
	}
	return string(rendered)
}
