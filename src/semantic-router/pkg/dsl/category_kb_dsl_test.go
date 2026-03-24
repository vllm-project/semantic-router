package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestParseCategoryKBSignal(t *testing.T) {
	input := `
SIGNAL category_kb privacy_classifier {
  kb_dir: "knowledge_bases/"
  taxonomy_path: "knowledge_bases/taxonomy.json"
  threshold: 0.55
  security_threshold: 0.7
}
`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("Parse errors: %v", errs)
	}

	if len(prog.Signals) != 1 {
		t.Fatalf("expected 1 signal, got %d", len(prog.Signals))
	}

	sig := prog.Signals[0]
	if sig.SignalType != "category_kb" {
		t.Errorf("signal type = %q, want category_kb", sig.SignalType)
	}
	if sig.Name != "privacy_classifier" {
		t.Errorf("signal name = %q, want privacy_classifier", sig.Name)
	}

	if v, ok := getStringField(sig.Fields, "kb_dir"); !ok || v != "knowledge_bases/" {
		t.Errorf("kb_dir = %q, ok=%v", v, ok)
	}
	if v, ok := getStringField(sig.Fields, "taxonomy_path"); !ok || v != "knowledge_bases/taxonomy.json" {
		t.Errorf("taxonomy_path = %q, ok=%v", v, ok)
	}
	if v, ok := getFloat32Field(sig.Fields, "threshold"); !ok || v != 0.55 {
		t.Errorf("threshold = %v, ok=%v", v, ok)
	}
	if v, ok := getFloat32Field(sig.Fields, "security_threshold"); !ok || v != 0.7 {
		t.Errorf("security_threshold = %v, ok=%v", v, ok)
	}
}

func TestCompileCategoryKBSignal(t *testing.T) {
	input := `
SIGNAL category_kb privacy_classifier {
  kb_dir: "knowledge_bases/"
  taxonomy_path: "knowledge_bases/taxonomy.json"
  threshold: 0.55
  security_threshold: 0.7
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	if len(cfg.CategoryKBRules) != 1 {
		t.Fatalf("expected 1 CategoryKBRule, got %d", len(cfg.CategoryKBRules))
	}

	rule := cfg.CategoryKBRules[0]
	if rule.Name != "privacy_classifier" {
		t.Errorf("Name = %q, want privacy_classifier", rule.Name)
	}
	if rule.KBDir != "knowledge_bases/" {
		t.Errorf("KBDir = %q", rule.KBDir)
	}
	if rule.TaxonomyPath != "knowledge_bases/taxonomy.json" {
		t.Errorf("TaxonomyPath = %q", rule.TaxonomyPath)
	}
	if rule.Threshold != 0.55 {
		t.Errorf("Threshold = %v", rule.Threshold)
	}
	if rule.SecurityThreshold != 0.7 {
		t.Errorf("SecurityThreshold = %v", rule.SecurityThreshold)
	}
}

func TestDecompileCategoryKBSignal(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.CategoryKBRules = []config.CategoryKBRule{
		{
			Name:              "privacy_classifier",
			KBDir:             "knowledge_bases/",
			TaxonomyPath:      "knowledge_bases/taxonomy.json",
			Threshold:         0.55,
			SecurityThreshold: 0.7,
		},
	}

	output, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting: %v", err)
	}

	if !strings.Contains(output, "SIGNAL category_kb privacy_classifier") {
		t.Errorf("missing signal declaration in output:\n%s", output)
	}
	if !strings.Contains(output, `kb_dir: "knowledge_bases/"`) {
		t.Errorf("missing kb_dir in output:\n%s", output)
	}
	if !strings.Contains(output, `taxonomy_path: "knowledge_bases/taxonomy.json"`) {
		t.Errorf("missing taxonomy_path in output:\n%s", output)
	}
	if !strings.Contains(output, "threshold: 0.55") {
		t.Errorf("missing threshold in output:\n%s", output)
	}
	if !strings.Contains(output, "security_threshold: 0.7") {
		t.Errorf("missing security_threshold in output:\n%s", output)
	}
}

func TestCategoryKBSignalRoundTrip(t *testing.T) {
	input := `
SIGNAL category_kb privacy_classifier {
  kb_dir: "knowledge_bases/"
  taxonomy_path: "knowledge_bases/taxonomy.json"
  threshold: 0.55
  security_threshold: 0.7
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	output, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting: %v", err)
	}

	cfg2, errs2 := Compile(output)
	if len(errs2) > 0 {
		t.Fatalf("Re-compile errors: %v", errs2)
	}

	if len(cfg2.CategoryKBRules) != 1 {
		t.Fatalf("re-compiled CategoryKBRules count = %d, want 1", len(cfg2.CategoryKBRules))
	}
	rule := cfg2.CategoryKBRules[0]
	if rule.Name != "privacy_classifier" {
		t.Errorf("round-trip Name = %q", rule.Name)
	}
	if rule.KBDir != "knowledge_bases/" {
		t.Errorf("round-trip KBDir = %q", rule.KBDir)
	}
	if rule.TaxonomyPath != "knowledge_bases/taxonomy.json" {
		t.Errorf("round-trip TaxonomyPath = %q", rule.TaxonomyPath)
	}
	if rule.Threshold != 0.55 {
		t.Errorf("round-trip Threshold = %v", rule.Threshold)
	}
	if rule.SecurityThreshold != 0.7 {
		t.Errorf("round-trip SecurityThreshold = %v", rule.SecurityThreshold)
	}
}

func TestParseToolScope(t *testing.T) {
	input := `
ROUTE security_containment {
  PRIORITY 300
  TOOL_SCOPE "none"
  WHEN keyword("threat_detected")
  MODEL "local-guard"
}
`
	prog, errs := Parse(input)
	if len(errs) > 0 {
		t.Fatalf("Parse errors: %v", errs)
	}

	if len(prog.Routes) != 1 {
		t.Fatalf("expected 1 route, got %d", len(prog.Routes))
	}

	route := prog.Routes[0]
	if route.ToolScope != "none" {
		t.Errorf("ToolScope = %q, want none", route.ToolScope)
	}
	if route.Priority != 300 {
		t.Errorf("Priority = %d, want 300", route.Priority)
	}
}

func TestCompileToolScope(t *testing.T) {
	input := `
ROUTE security_containment {
  PRIORITY 300
  TOOL_SCOPE "none"
  MODEL "local-guard"
}

ROUTE privacy_policy {
  PRIORITY 250
  TOOL_SCOPE "local_only"
  WHEN keyword("sensitive")
  MODEL "local-model"
}

ROUTE frontier_reasoning {
  PRIORITY 200
  TOOL_SCOPE "standard"
  MODEL "frontier-model"
}

ROUTE local_standard {
  PRIORITY 100
  TOOL_SCOPE "full"
  MODEL "default-model"
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	if len(cfg.Decisions) != 4 {
		t.Fatalf("expected 4 decisions, got %d", len(cfg.Decisions))
	}

	expected := map[string]string{
		"security_containment": "none",
		"privacy_policy":       "local_only",
		"frontier_reasoning":   "standard",
		"local_standard":       "full",
	}

	for _, dec := range cfg.Decisions {
		want, ok := expected[dec.Name]
		if !ok {
			t.Errorf("unexpected decision %q", dec.Name)
			continue
		}
		if dec.ToolScope != want {
			t.Errorf("decision %q: ToolScope = %q, want %q", dec.Name, dec.ToolScope, want)
		}
	}
}

func TestDecompileToolScope(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "security_containment",
					Priority:  300,
					ToolScope: "none",
					ModelRefs: []config.ModelRef{{Model: "local-guard"}},
				},
				{
					Name:      "privacy_policy",
					Priority:  250,
					ToolScope: "local_only",
					ModelRefs: []config.ModelRef{{Model: "local-model"}},
				},
			},
		},
	}

	output, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting: %v", err)
	}

	if !strings.Contains(output, `TOOL_SCOPE "none"`) {
		t.Errorf("missing TOOL_SCOPE none in:\n%s", output)
	}
	if !strings.Contains(output, `TOOL_SCOPE "local_only"`) {
		t.Errorf("missing TOOL_SCOPE local_only in:\n%s", output)
	}
}

func TestToolScopeRoundTrip(t *testing.T) {
	input := `
ROUTE test_route {
  PRIORITY 100
  TOOL_SCOPE "standard"
  MODEL "test-model"
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	if cfg.Decisions[0].ToolScope != "standard" {
		t.Fatalf("ToolScope = %q after compile", cfg.Decisions[0].ToolScope)
	}

	output, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting: %v", err)
	}

	cfg2, errs2 := Compile(output)
	if len(errs2) > 0 {
		t.Fatalf("Re-compile errors: %v", errs2)
	}
	if cfg2.Decisions[0].ToolScope != "standard" {
		t.Errorf("round-trip ToolScope = %q, want standard", cfg2.Decisions[0].ToolScope)
	}
}

func TestToolScopeEmpty(t *testing.T) {
	input := `
ROUTE no_scope {
  PRIORITY 100
  MODEL "test-model"
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	if cfg.Decisions[0].ToolScope != "" {
		t.Errorf("expected empty ToolScope for route without TOOL_SCOPE, got %q", cfg.Decisions[0].ToolScope)
	}

	output, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("Decompile: %v", err)
	}
	if strings.Contains(output, "TOOL_SCOPE") {
		t.Errorf("empty ToolScope should not be decompiled, got:\n%s", output)
	}
}

func TestCategoryKBSignalInWhenClause(t *testing.T) {
	input := `
SIGNAL category_kb privacy_classifier {
  kb_dir: "kbs/"
  threshold: 0.5
}

ROUTE privacy_route {
  PRIORITY 200
  WHEN category_kb("privacy_classifier")
  MODEL "local-model"
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	if len(cfg.CategoryKBRules) != 1 {
		t.Errorf("expected 1 CategoryKBRule, got %d", len(cfg.CategoryKBRules))
	}

	if len(cfg.Decisions) != 1 {
		t.Fatalf("expected 1 decision, got %d", len(cfg.Decisions))
	}
	dec := cfg.Decisions[0]
	// Single condition is wrapped in AND([leaf]) by the compiler
	if dec.Rules.Operator != "AND" {
		t.Fatalf("expected AND wrapper for single condition, got %q", dec.Rules.Operator)
	}
	if len(dec.Rules.Conditions) != 1 {
		t.Fatalf("expected 1 condition, got %d", len(dec.Rules.Conditions))
	}
	cond := dec.Rules.Conditions[0]
	if cond.Type != "category_kb" || cond.Name != "privacy_classifier" {
		t.Errorf("WHEN clause condition: type=%q name=%q, want category_kb/privacy_classifier",
			cond.Type, cond.Name)
	}
}

func TestCategoryKBWithANDClause(t *testing.T) {
	input := `
SIGNAL category_kb privacy_classifier {
  kb_dir: "kbs/"
  threshold: 0.5
}

SIGNAL keyword sensitive_kw {
  operator: "any"
  keywords: ["confidential", "internal"]
}

ROUTE privacy_route {
  PRIORITY 200
  WHEN category_kb("privacy_classifier") AND keyword("sensitive_kw")
  MODEL "local-model"
}
`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	dec := cfg.Decisions[0]
	if dec.Rules.Operator != "AND" {
		t.Errorf("expected AND operator, got %q", dec.Rules.Operator)
	}
	if len(dec.Rules.Conditions) != 2 {
		t.Fatalf("expected 2 conditions, got %d", len(dec.Rules.Conditions))
	}

	hasCategoryKB := false
	hasKeyword := false
	for _, cond := range dec.Rules.Conditions {
		if cond.Type == "category_kb" && cond.Name == "privacy_classifier" {
			hasCategoryKB = true
		}
		if cond.Type == "keyword" && cond.Name == "sensitive_kw" {
			hasKeyword = true
		}
	}
	if !hasCategoryKB {
		t.Error("missing category_kb condition")
	}
	if !hasKeyword {
		t.Error("missing keyword condition")
	}
}

func TestCompileAndDecompileFullPrivacyRecipe(t *testing.T) {
	input := `
SIGNAL category_kb privacy_classifier {
  kb_dir: "knowledge_bases/"
  taxonomy_path: "knowledge_bases/taxonomy.json"
  threshold: 0.55
  security_threshold: 0.7
}

SIGNAL keyword threat_detector {
  operator: "any"
  keywords: ["ignore previous", "DAN mode"]
}

PROJECTION score privacy_contrastive_score {
  method: "weighted_sum"
  inputs: [{ type: "category_kb", name: "__contrastive__", weight: 1.0 }]
}

PROJECTION mapping privacy_override_band {
  source: "privacy_contrastive_score"
  method: "threshold_bands"
  outputs: [{ name: "privacy_override_active", gt: 0.15 }]
}

ROUTE local_security_containment {
  PRIORITY 300
  TOOL_SCOPE "none"
  WHEN keyword("threat_detector") OR category_kb("privacy_classifier")
  MODEL "local-guard" (reasoning = false)
}

ROUTE local_privacy_policy {
  PRIORITY 250
  TOOL_SCOPE "local_only"
  WHEN category_kb("privacy_classifier")
  MODEL "local-reasoning" (reasoning = true, effort = "high")
}
`

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	// Verify signals
	if len(cfg.CategoryKBRules) != 1 {
		t.Errorf("CategoryKBRules = %d", len(cfg.CategoryKBRules))
	}
	if len(cfg.KeywordRules) != 1 {
		t.Errorf("KeywordRules = %d", len(cfg.KeywordRules))
	}

	// Verify projections
	if len(cfg.Projections.Scores) != 1 {
		t.Errorf("Projection scores = %d", len(cfg.Projections.Scores))
	}
	if len(cfg.Projections.Mappings) != 1 {
		t.Errorf("Projection mappings = %d", len(cfg.Projections.Mappings))
	}

	// Verify decisions
	if len(cfg.Decisions) != 2 {
		t.Fatalf("Decisions = %d", len(cfg.Decisions))
	}

	secDec := cfg.Decisions[0]
	if secDec.ToolScope != "none" {
		t.Errorf("security decision ToolScope = %q", secDec.ToolScope)
	}
	privDec := cfg.Decisions[1]
	if privDec.ToolScope != "local_only" {
		t.Errorf("privacy decision ToolScope = %q", privDec.ToolScope)
	}

	// Round-trip
	output, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting: %v", err)
	}

	cfg2, errs2 := Compile(output)
	if len(errs2) > 0 {
		t.Fatalf("Round-trip compile errors: %v", errs2)
	}

	if len(cfg2.CategoryKBRules) != 1 {
		t.Errorf("round-trip CategoryKBRules = %d", len(cfg2.CategoryKBRules))
	}
	if len(cfg2.Decisions) != 2 {
		t.Errorf("round-trip Decisions = %d", len(cfg2.Decisions))
	}
	if cfg2.Decisions[0].ToolScope != "none" {
		t.Errorf("round-trip security ToolScope = %q", cfg2.Decisions[0].ToolScope)
	}
}
