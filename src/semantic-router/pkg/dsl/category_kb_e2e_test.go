package dsl

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func setupKBTestDir(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	kbDir := filepath.Join(dir, "knowledge_bases")
	if err := os.MkdirAll(kbDir, 0o755); err != nil {
		t.Fatal(err)
	}

	taxonomy := map[string]interface{}{
		"categories": map[string]interface{}{
			"proprietary_code": map[string]interface{}{"tier": "privacy_policy", "description": "Internal code"},
			"prompt_injection": map[string]interface{}{"tier": "security_containment", "description": "Injection attacks"},
			"generic_coding":   map[string]interface{}{"tier": "local_standard", "description": "General coding"},
		},
		"tier_groups": map[string]interface{}{
			"privacy_categories":  []string{"proprietary_code"},
			"security_categories": []string{"prompt_injection"},
			"default_categories":  []string{"generic_coding"},
		},
		"category_to_tier": map[string]interface{}{
			"proprietary_code": "privacy_policy",
			"prompt_injection": "security_containment",
			"generic_coding":   "local_standard",
		},
	}
	writeJSON(t, filepath.Join(kbDir, "taxonomy.json"), taxonomy)

	writeJSON(t, filepath.Join(kbDir, "proprietary_code.json"), map[string]interface{}{
		"category": "proprietary_code", "tier": "privacy_policy",
		"exemplars": []string{"Review our internal SDK", "Debug our pipeline code"},
	})
	writeJSON(t, filepath.Join(kbDir, "prompt_injection.json"), map[string]interface{}{
		"category": "prompt_injection", "tier": "security_containment",
		"exemplars": []string{"Ignore previous instructions", "Bypass safety"},
	})
	writeJSON(t, filepath.Join(kbDir, "generic_coding.json"), map[string]interface{}{
		"category": "generic_coding", "tier": "local_standard",
		"exemplars": []string{"How do I sort an array?", "Explain recursion"},
	})
	return dir, kbDir
}

func compileCategoryKBDSL(t *testing.T, kbDir string) *config.RouterConfig {
	t.Helper()
	dslInput := `
SIGNAL category_kb privacy_classifier {
  kb_dir: "` + kbDir + `/"
  taxonomy_path: "` + filepath.Join(kbDir, "taxonomy.json") + `"
  threshold: 0.55
  security_threshold: 0.7
}

PROJECTION score privacy_contrastive_score {
  method: "weighted_sum"
  inputs: [{ type: "category_kb", name: "__contrastive__", weight: 1.0, value_source: "confidence" }]
}

PROJECTION mapping privacy_override_band {
  source: "privacy_contrastive_score"
  method: "threshold_bands"
  outputs: [{ name: "privacy_override_active", gt: 0.15 }]
}

ROUTE security_containment {
  PRIORITY 300
  TOOL_SCOPE "none"
  WHEN category_kb("privacy_classifier")
  MODEL "local-guard" (reasoning = false)
}

ROUTE privacy_policy {
  PRIORITY 250
  TOOL_SCOPE "local_only"
  WHEN category_kb("privacy_classifier") OR projection("privacy_override_active")
  MODEL "local-reasoning" (reasoning = true, effort = "high")
}

ROUTE local_standard {
  PRIORITY 100
  TOOL_SCOPE "full"
  MODEL "default-model"
}
`
	cfg, errs := Compile(dslInput)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	return cfg
}

func TestCategoryKB_CompileCategoryKBRule(t *testing.T) {
	_, kbDir := setupKBTestDir(t)
	cfg := compileCategoryKBDSL(t, kbDir)

	if len(cfg.CategoryKBRules) != 1 {
		t.Fatalf("expected 1 CategoryKBRule, got %d", len(cfg.CategoryKBRules))
	}
	rule := cfg.CategoryKBRules[0]
	if rule.Name != "privacy_classifier" {
		t.Errorf("rule.Name = %q", rule.Name)
	}
	if rule.KBDir != kbDir+"/" {
		t.Errorf("rule.KBDir = %q", rule.KBDir)
	}
	if rule.Threshold != 0.55 {
		t.Errorf("rule.Threshold = %v", rule.Threshold)
	}
	if rule.SecurityThreshold != 0.7 {
		t.Errorf("rule.SecurityThreshold = %v", rule.SecurityThreshold)
	}
}

func TestCategoryKB_CompileDecisionsAndToolScope(t *testing.T) {
	_, kbDir := setupKBTestDir(t)
	cfg := compileCategoryKBDSL(t, kbDir)

	if len(cfg.Decisions) != 3 {
		t.Fatalf("expected 3 decisions, got %d", len(cfg.Decisions))
	}
	scopeMap := map[string]string{}
	for _, dec := range cfg.Decisions {
		scopeMap[dec.Name] = dec.ToolScope
	}
	if scopeMap["security_containment"] != "none" {
		t.Errorf("security_containment ToolScope = %q", scopeMap["security_containment"])
	}
	if scopeMap["privacy_policy"] != "local_only" {
		t.Errorf("privacy_policy ToolScope = %q", scopeMap["privacy_policy"])
	}
	if scopeMap["local_standard"] != "full" {
		t.Errorf("local_standard ToolScope = %q", scopeMap["local_standard"])
	}
}

func TestCategoryKB_CompileProjections(t *testing.T) {
	_, kbDir := setupKBTestDir(t)
	cfg := compileCategoryKBDSL(t, kbDir)

	if len(cfg.Projections.Scores) != 1 {
		t.Errorf("expected 1 projection score, got %d", len(cfg.Projections.Scores))
	} else {
		score := cfg.Projections.Scores[0]
		if score.Name != "privacy_contrastive_score" {
			t.Errorf("score name = %q", score.Name)
		}
		if len(score.Inputs) != 1 || score.Inputs[0].Type != "category_kb" {
			t.Errorf("score input type = %v", score.Inputs)
		}
	}
	if len(cfg.Projections.Mappings) != 1 {
		t.Errorf("expected 1 projection mapping, got %d", len(cfg.Projections.Mappings))
	}
}

func TestCategoryKB_CompileWHENClauseSignalTypes(t *testing.T) {
	_, kbDir := setupKBTestDir(t)
	cfg := compileCategoryKBDSL(t, kbDir)

	secDec := cfg.Decisions[0]
	if !ruleTreeContainsSignalType(&secDec.Rules, "category_kb") {
		t.Error("security_containment WHEN clause should reference category_kb")
	}
	privDec := cfg.Decisions[1]
	if !ruleTreeContainsSignalType(&privDec.Rules, "category_kb") &&
		!ruleTreeContainsSignalType(&privDec.Rules, "projection") {
		t.Error("privacy_policy WHEN clause should reference category_kb or projection")
	}
}

func TestCategoryKB_RoutingYAMLRoundTrip(t *testing.T) {
	_, kbDir := setupKBTestDir(t)
	cfg := compileCategoryKBDSL(t, kbDir)

	routingYAML, err := EmitRoutingYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("EmitRoutingYAMLFromConfig: %v", err)
	}
	routingStr := string(routingYAML)

	for _, required := range []string{"category_kb:", "kb_dir:", "taxonomy_path:", "tool_scope:"} {
		if !strings.Contains(routingStr, required) {
			t.Errorf("routing YAML missing %s", required)
		}
	}

	var routingDoc struct {
		Routing config.CanonicalRouting `yaml:"routing"`
	}
	if err = yaml.Unmarshal(routingYAML, &routingDoc); err != nil {
		t.Fatalf("routing YAML unmarshal: %v", err)
	}
	if len(routingDoc.Routing.Signals.CategoryKB) != 1 {
		t.Fatalf("round-trip CategoryKB = %d", len(routingDoc.Routing.Signals.CategoryKB))
	}
	if routingDoc.Routing.Signals.CategoryKB[0].KBDir != kbDir+"/" {
		t.Errorf("round-trip KBDir = %q", routingDoc.Routing.Signals.CategoryKB[0].KBDir)
	}
}

func TestCategoryKB_UserYAML_IncludesCategoryKB(t *testing.T) {
	_, kbDir := setupKBTestDir(t)
	cfg := compileCategoryKBDSL(t, kbDir)

	userYAML, err := EmitUserYAML(cfg)
	if err != nil {
		t.Fatalf("EmitUserYAML: %v", err)
	}
	if !strings.Contains(string(userYAML), "category_kb:") {
		t.Error("user YAML missing category_kb signal section")
	}
}

func TestCategoryKB_KBFilesReadableFromConfig(t *testing.T) {
	_, kbDir := setupKBTestDir(t)
	cfg := compileCategoryKBDSL(t, kbDir)

	routingYAML, err := EmitRoutingYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("EmitRoutingYAMLFromConfig: %v", err)
	}
	var routingDoc struct {
		Routing config.CanonicalRouting `yaml:"routing"`
	}
	if err = yaml.Unmarshal(routingYAML, &routingDoc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	kbPath := routingDoc.Routing.Signals.CategoryKB[0].KBDir
	jsonCount := countAndValidateKBFiles(t, kbPath)
	if jsonCount != 3 {
		t.Errorf("expected 3 KB JSON files, found %d", jsonCount)
	}
}

func TestCategoryKB_TaxonomyLoadableFromConfig(t *testing.T) {
	_, kbDir := setupKBTestDir(t)
	cfg := compileCategoryKBDSL(t, kbDir)

	routingYAML, err := EmitRoutingYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("EmitRoutingYAMLFromConfig: %v", err)
	}
	var routingDoc struct {
		Routing config.CanonicalRouting `yaml:"routing"`
	}
	if err = yaml.Unmarshal(routingYAML, &routingDoc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	taxPath := routingDoc.Routing.Signals.CategoryKB[0].TaxonomyPath
	validateTaxonomyFile(t, taxPath, 3)
}

func TestCategoryKB_PrivacyRecipeDSL_ProducesValidRoutingYAML(t *testing.T) {
	dslPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "privacy", "privacy-router.dsl")
	dslData, err := os.ReadFile(dslPath)
	if err != nil {
		t.Skipf("privacy recipe DSL not found: %v", err)
	}

	cfg, errs := Compile(string(dslData))
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	if len(cfg.CategoryKBRules) != 1 {
		t.Fatalf("expected 1 CategoryKBRule, got %d", len(cfg.CategoryKBRules))
	}
	if cfg.CategoryKBRules[0].Name != "privacy_classifier" {
		t.Errorf("rule name = %q", cfg.CategoryKBRules[0].Name)
	}

	routingYAML, err := EmitRoutingYAMLFromConfig(cfg)
	if err != nil {
		t.Fatalf("EmitRoutingYAMLFromConfig: %v", err)
	}
	var routingDoc struct {
		Routing config.CanonicalRouting `yaml:"routing"`
	}
	if err = yaml.Unmarshal(routingYAML, &routingDoc); err != nil {
		t.Fatalf("routing YAML is not valid: %v", err)
	}
	if len(routingDoc.Routing.Signals.CategoryKB) != 1 {
		t.Fatal("routing YAML missing category_kb signal")
	}
	ckb := routingDoc.Routing.Signals.CategoryKB[0]
	if ckb.Name != "privacy_classifier" {
		t.Errorf("category_kb name = %q", ckb.Name)
	}
	if ckb.KBDir != "knowledge_bases/" {
		t.Errorf("category_kb kb_dir = %q", ckb.KBDir)
	}

	hasToolScope := false
	for _, dec := range routingDoc.Routing.Decisions {
		if dec.ToolScope != "" {
			hasToolScope = true
			break
		}
	}
	if !hasToolScope {
		t.Error("routing YAML decisions should include tool_scope")
	}

	recipeDir := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "privacy")
	verifyRecipeKBFiles(t, recipeDir, ckb)
}

func TestCategoryKB_UserYAML_IncludesCategoryKBUnderSignals(t *testing.T) {
	dslInput := `
SIGNAL category_kb test_classifier {
  kb_dir: "kbs/"
  taxonomy_path: "kbs/taxonomy.json"
  threshold: 0.3
}

ROUTE test_route {
  PRIORITY 100
  TOOL_SCOPE "standard"
  WHEN category_kb("test_classifier")
  MODEL "test-model"
}
`
	cfg, errs := Compile(dslInput)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	userYAML, err := EmitUserYAML(cfg)
	if err != nil {
		t.Fatalf("EmitUserYAML: %v", err)
	}

	var raw map[string]interface{}
	if err := yaml.Unmarshal(userYAML, &raw); err != nil {
		t.Fatalf("user YAML unmarshal: %v", err)
	}

	signals, ok := raw["signals"].(map[interface{}]interface{})
	if !ok {
		t.Fatal("user YAML missing signals section")
	}

	categoryKB, ok := signals["category_kb"]
	if !ok {
		t.Fatal("signals section missing category_kb")
	}
	kbList, ok := categoryKB.([]interface{})
	if !ok || len(kbList) != 1 {
		t.Fatalf("category_kb should have 1 entry, got %v", categoryKB)
	}

	entry, _ := kbList[0].(map[interface{}]interface{})
	if entry["name"] != "test_classifier" {
		t.Errorf("category_kb[0].name = %v", entry["name"])
	}
	if entry["kb_dir"] != "kbs/" {
		t.Errorf("category_kb[0].kb_dir = %v", entry["kb_dir"])
	}

	if _, topLevel := raw["category_kb"]; topLevel {
		t.Error("category_kb should not appear as top-level key; should be under signals")
	}
}

func TestCategoryKB_DecompileRoutingToAST_IncludesCategoryKB(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.CategoryKBRules = []config.CategoryKBRule{
		{
			Name:              "privacy_classifier",
			KBDir:             "knowledge_bases/",
			TaxonomyPath:      "knowledge_bases/taxonomy.json",
			Threshold:         0.30,
			SecurityThreshold: 0.25,
		},
	}
	cfg.Decisions = []config.Decision{
		{
			Name:     "test_route",
			Priority: 100,
			Rules: config.RuleCombination{
				Type: "category_kb",
				Name: "privacy_classifier",
			},
		},
	}

	prog := DecompileRoutingToAST(cfg)

	found := false
	for _, sig := range prog.Signals {
		if sig.SignalType == "category_kb" && sig.Name == "privacy_classifier" {
			found = true
			kbDir, ok := sig.Fields["kb_dir"]
			if !ok {
				t.Error("missing kb_dir field in AST signal")
			} else if kbDir.(StringValue).V != "knowledge_bases/" {
				t.Errorf("kb_dir = %q", kbDir.(StringValue).V)
			}
			break
		}
	}
	if !found {
		t.Error("DecompileRoutingToAST should include category_kb signal")
	}
}

func TestCLICompile_WithBase_ProducesCompleteConfig(t *testing.T) {
	dir := t.TempDir()

	dslContent := `
SIGNAL keyword test_kw {
  operator: "OR"
  keywords: ["private repo"]
  method: "bm25"
  bm25_threshold: 0.1
}

SIGNAL category_kb test_kb {
  kb_dir: "kbs/"
  taxonomy_path: "kbs/taxonomy.json"
  threshold: 0.3
}

MODEL local/test-model {
  context_window_size: 32000
  description: "Test model"
}

ROUTE test_route {
  PRIORITY 100
  TOOL_SCOPE "standard"
  WHEN keyword("test_kw")
  MODEL "local/test-model"
}
`
	dslPath := filepath.Join(dir, "test.dsl")
	if err := os.WriteFile(dslPath, []byte(dslContent), 0o644); err != nil {
		t.Fatal(err)
	}

	baseContent := `
version: v0.3

listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 600s

providers:
  defaults:
    default_model: local/test-model
  models:
    - name: local/test-model
      provider_model_id: test/model
      backend_refs:
        - name: test-backend
          endpoint: localhost:8000
`
	basePath := filepath.Join(dir, "providers.yaml")
	if err := os.WriteFile(basePath, []byte(baseContent), 0o644); err != nil {
		t.Fatal(err)
	}

	outPath := filepath.Join(dir, "config.yaml")
	if err := CLICompile(dslPath, outPath, "yaml", "", "", basePath); err != nil {
		t.Fatalf("CLICompile with --base failed: %v", err)
	}

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	output := string(data)

	for _, required := range []string{"version: v0.3", "listeners:", "providers:", "routing:", "category_kb:", "tool_scope:", "localhost:8000", "test_route"} {
		if !strings.Contains(output, required) {
			t.Errorf("missing %q in merged output", required)
		}
	}
}

func TestCLICompile_WithoutBase_ProducesRoutingOnly(t *testing.T) {
	dir := t.TempDir()

	dslContent := `
SIGNAL keyword test_kw {
  operator: "OR"
  keywords: ["private repo"]
  method: "bm25"
  bm25_threshold: 0.1
}

ROUTE test_route {
  PRIORITY 100
  WHEN keyword("test_kw")
  MODEL "local/test-model"
}
`
	dslPath := filepath.Join(dir, "test.dsl")
	if err := os.WriteFile(dslPath, []byte(dslContent), 0o644); err != nil {
		t.Fatal(err)
	}

	outPath := filepath.Join(dir, "routing.yaml")
	if err := CLICompile(dslPath, outPath, "yaml", "", "", ""); err != nil {
		t.Fatalf("CLICompile without --base failed: %v", err)
	}

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	output := string(data)

	if !strings.Contains(output, "routing:") {
		t.Error("missing routing section")
	}
	for _, absent := range []string{"version:", "listeners:", "providers:"} {
		if strings.Contains(output, absent) {
			t.Errorf("routing-only output should not have %s", absent)
		}
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func ruleTreeContainsSignalType(node *config.RuleCombination, signalType string) bool {
	if node == nil {
		return false
	}
	if node.Type == signalType {
		return true
	}
	for i := range node.Conditions {
		if ruleTreeContainsSignalType(&node.Conditions[i], signalType) {
			return true
		}
	}
	return false
}

func writeJSON(t *testing.T, path string, v interface{}) {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func countAndValidateKBFiles(t *testing.T, kbPath string) int {
	t.Helper()
	entries, err := os.ReadDir(kbPath)
	if err != nil {
		t.Fatalf("Cannot read KB dir at %q: %v", kbPath, err)
	}
	count := 0
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") || entry.Name() == "taxonomy.json" {
			continue
		}
		count++
		data, readErr := os.ReadFile(filepath.Join(kbPath, entry.Name()))
		if readErr != nil {
			t.Errorf("Cannot read KB file %s: %v", entry.Name(), readErr)
			continue
		}
		var kb struct {
			Exemplars []string `json:"exemplars"`
		}
		if err = json.Unmarshal(data, &kb); err != nil {
			t.Errorf("Cannot parse KB file %s: %v", entry.Name(), err)
			continue
		}
		if len(kb.Exemplars) == 0 {
			t.Errorf("KB file %s has no exemplars", entry.Name())
		}
	}
	return count
}

func validateTaxonomyFile(t *testing.T, taxPath string, expectedCategories int) {
	t.Helper()
	data, err := os.ReadFile(taxPath)
	if err != nil {
		t.Fatalf("Cannot read taxonomy at %q: %v", taxPath, err)
	}
	var taxMap map[string]interface{}
	if err := json.Unmarshal(data, &taxMap); err != nil {
		t.Fatalf("Cannot parse taxonomy: %v", err)
	}
	cats, ok := taxMap["categories"].(map[string]interface{})
	if !ok || len(cats) != expectedCategories {
		t.Errorf("taxonomy categories count = %d, want %d", len(cats), expectedCategories)
	}
}

func verifyRecipeKBFiles(t *testing.T, recipeDir string, ckb config.CategoryKBRule) {
	t.Helper()
	kbDirAbs := filepath.Join(recipeDir, ckb.KBDir)
	entries, err := os.ReadDir(kbDirAbs)
	if err != nil {
		t.Fatalf("KB directory not readable at %q: %v", kbDirAbs, err)
	}

	kbFileCount := 0
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".json") && entry.Name() != "taxonomy.json" {
			kbFileCount++
		}
	}
	if kbFileCount == 0 {
		t.Error("no KB JSON files found in recipe knowledge_bases/ directory")
	}
	t.Logf("Found %d category KB files in %s", kbFileCount, kbDirAbs)

	taxPath := filepath.Join(recipeDir, ckb.TaxonomyPath)
	taxData, err := os.ReadFile(taxPath)
	if err != nil {
		t.Fatalf("taxonomy not readable at %q: %v", taxPath, err)
	}
	var taxMap map[string]interface{}
	if err := json.Unmarshal(taxData, &taxMap); err != nil {
		t.Fatalf("taxonomy JSON invalid: %v", err)
	}
	cats, _ := taxMap["categories"].(map[string]interface{})
	tierGroups, _ := taxMap["tier_groups"].(map[string]interface{})
	if len(cats) == 0 {
		t.Error("taxonomy has no categories")
	}
	if len(tierGroups) == 0 {
		t.Error("taxonomy has no tier_groups")
	}
	t.Logf("Taxonomy: %d categories, %d tier groups", len(cats), len(tierGroups))

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") || entry.Name() == "taxonomy.json" {
			continue
		}
		catName := strings.TrimSuffix(entry.Name(), ".json")
		if _, ok := cats[catName]; !ok {
			t.Errorf("KB file %s has no matching category in taxonomy", entry.Name())
		}
	}
}
