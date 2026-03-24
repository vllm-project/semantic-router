package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestCategoryKBRuleYAMLRoundTrip(t *testing.T) {
	rule := CategoryKBRule{
		Name:              "privacy_classifier",
		KBDir:             "knowledge_bases/",
		TaxonomyPath:      "knowledge_bases/taxonomy.json",
		Threshold:         0.55,
		SecurityThreshold: 0.7,
	}

	data, err := yaml.Marshal(rule)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var got CategoryKBRule
	if err := yaml.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	if got.Name != rule.Name {
		t.Errorf("Name = %q, want %q", got.Name, rule.Name)
	}
	if got.KBDir != rule.KBDir {
		t.Errorf("KBDir = %q, want %q", got.KBDir, rule.KBDir)
	}
	if got.TaxonomyPath != rule.TaxonomyPath {
		t.Errorf("TaxonomyPath = %q, want %q", got.TaxonomyPath, rule.TaxonomyPath)
	}
	if got.Threshold != rule.Threshold {
		t.Errorf("Threshold = %v, want %v", got.Threshold, rule.Threshold)
	}
	if got.SecurityThreshold != rule.SecurityThreshold {
		t.Errorf("SecurityThreshold = %v, want %v", got.SecurityThreshold, rule.SecurityThreshold)
	}
}

func TestDecisionToolScopeYAMLRoundTrip(t *testing.T) {
	dec := Decision{
		Name:       "security_containment",
		Priority:   300,
		ToolScope:  "none",
		AllowTools: []string{"read_file"},
		BlockTools: []string{"exec_cmd"},
		ModelRefs:  []ModelRef{{Model: "local-guard"}},
	}

	data, err := yaml.Marshal(dec)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var got Decision
	if err := yaml.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	if got.ToolScope != "none" {
		t.Errorf("ToolScope = %q, want none", got.ToolScope)
	}
	if len(got.AllowTools) != 1 || got.AllowTools[0] != "read_file" {
		t.Errorf("AllowTools = %v, want [read_file]", got.AllowTools)
	}
	if len(got.BlockTools) != 1 || got.BlockTools[0] != "exec_cmd" {
		t.Errorf("BlockTools = %v, want [exec_cmd]", got.BlockTools)
	}
}

func TestDecisionToolScopeOmitsWhenEmpty(t *testing.T) {
	dec := Decision{
		Name:     "test",
		Priority: 100,
	}

	data, err := yaml.Marshal(dec)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	raw := string(data)
	if strings.Contains(raw, "tool_scope") {
		t.Errorf("expected tool_scope to be omitted in YAML, got:\n%s", raw)
	}
	if strings.Contains(raw, "allow_tools") {
		t.Errorf("expected allow_tools to be omitted, got:\n%s", raw)
	}
	if strings.Contains(raw, "block_tools") {
		t.Errorf("expected block_tools to be omitted, got:\n%s", raw)
	}
}

func TestCanonicalSignalsCategoryKBRoundTrip(t *testing.T) {
	signals := CanonicalSignals{
		CategoryKB: []CategoryKBRule{
			{
				Name:              "privacy_classifier",
				KBDir:             "kbs/",
				TaxonomyPath:      "taxonomy.json",
				Threshold:         0.5,
				SecurityThreshold: 0.8,
			},
		},
	}

	data, err := yaml.Marshal(signals)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var got CanonicalSignals
	if err := yaml.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	if len(got.CategoryKB) != 1 {
		t.Fatalf("expected 1 CategoryKB rule, got %d", len(got.CategoryKB))
	}
	if got.CategoryKB[0].Name != "privacy_classifier" {
		t.Errorf("Name = %q", got.CategoryKB[0].Name)
	}
	if got.CategoryKB[0].Threshold != 0.5 {
		t.Errorf("Threshold = %v", got.CategoryKB[0].Threshold)
	}
}

func TestToolScopeConstants(t *testing.T) {
	if ToolScopeNone != "none" {
		t.Errorf("ToolScopeNone = %q", ToolScopeNone)
	}
	if ToolScopeLocalOnly != "local_only" {
		t.Errorf("ToolScopeLocalOnly = %q", ToolScopeLocalOnly)
	}
	if ToolScopeStandard != "standard" {
		t.Errorf("ToolScopeStandard = %q", ToolScopeStandard)
	}
	if ToolScopeFull != "full" {
		t.Errorf("ToolScopeFull = %q", ToolScopeFull)
	}
}

