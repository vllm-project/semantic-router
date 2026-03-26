package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestKnowledgeBaseYAMLRoundTrip(t *testing.T) {
	kb := KnowledgeBaseConfig{
		Name: "privacy_kb",
		Source: KnowledgeBaseSource{
			Path:     "kb/privacy/",
			Manifest: "labels.json",
		},
		Threshold: 0.55,
		LabelThresholds: map[string]float32{
			"prompt_injection": 0.7,
		},
		Groups: map[string][]string{
			"private": {"prompt_injection", "proprietary_code"},
			"public":  {"generic_coding"},
		},
		Metrics: []KnowledgeBaseMetricConfig{
			{
				Name:          "private_vs_public",
				Type:          KBMetricTypeGroupMargin,
				PositiveGroup: "private",
				NegativeGroup: "public",
			},
		},
	}

	data, err := yaml.Marshal(kb)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var got KnowledgeBaseConfig
	if err := yaml.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	if got.Name != kb.Name {
		t.Errorf("Name = %q, want %q", got.Name, kb.Name)
	}
	if got.Source.Manifest != kb.Source.Manifest {
		t.Errorf("Source.Manifest = %q, want %q", got.Source.Manifest, kb.Source.Manifest)
	}
	if got.Threshold != kb.Threshold {
		t.Errorf("Threshold = %v, want %v", got.Threshold, kb.Threshold)
	}
	if got.LabelThresholds["prompt_injection"] != 0.7 {
		t.Fatalf("LabelThresholds[prompt_injection] = %v, want 0.7", got.LabelThresholds["prompt_injection"])
	}
	if len(got.Groups["private"]) != 2 {
		t.Fatalf("Groups[private] = %v", got.Groups["private"])
	}
}

func TestKBConfigJSONUsesLowercaseFieldNames(t *testing.T) {
	payload := struct {
		KB     KnowledgeBaseConfig `json:"kb"`
		Signal KBSignalRule        `json:"signal"`
	}{
		KB: KnowledgeBaseConfig{
			Name: "privacy_kb",
			Source: KnowledgeBaseSource{
				Path:     "kb/privacy/",
				Manifest: "labels.json",
			},
			Threshold: 0.55,
		},
		Signal: KBSignalRule{
			Name: "privacy_policy",
			KB:   "privacy_kb",
			Target: KBSignalTarget{
				Kind:  KBTargetKindGroup,
				Value: "privacy_policy",
			},
			Match: KBMatchBest,
		},
	}

	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	jsonText := string(data)
	for _, unexpected := range []string{`"Path"`, `"Manifest"`, `"Kind"`, `"Value"`} {
		if strings.Contains(jsonText, unexpected) {
			t.Fatalf("expected lowercase JSON field names, found %s in %s", unexpected, jsonText)
		}
	}
	for _, expected := range []string{`"path"`, `"manifest"`, `"kind"`, `"value"`} {
		if !strings.Contains(jsonText, expected) {
			t.Fatalf("expected JSON output to contain %s, got %s", expected, jsonText)
		}
	}
}

func TestDefaultCanonicalModelCatalogUsesPrivacyKBDefaults(t *testing.T) {
	catalog := defaultCanonicalModelCatalog()
	if len(catalog.KBs) != 1 {
		t.Fatalf("expected 1 default kb, got %d", len(catalog.KBs))
	}

	kb := catalog.KBs[0]
	if kb.Name != "privacy_kb" {
		t.Fatalf("default kb name = %q, want privacy_kb", kb.Name)
	}
	if kb.Threshold != 0.55 {
		t.Fatalf("default kb threshold = %.2f, want 0.55", kb.Threshold)
	}
	if kb.LabelThresholds["prompt_injection"] != 0.7 {
		t.Fatalf("default kb label threshold = %.2f, want 0.7", kb.LabelThresholds["prompt_injection"])
	}
}

func TestKnowledgeBaseSourceResolvePathFallsBackToBundledAssets(t *testing.T) {
	source := KnowledgeBaseSource{
		Path:     "kb/privacy/",
		Manifest: "labels.json",
	}

	resolved := source.ResolvePath(t.TempDir())
	if !strings.HasSuffix(filepath.ToSlash(resolved), "config/kb/privacy") {
		t.Fatalf("ResolvePath() = %q, want bundled config/kb/privacy fallback", resolved)
	}
	if _, err := os.Stat(filepath.Join(resolved, source.Manifest)); err != nil {
		t.Fatalf("expected bundled manifest to exist: %v", err)
	}
}

func TestCanonicalSignalsKBRoundTrip(t *testing.T) {
	signals := CanonicalSignals{
		KB: []KBSignalRule{
			{
				Name: "privacy_policy",
				KB:   "privacy_kb",
				Target: KBSignalTarget{
					Kind:  KBTargetKindGroup,
					Value: "privacy_policy",
				},
				Match: KBMatchBest,
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

	if len(got.KB) != 1 {
		t.Fatalf("expected 1 kb rule, got %d", len(got.KB))
	}
	if got.KB[0].Target.Kind != KBTargetKindGroup {
		t.Errorf("Target.Kind = %q", got.KB[0].Target.Kind)
	}
}

func TestParseYAMLBytesRejectsLegacyCategoryKBSignal(t *testing.T) {
	legacyYAML := `
version: v0.3
listeners: []
providers:
  defaults: {}
  models: []
routing:
  signals:
    category_kb:
      - name: legacy_privacy_classifier
        kb_dir: knowledge_bases/
        taxonomy_path: knowledge_bases/taxonomy.json
        threshold: 0.5
  decisions: []
`

	_, err := ParseYAMLBytes([]byte(legacyYAML))
	if err == nil {
		t.Fatal("expected legacy category_kb signal to be rejected")
	}
	if !strings.Contains(err.Error(), "routing.signals.category_kb is no longer supported") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateKnowledgeBaseContractsRejectsUnknownKB(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				KBRules: []KBSignalRule{
					{
						Name: "privacy_policy",
						KB:   "missing_kb",
						Target: KBSignalTarget{
							Kind:  KBTargetKindGroup,
							Value: "privacy_policy",
						},
					},
				},
			},
		},
	}

	err := validateKnowledgeBaseContracts(cfg)
	if err == nil {
		t.Fatal("expected unknown kb to fail validation")
	}
	if !strings.Contains(err.Error(), "global.model_catalog.kbs") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateKnowledgeBaseContractsResolvesManifestBindings(t *testing.T) {
	root := writeKnowledgeBaseFixture(t)

	cfg := &RouterConfig{
		KnowledgeBases: []KnowledgeBaseConfig{
			{
				Name: "privacy_kb",
				Source: KnowledgeBaseSource{
					Path: root,
				},
				Threshold: 0.5,
				Groups: map[string][]string{
					"privacy_policy": {"proprietary_code"},
					"local_standard": {"generic_coding"},
				},
				Metrics: []KnowledgeBaseMetricConfig{
					{
						Name:          "private_vs_public",
						Type:          KBMetricTypeGroupMargin,
						PositiveGroup: "privacy_policy",
						NegativeGroup: "local_standard",
					},
				},
			},
		},
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				KBRules: []KBSignalRule{
					{
						Name: "privacy_policy",
						KB:   "privacy_kb",
						Target: KBSignalTarget{
							Kind:  KBTargetKindGroup,
							Value: "privacy_policy",
						},
						Match: KBMatchBest,
					},
					{
						Name: "proprietary_code",
						KB:   "privacy_kb",
						Target: KBSignalTarget{
							Kind:  KBTargetKindLabel,
							Value: "proprietary_code",
						},
						Match: KBMatchThreshold,
					},
				},
			},
		},
	}

	if err := validateKnowledgeBaseContracts(cfg); err != nil {
		t.Fatalf("validateKnowledgeBaseContracts returned error: %v", err)
	}
}

func TestValidateKnowledgeBaseContractsRejectsUnknownTargetValue(t *testing.T) {
	root := writeKnowledgeBaseFixture(t)

	cfg := &RouterConfig{
		KnowledgeBases: []KnowledgeBaseConfig{
			{
				Name: "privacy_kb",
				Source: KnowledgeBaseSource{
					Path: root,
				},
				Threshold: 0.5,
				Groups: map[string][]string{
					"privacy_policy": {"proprietary_code"},
				},
			},
		},
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				KBRules: []KBSignalRule{
					{
						Name: "unknown_group",
						KB:   "privacy_kb",
						Target: KBSignalTarget{
							Kind:  KBTargetKindGroup,
							Value: "not_declared",
						},
					},
				},
			},
		},
	}

	err := validateKnowledgeBaseContracts(cfg)
	if err == nil {
		t.Fatal("expected unknown kb target value to fail validation")
	}
	if !strings.Contains(err.Error(), "not a declared group") && !strings.Contains(err.Error(), "is not a declared group") {
		t.Fatalf("unexpected error: %v", err)
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
}

func TestDecisionToolScopeOmitsWhenEmpty(t *testing.T) {
	dec := Decision{Name: "test", Priority: 100}
	data, err := yaml.Marshal(dec)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	raw := string(data)
	if strings.Contains(raw, "tool_scope") {
		t.Errorf("expected tool_scope to be omitted in YAML, got:\n%s", raw)
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

func makeRouterConfigWithToolScope(name, scope string) *RouterConfig {
	return &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{{Name: name, ToolScope: scope}},
		},
	}
}

func TestValidateDecisionToolScopesRejectsUnknown(t *testing.T) {
	cfg := makeRouterConfigWithToolScope("test_decision", "bogus_scope")
	err := validateDecisionToolScopes(cfg)
	if err == nil {
		t.Fatal("expected error for unrecognized tool_scope, got nil")
	}
}

func TestValidateDecisionToolScopesAcceptsValid(t *testing.T) {
	for _, scope := range []string{"", "none", "local_only", "standard", "full"} {
		cfg := makeRouterConfigWithToolScope("d", scope)
		if err := validateDecisionToolScopes(cfg); err != nil {
			t.Errorf("tool_scope=%q should be valid, got: %v", scope, err)
		}
	}
}

func writeKnowledgeBaseFixture(t *testing.T) string {
	t.Helper()

	root := t.TempDir()
	manifest := `{
		"version": "1.0.0",
		"description": "Privacy KB",
		"labels": {
			"proprietary_code": {
				"description": "Private code",
				"exemplars": ["review this internal repository code"]
			},
			"generic_coding": {
				"description": "Public coding",
				"exemplars": ["write a python function"]
			}
		}
	}`

	path := filepath.Join(root, "labels.json")
	if err := os.WriteFile(path, []byte(manifest), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}

	return root
}
