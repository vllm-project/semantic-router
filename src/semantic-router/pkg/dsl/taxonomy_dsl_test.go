package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func taxonomyDSLFixture() string {
	return `
SIGNAL kb privacy_policy {
  kb: "privacy_kb"
  target: { kind: "group", value: "privacy_policy" }
  match: "best"
}

PROJECTION score privacy_contrastive_score {
  method: "weighted_sum"
  inputs: [{ type: "kb_metric", kb: "privacy_kb", metric: "private_vs_public", weight: 1.0, value_source: "score" }]
}

ROUTE local_privacy_policy {
  PRIORITY 250
  WHEN kb("privacy_policy")
  MODEL "local-model"
  PLUGIN tools {
    enabled: true
    mode: "passthrough"
    semantic_selection: true
  }
}
`
}

func TestParseTaxonomyClassifierAndSignal(t *testing.T) {
	prog, errs := Parse(taxonomyDSLFixture())
	if len(errs) > 0 {
		t.Fatalf("Parse errors: %v", errs)
	}

	if len(prog.Signals) == 0 {
		t.Fatal("expected kb signal in AST")
	}
	sig := prog.Signals[0]
	if sig.SignalType != "kb" {
		t.Errorf("signal type = %q, want kb", sig.SignalType)
	}
	if sig.Name != "privacy_policy" {
		t.Errorf("signal name = %q, want privacy_policy", sig.Name)
	}
}

func TestCompileTaxonomyClassifierAndSignal(t *testing.T) {
	cfg, errs := Compile(taxonomyDSLFixture())
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	if len(cfg.KnowledgeBases) != 0 {
		t.Fatalf("expected routing DSL compile to omit global knowledge bases, got %d", len(cfg.KnowledgeBases))
	}
	assertCompiledKBRule(t, cfg)
	assertCompiledKBProjection(t, cfg)
	assertCompiledKBDecision(t, cfg)
}

func assertCompiledKBRule(t *testing.T, cfg *config.RouterConfig) {
	t.Helper()
	if len(cfg.KBRules) != 1 {
		t.Fatalf("expected 1 kb signal, got %d", len(cfg.KBRules))
	}
	rule := cfg.KBRules[0]
	if rule.KB != "privacy_kb" {
		t.Errorf("signal kb = %q", rule.KB)
	}
	if rule.Target.Kind != "group" || rule.Target.Value != "privacy_policy" {
		t.Errorf("target = %+v", rule.Target)
	}
	if rule.Match != "best" {
		t.Errorf("match = %q", rule.Match)
	}
}

func assertCompiledKBProjection(t *testing.T, cfg *config.RouterConfig) {
	t.Helper()
	if len(cfg.Projections.Scores) != 1 {
		t.Fatalf("expected 1 projection score, got %d", len(cfg.Projections.Scores))
	}
	input := cfg.Projections.Scores[0].Inputs[0]
	if input.Type != "kb_metric" {
		t.Errorf("projection input type = %q", input.Type)
	}
	if input.KB != "privacy_kb" {
		t.Errorf("projection input kb = %q", input.KB)
	}
	if input.Metric != "private_vs_public" {
		t.Errorf("projection input metric = %q", input.Metric)
	}
}

func assertCompiledKBDecision(t *testing.T, cfg *config.RouterConfig) {
	t.Helper()
	if len(cfg.Decisions) != 1 {
		t.Fatalf("expected 1 decision, got %d", len(cfg.Decisions))
	}
	cond := cfg.Decisions[0].Rules.Conditions[0]
	if cond.Type != "kb" || cond.Name != "privacy_policy" {
		t.Errorf("WHEN condition = %+v", cond)
	}
	toolsCfg := cfg.Decisions[0].GetToolsConfig()
	if toolsCfg == nil {
		t.Fatal("expected tools plugin on compiled decision")
	}
	if toolsCfg.EffectiveMode() != config.ToolsPluginModePassthrough {
		t.Errorf("tools mode = %q", toolsCfg.EffectiveMode())
	}
}

func TestDecompileTaxonomyClassifierAndSignal(t *testing.T) {
	cfg, errs := Compile(taxonomyDSLFixture())
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	output, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting: %v", err)
	}

	for _, needle := range []string{
		"SIGNAL kb privacy_policy",
		`kb: "privacy_kb"`,
		`target: { kind: "group", value: "privacy_policy" }`,
		`match: "best"`,
		`type: "kb_metric"`,
		`kb: "privacy_kb"`,
		`metric: "private_vs_public"`,
		`value_source: "score"`,
		`WHEN kb("privacy_policy")`,
		`PLUGIN tools`,
	} {
		if !strings.Contains(output, needle) {
			t.Fatalf("missing %q in decompiled DSL:\n%s", needle, output)
		}
	}
}

func TestTaxonomyDSLRoundTrip(t *testing.T) {
	cfg, errs := Compile(taxonomyDSLFixture())
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	cfg.KnowledgeBases = []config.KnowledgeBaseConfig{
		{
			Name: "privacy_kb",
			Source: config.KnowledgeBaseSource{
				Path:     "kb/privacy/",
				Manifest: "labels.json",
			},
			Threshold: 0.55,
			Groups: map[string][]string{
				"privacy_policy": {"proprietary_code"},
				"public":         {"generic_coding"},
			},
			Metrics: []config.KnowledgeBaseMetricConfig{
				{
					Name:          "private_vs_public",
					Type:          config.KBMetricTypeGroupMargin,
					PositiveGroup: "privacy_policy",
					NegativeGroup: "public",
				},
			},
		},
	}

	output, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting: %v", err)
	}

	cfg2, errs2 := Compile(output)
	if len(errs2) > 0 {
		t.Fatalf("Round-trip compile errors: %v", errs2)
	}

	if len(cfg2.KnowledgeBases) != 0 {
		t.Fatalf("round-trip routing DSL should not carry knowledge bases, got %d", len(cfg2.KnowledgeBases))
	}
	if len(cfg2.KBRules) != 1 {
		t.Fatalf("round-trip kb rules = %d", len(cfg2.KBRules))
	}
	if len(cfg2.Projections.Scores) != 1 {
		t.Fatalf("round-trip projection scores = %d", len(cfg2.Projections.Scores))
	}
	if len(cfg2.Decisions) != 1 {
		t.Fatalf("round-trip decisions = %d", len(cfg2.Decisions))
	}
}

func TestCompileToolsPlugin(t *testing.T) {
	input := `
ROUTE security_containment {
  PRIORITY 300
  PLUGIN tools { enabled: true, mode: "none", semantic_selection: false }
  MODEL "local-guard"
}

ROUTE privacy_policy {
  PRIORITY 250
  PLUGIN tools { enabled: true, mode: "passthrough", semantic_selection: true }
  WHEN keyword("sensitive_kw")
  MODEL "local-model"
}

ROUTE frontier_reasoning {
  PRIORITY 200
  PLUGIN tools { enabled: true, mode: "filtered", semantic_selection: true, allow_tools: ["search_web"], block_tools: ["exec_cmd"] }
  MODEL "frontier-model"
}

ROUTE local_standard {
  PRIORITY 100
  PLUGIN tools { enabled: true, mode: "passthrough", semantic_selection: false }
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
	assertToolsMode := func(index int, want string) {
		t.Helper()
		cfg := cfg.Decisions[index].GetToolsConfig()
		if cfg == nil {
			t.Fatalf("decision[%d] missing tools plugin", index)
		}
		if cfg.EffectiveMode() != want {
			t.Errorf("decision[%d] tools mode = %q, want %q", index, cfg.EffectiveMode(), want)
		}
	}
	assertToolsMode(0, config.ToolsPluginModeNone)
	assertToolsMode(1, config.ToolsPluginModePassthrough)
	assertToolsMode(2, config.ToolsPluginModeFiltered)
	assertToolsMode(3, config.ToolsPluginModePassthrough)
}
