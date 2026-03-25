package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func taxonomyDSLFixture() string {
	return `
SIGNAL taxonomy privacy_policy {
  classifier: "privacy_classifier"
  bind: { kind: "tier", value: "privacy_policy" }
}

PROJECTION score privacy_contrastive_score {
  method: "weighted_sum"
  inputs: [{ type: "taxonomy_metric", classifier: "privacy_classifier", metric: "contrastive", weight: 1.0, value_source: "score" }]
}

ROUTE local_privacy_policy {
  PRIORITY 250
  TOOL_SCOPE "local_only"
  WHEN taxonomy("privacy_policy")
  MODEL "local-model"
}
`
}

func TestParseTaxonomyClassifierAndSignal(t *testing.T) {
	prog, errs := Parse(taxonomyDSLFixture())
	if len(errs) > 0 {
		t.Fatalf("Parse errors: %v", errs)
	}

	if len(prog.Signals) == 0 {
		t.Fatal("expected taxonomy signal in AST")
	}
	sig := prog.Signals[0]
	if sig.SignalType != "taxonomy" {
		t.Errorf("signal type = %q, want taxonomy", sig.SignalType)
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

	if len(cfg.TaxonomyClassifiers) != 0 {
		t.Fatalf("expected routing DSL compile to omit global taxonomy classifiers, got %d", len(cfg.TaxonomyClassifiers))
	}

	if len(cfg.TaxonomyRules) != 1 {
		t.Fatalf("expected 1 taxonomy signal, got %d", len(cfg.TaxonomyRules))
	}
	rule := cfg.TaxonomyRules[0]
	if rule.Classifier != "privacy_classifier" {
		t.Errorf("signal classifier = %q", rule.Classifier)
	}
	if rule.Bind.Kind != "tier" || rule.Bind.Value != "privacy_policy" {
		t.Errorf("bind = %+v", rule.Bind)
	}

	if len(cfg.Projections.Scores) != 1 {
		t.Fatalf("expected 1 projection score, got %d", len(cfg.Projections.Scores))
	}
	input := cfg.Projections.Scores[0].Inputs[0]
	if input.Type != "taxonomy_metric" {
		t.Errorf("projection input type = %q", input.Type)
	}
	if input.Classifier != "privacy_classifier" {
		t.Errorf("projection input classifier = %q", input.Classifier)
	}
	if input.Metric != "contrastive" {
		t.Errorf("projection input metric = %q", input.Metric)
	}

	if len(cfg.Decisions) != 1 {
		t.Fatalf("expected 1 decision, got %d", len(cfg.Decisions))
	}
	cond := cfg.Decisions[0].Rules.Conditions[0]
	if cond.Type != "taxonomy" || cond.Name != "privacy_policy" {
		t.Errorf("WHEN condition = %+v", cond)
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
		"SIGNAL taxonomy privacy_policy",
		`classifier: "privacy_classifier"`,
		`bind: { kind: "tier", value: "privacy_policy" }`,
		`type: "taxonomy_metric"`,
		`classifier: "privacy_classifier"`,
		`metric: "contrastive"`,
		`value_source: "score"`,
		`WHEN taxonomy("privacy_policy")`,
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
	cfg.TaxonomyClassifiers = []config.TaxonomyClassifierConfig{
		{
			Name: "privacy_classifier",
			Type: config.ClassifierTypeTaxonomy,
			Source: config.TaxonomyClassifierSource{
				Path:         "classifiers/privacy/",
				TaxonomyFile: "taxonomy.json",
			},
			Threshold:         0.55,
			SecurityThreshold: 0.7,
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

	if len(cfg2.TaxonomyClassifiers) != 0 {
		t.Fatalf("round-trip routing DSL should not carry classifiers, got %d", len(cfg2.TaxonomyClassifiers))
	}
	if len(cfg2.TaxonomyRules) != 1 {
		t.Fatalf("round-trip taxonomy rules = %d", len(cfg2.TaxonomyRules))
	}
	if len(cfg2.Projections.Scores) != 1 {
		t.Fatalf("round-trip projection scores = %d", len(cfg2.Projections.Scores))
	}
	if len(cfg2.Decisions) != 1 {
		t.Fatalf("round-trip decisions = %d", len(cfg2.Decisions))
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
  WHEN keyword("sensitive_kw")
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
	if cfg.Decisions[0].ToolScope != "none" {
		t.Errorf("ToolScope = %q, want none", cfg.Decisions[0].ToolScope)
	}
	if cfg.Decisions[1].ToolScope != "local_only" {
		t.Errorf("ToolScope = %q, want local_only", cfg.Decisions[1].ToolScope)
	}
	if cfg.Decisions[2].ToolScope != "standard" {
		t.Errorf("ToolScope = %q, want standard", cfg.Decisions[2].ToolScope)
	}
	if cfg.Decisions[3].ToolScope != "full" {
		t.Errorf("ToolScope = %q, want full", cfg.Decisions[3].ToolScope)
	}
}
