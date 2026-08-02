package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPolicySignalsAndClassifierLeafRoundTrip(t *testing.T) {
	input := `
SIGNAL metadata "canary" {
  key: "cohort"
  predicate: { equals: "canary" }
}

SIGNAL classifier "risk" {
  type: "local"
  model_path: "models/risk"
  labels: ["SAFE", "RISKY"]
  use_cpu: true
}

ROUTE "policy-route" {
  PRIORITY 100
  WHEN metadata("canary") AND classifier(
    "risk",
    label: "RISKY",
    predicate: { gte: 0.8 },
    on_error: "match"
  )
  MODEL "model-a"
}`
	cfg := mustCompilePolicyDSL(t, input)
	assertPolicySignals(t, cfg)
	assertPolicyClassifierLeaf(t, cfg.Decisions[0].Rules.Conditions[1])

	source, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}
	assertPolicyDSLSource(t, source)
	roundTrip := mustCompilePolicyDSL(t, source)
	assertPolicyClassifierLeaf(
		t,
		roundTrip.Decisions[0].Rules.Conditions[1],
	)
}

func mustCompilePolicyDSL(
	t *testing.T,
	source string,
) *config.RouterConfig {
	t.Helper()
	cfg, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	return cfg
}

func assertPolicySignals(t *testing.T, cfg *config.RouterConfig) {
	t.Helper()
	if len(cfg.MetadataRules) != 1 ||
		cfg.MetadataRules[0].Key != "cohort" {
		t.Fatalf("metadata rules = %#v", cfg.MetadataRules)
	}
	if len(cfg.ClassifierRules) != 1 ||
		cfg.ClassifierRules[0].Labels[1] != "RISKY" {
		t.Fatalf("classifier rules = %#v", cfg.ClassifierRules)
	}
}

func assertPolicyClassifierLeaf(
	t *testing.T,
	classifierLeaf config.RuleNode,
) {
	t.Helper()
	if classifierLeaf.Label != "RISKY" ||
		classifierLeaf.Predicate == nil ||
		classifierLeaf.Predicate.GTE == nil ||
		*classifierLeaf.Predicate.GTE != 0.8 ||
		classifierLeaf.OnError != "match" {
		t.Fatalf("classifier leaf = %#v", classifierLeaf)
	}
}

func assertPolicyDSLSource(t *testing.T, source string) {
	t.Helper()
	for _, expected := range []string{
		`SIGNAL metadata canary`,
		`SIGNAL classifier risk`,
		`label: "RISKY"`,
		`predicate: { gte: 0.8 }`,
		`on_error: "match"`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("decompiled source missing %q:\n%s", expected, source)
		}
	}
}
