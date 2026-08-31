package dsl

import (
	"reflect"
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

ROUTE "policy-route" (on_unknown = "fail_request") {
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
	if cfg.Decisions[0].Rules.OnUnknown != "fail_request" {
		t.Fatalf("on_unknown = %q", cfg.Decisions[0].Rules.OnUnknown)
	}
	assertPolicyClassifierLeaf(t, cfg.Decisions[0].Rules.Conditions[1])

	source, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}
	assertPolicyDSLSource(t, source)
	roundTrip := mustCompilePolicyDSL(t, source)
	if roundTrip.Decisions[0].Rules.OnUnknown != "fail_request" {
		t.Fatalf("round-trip on_unknown = %q", roundTrip.Decisions[0].Rules.OnUnknown)
	}
	assertPolicyClassifierLeaf(
		t,
		roundTrip.Decisions[0].Rules.Conditions[1],
	)
}

func TestPolicySignalsRoundTripInsideIsolatedRecipes(t *testing.T) {
	input := `
MODEL "model-a" {}

RECIPE alpha {
  SIGNAL metadata "cohort" {
    key: "tenant"
    predicate: { equals: "alpha" }
  }
  SIGNAL classifier "risk" {
    type: "local"
    model_path: "models/risk"
    labels: ["SAFE", "RISKY"]
    use_cpu: true
  }
  ROUTE "policy-route" {
    PRIORITY 100
    WHEN metadata("cohort") AND classifier(
      "risk",
      label: "RISKY",
      predicate: { gte: 0.8 },
      on_error: "no_match"
    )
    MODEL "model-a"
  }
}

RECIPE beta {
  SIGNAL metadata "cohort" {
    key: "tenant"
    predicate: { equals: "beta" }
  }
  SIGNAL classifier "risk" {
    type: "local"
    model_path: "models/risk"
    labels: ["SAFE", "RISKY"]
    use_cpu: true
  }
  ROUTE "policy-route" {
    PRIORITY 100
    WHEN metadata("cohort")
    MODEL "model-a"
  }
}`
	cfg := mustCompilePolicyDSL(t, input)
	alpha, alphaOK := cfg.RecipeByName("alpha")
	beta, betaOK := cfg.RecipeByName("beta")
	if !alphaOK || !betaOK {
		t.Fatalf("recipe scopes were not compiled: %#v", cfg.Recipes)
	}
	if got := *alpha.Profile.Signals.MetadataRules[0].Predicate.Equals; got != "alpha" {
		t.Fatalf("alpha metadata predicate = %q", got)
	}
	if got := *beta.Profile.Signals.MetadataRules[0].Predicate.Equals; got != "beta" {
		t.Fatalf("beta metadata predicate = %q", got)
	}

	source, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile error: %v", err)
	}
	roundTrip := mustCompilePolicyDSL(t, source)
	if !reflect.DeepEqual(cfg.Recipes, roundTrip.Recipes) {
		t.Fatalf(
			"recipe policy changed after round trip:\n%#v\n%#v\n%s",
			cfg.Recipes,
			roundTrip.Recipes,
			source,
		)
	}
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
		`on_unknown = "fail_request"`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("decompiled source missing %q:\n%s", expected, source)
		}
	}
}
