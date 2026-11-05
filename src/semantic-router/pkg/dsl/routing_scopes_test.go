package dsl

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const scopedRoutingDSL = `
ROUTING { strategy: priority }

MODEL shared_model { context_window_size: 32768 }

ENTRYPOINT {
  model_names: ["router/alpha"]
  recipe: "alpha"
}

ENTRYPOINT {
  model_names: ["router/beta", "router/beta-v2"]
  recipe: "beta"
}

RECIPE alpha (description = "Alpha objective") {
  ROUTING { strategy: confidence }
  SIGNAL keyword shared_name { keywords: ["alpha"] }
  ROUTE alpha_route {
    PRIORITY 10
    WHEN keyword("shared_name")
    MODEL shared_model
  }
}

RECIPE beta (description = "Beta objective") {
  ROUTING { strategy: priority }
  SIGNAL keyword shared_name { keywords: ["beta"] }
  ROUTE beta_route {
    PRIORITY 20
    WHEN keyword("shared_name")
    MODEL shared_model
  }
}
`

func TestCompileAndDecompileRoutingScopes(t *testing.T) {
	cfg, errs := Compile(scopedRoutingDSL)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.Recipes) != 3 {
		t.Fatalf("recipe count = %d, want default plus two named recipes", len(cfg.Recipes))
	}
	if len(cfg.Entrypoints) != 2 {
		t.Fatalf("entrypoint count = %d, want 2", len(cfg.Entrypoints))
	}
	alpha, ok := cfg.RecipeByName("alpha")
	if !ok || alpha.Profile.Strategy != config.RoutingStrategyConfidence {
		t.Fatalf("alpha recipe was not isolated or strategy was lost: %+v", alpha)
	}

	text, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile: %v", err)
	}
	recompiled, errs := Compile(text)
	if len(errs) > 0 {
		t.Fatalf("recompile errors: %v\n%s", errs, text)
	}
	if !reflect.DeepEqual(cfg.Entrypoints, recompiled.Entrypoints) {
		t.Fatalf("entrypoints changed after round trip:\n%+v\n%+v", cfg.Entrypoints, recompiled.Entrypoints)
	}
	if !reflect.DeepEqual(cfg.Recipes, recompiled.Recipes) {
		t.Fatalf("recipes changed after round trip:\n%+v\n%+v", cfg.Recipes, recompiled.Recipes)
	}
}

func TestEntrypointAPISurvivesRoundTrip(t *testing.T) {
	source := `
MODEL shared_model {}
ENTRYPOINT { model_names: ["router/auto"] recipe: "alpha" }
ENTRYPOINT { api: "systemone" model_names: ["router/auto"] recipe: "native" }
RECIPE alpha { ROUTE alpha_route { MODEL shared_model } }
RECIPE native { ROUTE native_route { MODEL shared_model } }
`
	diagnostics, parseErrs := Validate(source)
	if len(parseErrs) > 0 {
		t.Fatalf("parse errors: %v", parseErrs)
	}
	for _, diagnostic := range diagnostics {
		if strings.Contains(diagnostic.Message, "is already mapped") {
			t.Fatalf("Chat and System One names were checked as one namespace: %s", diagnostic.Message)
		}
	}
	cfg, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.Entrypoints) != 2 || cfg.Entrypoints[0].API != "" || cfg.Entrypoints[1].API != config.SystemOneAPI {
		t.Fatalf("entrypoint APIs = %+v, want an omitted Chat api and systemone", cfg.Entrypoints)
	}

	text, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("decompile: %v", err)
	}
	if strings.Count(text, "api:") != 1 || !strings.Contains(text, `api: "systemone"`) {
		t.Fatalf("decompiled DSL should write only the explicit systemone api:\n%s", text)
	}
	recompiled, errs := Compile(text)
	if len(errs) > 0 {
		t.Fatalf("recompile errors: %v\n%s", errs, text)
	}
	if !reflect.DeepEqual(cfg.Entrypoints, recompiled.Entrypoints) {
		t.Fatalf("entrypoints changed after round trip:\n%+v\n%+v", cfg.Entrypoints, recompiled.Entrypoints)
	}
}

func TestCompileRejectsUnknownEntrypointAPI(t *testing.T) {
	_, errs := Compile(`
MODEL shared_model {}
ENTRYPOINT { api: "batch" model_names: ["router/auto"] recipe: "alpha" }
RECIPE alpha { ROUTE alpha_route { MODEL shared_model } }
`)
	for _, err := range errs {
		if strings.Contains(err.Error(), `ENTRYPOINT api must be chat or systemone, got "batch"`) {
			return
		}
	}
	t.Fatalf("compile errors = %v, want an unknown api error", errs)
}

func TestValidateRoutingScopesDoesNotLeakSignals(t *testing.T) {
	input := `
MODEL shared_model {}
RECIPE alpha {
  SIGNAL keyword alpha_only { keywords: ["alpha"] }
  ROUTE alpha_route { WHEN keyword("alpha_only") MODEL shared_model }
}
RECIPE beta {
  ROUTE beta_route { WHEN keyword("alpha_only") MODEL shared_model }
}`
	diagnostics, parseErrs := Validate(input)
	if len(parseErrs) > 0 {
		t.Fatalf("parse errors: %v", parseErrs)
	}
	for _, diagnostic := range diagnostics {
		if strings.Contains(diagnostic.Message, `RECIPE "beta"`) &&
			strings.Contains(diagnostic.Message, `alpha_only`) {
			return
		}
	}
	t.Fatalf("expected a beta-scoped undefined signal diagnostic, got %+v", diagnostics)
}

func TestCompilePreservesUnifiedPluginFields(t *testing.T) {
	input := `
SIGNAL keyword factual { keywords: ["verify"] }
ROUTE factual_route {
  WHEN keyword("factual")
  MODEL model
  PLUGIN semantic_cache { enabled: true, similarity_threshold: 0.9, ttl_seconds: 900 }
  PLUGIN hallucination {
    enabled: true
    hallucination_action: "header"
    unverified_factual_action: "header"
    include_hallucination_details: true
  }
}`
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	text, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{"ttl_seconds: 900", `unverified_factual_action: "header"`, "include_hallucination_details: true"} {
		if !strings.Contains(text, expected) {
			t.Fatalf("decompiled DSL is missing %q:\n%s", expected, text)
		}
	}
}
