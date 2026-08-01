package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// recipeReadSiteConfigYAML puts every routed behavior under test on a
// NON-default recipe: reasoning effort, a plugin that gates runtime wiring, and
// a model reference. Anything that reads the flat DefaultDecisions field
// instead of RouterConfig.AllRoutingDecisions() sees an empty set here.
const recipeReadSiteConfigYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
    - name: model-b
  signals:
    keywords:
      - name: urgent_keywords
        operator: OR
        keywords: ["urgent"]
  decisions:
    - name: default_route
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: urgent_keywords
      modelRefs:
        - model: model-a
          use_reasoning: false
recipes:
  - name: privacy
    routing:
      signals:
        keywords:
          - name: pii_keywords
            operator: OR
            keywords: ["ssn"]
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: pii_keywords
          modelRefs:
            - model: model-b
              use_reasoning: true
              reasoning_effort: high
          plugins:
            - type: memory
              configuration:
                enabled: true
entrypoints:
  - model_names: ["vllm-sr/privacy"]
    recipe: privacy
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
    - name: model-b
      backend_refs:
        - endpoint: 127.0.0.1:8001
`

// TestRecipeDecisionsReachDownstreamReadSites pins the read sites that must
// observe a non-default recipe's decisions. Each of these silently ignored
// recipes while they read the flat decision field.
func TestRecipeDecisionsReachDownstreamReadSites(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(recipeReadSiteConfigYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	router := &OpenAIRouter{Config: cfg}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("failed to build classifier: %v", err)
	}

	t.Run("config decision lookup by name", func(t *testing.T) {
		if decision := cfg.GetDecisionByName("privacy_route"); decision == nil {
			t.Fatal("expected the recipe's decision to be found by name")
		}
	})

	t.Run("classifier model lookup", func(t *testing.T) {
		models := classifier.GetModelsForCategory("privacy_route")
		if len(models) != 1 || models[0] != "model-b" {
			t.Fatalf("expected [model-b] for the recipe's decision, got %v", models)
		}
		if decision := classifier.GetDecisionByName("privacy_route"); decision == nil {
			t.Fatal("expected the classifier to resolve the recipe's decision")
		}
	})

	t.Run("reasoning effort", func(t *testing.T) {
		if effort := router.getReasoningEffort("privacy_route", "model-b"); effort != "high" {
			t.Fatalf("expected the recipe decision's reasoning effort %q, got %q", "high", effort)
		}
	})

	t.Run("memory plugin auto-enable", func(t *testing.T) {
		if !cfg.IsMemoryEnabled() {
			t.Fatal("expected the recipe decision's memory plugin to enable the memory runtime")
		}
	})

	// ValidateEndpoints is a config helper with no load-path caller: the
	// load path checks recipe modelRefs against modelCards only, and a
	// missing endpoint surfaces at request time. This pins the helper's
	// recipe scope, not a load-time endpoint check.
	t.Run("ValidateEndpoints helper sees recipe decisions", func(t *testing.T) {
		if err := cfg.ValidateEndpoints(); err != nil {
			t.Fatalf("expected the recipe's model reference to validate, got %v", err)
		}
	})
}

// TestAlgorithmSlugCannotReachRecipeDecisions pins recipe isolation against the
// algorithm virtual slugs. The slugs select decisions without consulting the
// entrypoint table, so a decision that lives only in a named recipe must stay
// unreachable through them — otherwise the recipe's entrypoint is bypassable.
func TestAlgorithmSlugCannotReachRecipeDecisions(t *testing.T) {
	const recipeOwnedFusionYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
    - name: model-b
  signals:
    keywords:
      - name: urgent_keywords
        operator: OR
        keywords: ["urgent"]
  decisions:
    - name: default_route
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: urgent_keywords
      modelRefs:
        - model: model-a
recipes:
  - name: privacy
    routing:
      signals:
        keywords:
          - name: pii_keywords
            operator: OR
            keywords: ["ssn"]
      decisions:
        - name: privacy_fusion
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: pii_keywords
          algorithm:
            type: fusion
            fusion:
              analysis_models: ["model-b"]
          modelRefs:
            - model: model-b
entrypoints:
  - model_names: ["vllm-sr/privacy"]
    recipe: privacy
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
    - name: model-b
      backend_refs:
        - endpoint: 127.0.0.1:8001
`

	cfg, err := config.ParseYAMLBytes([]byte(recipeOwnedFusionYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	router := &OpenAIRouter{Config: cfg}

	slugs := cfg.Looper.Fusion.EffectiveModelNames()
	if len(slugs) == 0 {
		t.Fatal("expected a Fusion virtual slug")
	}
	slug := slugs[0]

	ctx := &RequestContext{}
	router.resolveEntrypointForRequest(slug, ctx)
	if ctx.EntrypointRecipe != nil {
		t.Fatalf("the algorithm slug must not resolve an entrypoint, got %+v", ctx.EntrypointRecipe)
	}

	if candidates := router.decisionCandidatesForRequest(slug, ctx); len(candidates) != 0 {
		t.Fatalf("the privacy recipe's decision leaked into the %s slug: %+v", slug, candidates)
	}
	if decision := router.defaultLooperDecisionByAlgorithm("fusion"); decision != nil {
		t.Fatalf("the looper fallback leaked the recipe's decision: %s", decision.Name)
	}

	// ...and /v1/models must not advertise a slug that cannot route.
	if names := cfg.ExposedFusionModelNames(); len(names) != 0 {
		t.Fatalf("expected no Fusion slug to be advertised, got %v", names)
	}
}

// TestRecipeDecisionProjectionRefsAreValidated covers the contract validators:
// they must walk every recipe's decisions, not just the default profile's.
// Projection outputs are never auto-declared (unlike domain categories, which
// normalizeSignals synthesizes on reference), so an unknown output name is a
// genuine contract violation the loader has to reject.
func TestRecipeDecisionProjectionRefsAreValidated(t *testing.T) {
	const unknownProjectionYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
  signals:
    keywords:
      - name: urgent_keywords
        operator: OR
        keywords: ["urgent"]
  decisions:
    - name: default_route
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: urgent_keywords
      modelRefs:
        - model: model-a
recipes:
  - name: privacy
    routing:
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions:
              - type: projection
                name: no_such_projection_output
          modelRefs:
            - model: model-a
entrypoints:
  - model_names: ["vllm-sr/privacy"]
    recipe: privacy
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
`

	if _, err := config.ParseYAMLBytes([]byte(unknownProjectionYAML)); err == nil {
		t.Fatal("expected a recipe decision referencing an unknown projection output to be rejected")
	}
}
