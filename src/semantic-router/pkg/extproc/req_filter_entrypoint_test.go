package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
)

const entrypointTestConfigYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
      description: default tier
    - name: model-b
      description: privacy tier
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
              use_reasoning: false
entrypoints:
  - model_names: ["vllm-sr/privacy"]
    recipe: privacy
  - model_names: ["vllm-sr/default-alias"]
    recipe: default
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

func newEntrypointTestRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(entrypointTestConfigYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	return &OpenAIRouter{Config: cfg}
}

func TestResolveEntrypointForRequest(t *testing.T) {
	router := newEntrypointTestRouter(t)

	ctx := &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/privacy", ctx)
	if ctx.EntrypointRecipe == nil || ctx.EntrypointRecipe.Name != "privacy" {
		t.Fatalf("expected the privacy recipe to be resolved, got %+v", ctx.EntrypointRecipe)
	}

	ctx = &RequestContext{}
	router.resolveEntrypointForRequest("model-a", ctx)
	if ctx.EntrypointRecipe != nil {
		t.Fatalf("expected a plain model name to resolve no recipe, got %+v", ctx.EntrypointRecipe)
	}
}

func TestRequestModelActsAsAuto(t *testing.T) {
	router := newEntrypointTestRouter(t)

	cases := []struct {
		model string
		want  bool
	}{
		{model: config.DefaultVSRAutoModelName, want: true},
		{model: "vllm-sr/privacy", want: true},
		{model: "vllm-sr/default-alias", want: true},
		{model: "model-a", want: false},
		{model: "unknown-model", want: false},
	}
	for _, testCase := range cases {
		if got := router.requestModelActsAsAuto(testCase.model); got != testCase.want {
			t.Fatalf("requestModelActsAsAuto(%q) = %v, want %v", testCase.model, got, testCase.want)
		}
	}
}

func TestDecisionCandidatesForRequest(t *testing.T) {
	router := newEntrypointTestRouter(t)

	ctx := &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/privacy", ctx)
	candidates := router.decisionCandidatesForRequest("vllm-sr/privacy", ctx)
	if len(candidates) != 1 || candidates[0].Name != "privacy_route" {
		t.Fatalf("expected the privacy recipe's decisions as candidates, got %+v", candidates)
	}

	// An entrypoint alias of the default recipe keeps the engine-default
	// candidate path (nil), which evaluates the flat default decisions.
	ctx = &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/default-alias", ctx)
	if ctx.EntrypointRecipe == nil || ctx.EntrypointRecipe.Name != config.DefaultRecipeName {
		t.Fatalf("expected the default recipe to be resolved, got %+v", ctx.EntrypointRecipe)
	}
	if candidates := router.decisionCandidatesForRequest("vllm-sr/default-alias", ctx); candidates != nil {
		t.Fatalf("expected nil candidates for a default-recipe alias, got %+v", candidates)
	}

	ctx = &RequestContext{}
	if candidates := router.decisionCandidatesForRequest(config.DefaultVSRAutoModelName, ctx); candidates != nil {
		t.Fatalf("expected nil candidates for the auto model, got %+v", candidates)
	}
}

// newEntrypointFlowRouter builds a router with a real classifier so decision
// evaluation runs the full signal → decision → model-selection chain. The
// fixture only uses keyword signals, which need no local model artifacts.
func newEntrypointFlowRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(entrypointTestConfigYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("failed to build classifier: %v", err)
	}
	return &OpenAIRouter{Config: cfg, Classifier: classifier}
}

func TestPerformDecisionEvaluationSelectsRecipeByEntrypoint(t *testing.T) {
	router := newEntrypointFlowRouter(t)

	cases := []struct {
		name         string
		model        string
		message      string
		wantDecision string
		wantModel    string
	}{
		{
			name:         "privacy entrypoint routes through the privacy recipe",
			model:        "vllm-sr/privacy",
			message:      "my ssn is exposed",
			wantDecision: "privacy_route",
			wantModel:    "model-b",
		},
		{
			// The pii signal matches globally, but the default recipe's
			// decisions do not reference it: the request falls back to the
			// default model instead of leaking into another recipe's routes.
			name:         "auto model ignores other recipes' decisions",
			model:        config.DefaultVSRAutoModelName,
			message:      "my ssn is exposed",
			wantDecision: "",
			wantModel:    "model-a",
		},
		{
			name:         "auto model matches the default recipe decision",
			model:        config.DefaultVSRAutoModelName,
			message:      "this is urgent",
			wantDecision: "default_route",
			wantModel:    "model-a",
		},
		{
			name:         "privacy entrypoint falls back to the default model when its recipe matches nothing",
			model:        "vllm-sr/privacy",
			message:      "this is urgent",
			wantDecision: "",
			wantModel:    "model-a",
		},
		{
			name:         "default recipe alias behaves like the auto model",
			model:        "vllm-sr/default-alias",
			message:      "this is urgent",
			wantDecision: "default_route",
			wantModel:    "model-a",
		},
		{
			name:         "explicit model preserves the client selection",
			model:        "model-a",
			message:      "this is urgent",
			wantDecision: "default_route",
			wantModel:    "",
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			ctx := &RequestContext{
				TraceContext: context.Background(),
				Headers:      map[string]string{},
			}
			router.resolveEntrypointForRequest(testCase.model, ctx)

			decisionName, _, _, selectedModel, err := router.performDecisionEvaluation(
				testCase.model,
				signalConversationHistory{currentUserMessage: testCase.message},
				ctx,
			)
			if err != nil {
				t.Fatalf("performDecisionEvaluation failed: %v", err)
			}
			if decisionName != testCase.wantDecision {
				t.Fatalf("expected decision %q, got %q", testCase.wantDecision, decisionName)
			}
			if selectedModel != testCase.wantModel {
				t.Fatalf("expected selected model %q, got %q", testCase.wantModel, selectedModel)
			}
			if testCase.wantDecision != "" && ctx.VSRSelectedDecision != nil && ctx.VSRSelectedDecision.Name != testCase.wantDecision {
				t.Fatalf("expected ctx.VSRSelectedDecision %q, got %q", testCase.wantDecision, ctx.VSRSelectedDecision.Name)
			}
		})
	}
}

func TestModelsListingIncludesEntrypointNames(t *testing.T) {
	router := newEntrypointTestRouter(t)

	response, err := router.handleModelsRequest("/v1/models")
	if err != nil {
		t.Fatalf("handleModelsRequest failed: %v", err)
	}
	immediateResp := response.GetImmediateResponse()
	if immediateResp == nil {
		t.Fatal("expected an immediate response")
	}

	var modelList OpenAIModelList
	if err := json.Unmarshal(immediateResp.Body, &modelList); err != nil {
		t.Fatalf("failed to parse response body: %v", err)
	}

	descriptionByID := make(map[string]string, len(modelList.Data))
	for _, model := range modelList.Data {
		descriptionByID[model.ID] = model.Description
	}
	if _, ok := descriptionByID["vllm-sr/privacy"]; !ok {
		t.Fatalf("expected vllm-sr/privacy in the model list, got %+v", descriptionByID)
	}
	if _, ok := descriptionByID["vllm-sr/default-alias"]; !ok {
		t.Fatalf("expected vllm-sr/default-alias in the model list, got %+v", descriptionByID)
	}
	if got := descriptionByID["vllm-sr/default-alias"]; got != "Entrypoint for the default routing recipe" {
		t.Fatalf("expected the generic entrypoint description for the default alias, got %q", got)
	}
}

func TestModelsListingUsesExplicitRoutingTypes(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterOptions: config.RouterOptions{
				AutoModelNames: []string{"router/balanced"},
			},
			Entrypoints: []config.EntrypointMapping{
				{ModelNames: []string{"router/flash"}, Recipe: "speed-first"},
			},
			Recipes: []config.RoutingRecipe{
				{
					Name:        "speed-first",
					Description: "Intelligent Router for Mixture-of-Models",
				},
			},
		},
	}

	response, err := router.handleModelsRequest("/v1/models")
	if err != nil {
		t.Fatalf("handleModelsRequest failed: %v", err)
	}
	var modelList OpenAIModelList
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &modelList); err != nil {
		t.Fatalf("failed to parse response body: %v", err)
	}
	if len(modelList.Data) != 2 {
		t.Fatalf("model count = %d, want 2", len(modelList.Data))
	}
	wantRoutingTypes := map[string]publicmodels.RoutingType{
		"router/balanced": publicmodels.RoutingTypeAutoAlias,
		"router/flash":    publicmodels.RoutingTypeEntrypoint,
	}
	for _, model := range modelList.Data {
		if model.OwnedBy != "vllm-semantic-router" {
			t.Fatalf("%s owned_by = %q, want vllm-semantic-router", model.ID, model.OwnedBy)
		}
		if model.RoutingType != wantRoutingTypes[model.ID] {
			t.Fatalf("%s routing_type = %q, want %q", model.ID, model.RoutingType, wantRoutingTypes[model.ID])
		}
		if model.Description == "" {
			t.Fatalf("%s has an empty description", model.ID)
		}
	}
}

// entrypointRecipesOnlyConfigYAML keeps every decision inside a non-default
// recipe: the flat Decisions field stays empty, which used to trip the
// "no decisions configured" short-circuit before decision evaluation.
const entrypointRecipesOnlyConfigYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
      description: default tier
    - name: model-b
      description: privacy tier
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
              use_reasoning: false
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

func TestPerformDecisionEvaluationRecipesOnlyConfig(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(entrypointRecipesOnlyConfigYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("failed to build classifier: %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}

	cases := []struct {
		name         string
		model        string
		message      string
		wantDecision string
		wantModel    string
	}{
		{
			name:         "entrypoint routes through its recipe despite empty flat decisions",
			model:        "vllm-sr/privacy",
			message:      "my ssn is exposed",
			wantDecision: "privacy_route",
			wantModel:    "model-b",
		},
		{
			name:         "auto model keeps the default profile fallback",
			model:        config.DefaultVSRAutoModelName,
			message:      "my ssn is exposed",
			wantDecision: "",
			wantModel:    "model-a",
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			ctx := &RequestContext{
				TraceContext: context.Background(),
				Headers:      map[string]string{},
			}
			router.resolveEntrypointForRequest(testCase.model, ctx)

			decisionName, _, _, selectedModel, err := router.performDecisionEvaluation(
				testCase.model,
				signalConversationHistory{currentUserMessage: testCase.message},
				ctx,
			)
			if err != nil {
				t.Fatalf("performDecisionEvaluation failed: %v", err)
			}
			if decisionName != testCase.wantDecision {
				t.Fatalf("expected decision %q, got %q", testCase.wantDecision, decisionName)
			}
			if selectedModel != testCase.wantModel {
				t.Fatalf("expected selected model %q, got %q", testCase.wantModel, selectedModel)
			}
		})
	}
}

// entrypointDecisionlessRecipeYAML maps an entrypoint to a recipe that
// declares signals but no decisions. Its traffic must not be routed by the
// default profile's decisions.
const entrypointDecisionlessRecipeYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
      description: default tier
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
  - name: screening
    routing:
      signals:
        keywords:
          - name: screening_keywords
            operator: OR
            keywords: ["screening"]
entrypoints:
  - model_names: ["vllm-sr/screening"]
    recipe: screening
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
`

func TestDecisionlessRecipeStaysIsolatedFromDefaultDecisions(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(entrypointDecisionlessRecipeYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("failed to build classifier: %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}

	ctx := &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/screening", ctx)
	candidates := router.decisionCandidatesForRequest("vllm-sr/screening", ctx)
	if candidates == nil {
		t.Fatal("a decision-less recipe must scope candidates to an empty slice, not nil")
	}
	if len(candidates) != 0 {
		t.Fatalf("expected no candidates, got %+v", candidates)
	}

	flowCtx := &RequestContext{
		TraceContext: context.Background(),
		Headers:      map[string]string{},
	}
	router.resolveEntrypointForRequest("vllm-sr/screening", flowCtx)
	decisionName, _, _, selectedModel, err := router.performDecisionEvaluation(
		"vllm-sr/screening",
		signalConversationHistory{currentUserMessage: "this is urgent"},
		flowCtx,
	)
	if err != nil {
		t.Fatalf("performDecisionEvaluation failed: %v", err)
	}
	if decisionName != "" {
		t.Fatalf("a decision-less recipe must not select the default profile's decision, got %q", decisionName)
	}
	if selectedModel != "model-a" {
		t.Fatalf("expected the default-model fallback, got %q", selectedModel)
	}
}

func TestFindDecisionByNameSeesRecipeDecisions(t *testing.T) {
	router := newEntrypointTestRouter(t)

	decision := router.findDecisionByName("privacy_route")
	if decision == nil || decision.Name != "privacy_route" {
		t.Fatalf("expected the looper decision lookup to see recipe decisions, got %+v", decision)
	}
}

func TestSemanticCacheScopeStaysRouteLocalForRecipesOnlyConfig(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(entrypointRecipesOnlyConfigYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	cfg.SemanticCache.Enabled = true
	router := &OpenAIRouter{Config: cfg}

	// Decisions exist (inside the privacy recipe), so an unmatched request
	// must not fall back to the global cache toggle.
	if router.semanticCacheEnabledForScope("") {
		t.Fatal("recipes-only configs must keep semantic cache route-local, not global")
	}
	if router.semanticCacheEnabledForScope("privacy_route") != cfg.IsCacheEnabledForDecision("privacy_route") {
		t.Fatal("decision-scoped lookup must delegate to IsCacheEnabledForDecision")
	}
}
