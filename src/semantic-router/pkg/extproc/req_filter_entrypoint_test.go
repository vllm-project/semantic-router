package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
)

const defaultEntrypointModel = "vllm-sr/auto"

const entrypointTestConfigYAML = `
version: v0.4
models:
  - name: model-a
    card: {description: default tier}
    connections: [{provider: vllm, endpoint: http://127.0.0.1:8000, model: model-a}]
  - name: model-b
    card: {description: privacy tier}
    connections: [{provider: vllm, endpoint: http://127.0.0.1:8001, model: model-b}]
recipes:
  - name: default
    document:
      signals:
        keywords:
          - name: route_keyword
            operator: OR
            keywords: ["urgent"]
      decisions:
        - name: default_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: route_keyword
  - name: privacy
    document:
      signals:
        keywords:
          - name: route_keyword
            operator: OR
            keywords: ["ssn"]
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: route_keyword
entrypoints:
  - name: vllm-sr/auto
    recipe: default
    assignments:
      default_route: {models: [{model: model-a}]}
  - name: vllm-sr/privacy
    recipe: privacy
    assignments:
      privacy_route: {models: [{model: model-b}]}
  - name: vllm-sr/privacy-fast
    recipe: privacy
    assignments:
      privacy_route: {models: [{model: model-a}]}
  - name: vllm-sr/default-alias
    recipe: default
    assignments:
      default_route: {models: [{model: model-a}]}
`

func newEntrypointTestRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := parseExtProcAuthoringConfig(t, entrypointTestConfigYAML)
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	cfg.DefaultModel = "model-a"
	return &OpenAIRouter{Config: cfg}
}

func TestResolveEntrypointForRequest(t *testing.T) {
	router := newEntrypointTestRouter(t)

	ctx := &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/privacy", ctx)
	if ctx.Routing.SelectedRecipe() == nil || ctx.Routing.SelectedRecipe().Name != "privacy" {
		t.Fatalf("expected the privacy recipe to be resolved, got %+v", ctx.Routing.SelectedRecipe())
	}
	privacyScope := ctx.Routing.RuntimeScope()
	if privacyScope == "" || privacyScope == config.RecipeName("privacy") {
		t.Fatalf("privacy entrypoint did not receive an isolated runtime scope: %q", privacyScope)
	}

	fastCtx := &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/privacy-fast", fastCtx)
	if fastCtx.Routing.RecipeName() != "privacy" || fastCtx.Routing.RuntimeScope() == privacyScope {
		t.Fatalf("entrypoints sharing one recipe were not runtime-isolated: privacy=%q fast=%q", privacyScope, fastCtx.Routing.RuntimeScope())
	}
	ctx.SessionID = "shared-session"
	fastCtx.SessionID = "shared-session"
	if routingSessionStateKey(ctx) == routingSessionStateKey(fastCtx) {
		t.Fatalf("entrypoint runtime scopes collided in session state: %q", routingSessionStateKey(ctx))
	}
	ctx.VSRSelectedDecisionName = "privacy_route"
	fastCtx.VSRSelectedDecisionName = "privacy_route"
	learning := newRouterLearningRuntime(nil, nil, nil)
	learning.recordModelExperience(
		requestDecisionStateKey(ctx),
		1,
		"model-a",
		routerLearningOutcomeGoodFit,
		1,
	)
	if got := learning.experienceSnapshot(requestDecisionStateKey(fastCtx), 1, "model-a"); got.GoodFitCount != 0 {
		t.Fatalf("entrypoint learning experience crossed runtime scopes: %+v", got)
	}
	replay := buildReplayRoutingRecord(fastCtx, "vllm-sr/privacy-fast", "model-a", "privacy_route")
	if replay.Recipe != "privacy" || replay.RoutingScope != string(fastCtx.Routing.RuntimeScope()) {
		t.Fatalf("replay identity mixed logical recipe and internal runtime scope: %+v", replay)
	}

	ctx = &RequestContext{}
	router.resolveEntrypointForRequest("model-a", ctx)
	if ctx.Routing.SelectedRecipe() != nil {
		t.Fatalf("expected a plain model name to resolve no recipe, got %+v", ctx.Routing.SelectedRecipe())
	}
}

func TestLooperHydrationRestoresBoundEntrypointViewByRuntimeScope(t *testing.T) {
	router := newEntrypointTestRouter(t)
	parent := &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/privacy-fast", parent)
	scope := parent.Routing.RuntimeScope()

	looperContext := &RequestContext{
		LooperRequest: true,
		Headers: map[string]string{
			headers.VSRSelectedRecipe: string(scope),
		},
	}
	router.hydrateLooperRoutingContext(looperContext)
	recipe := looperContext.Routing.SelectedRecipe()
	if recipe == nil || recipe.RuntimeScope() != scope {
		t.Fatalf("looper hydration did not restore runtime scope %q: %+v", scope, recipe)
	}
	decision := routingRecipeDecisionByName(recipe, "privacy_route")
	if decision == nil || len(decision.ModelRefs) != 1 || decision.ModelRefs[0].Model != "model-a" {
		t.Fatalf("looper hydration restored stale reusable-recipe models: %+v", decision)
	}
}

func TestClassifierForRequestNeverFallsBackToRootClassifier(t *testing.T) {
	defaultClassifier := &classification.Classifier{}
	router := &OpenAIRouter{Classifier: defaultClassifier}

	defaultContext := &RequestContext{}
	defaultContext.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
	if got := router.classifierForRequest(defaultContext); got != nil {
		t.Fatalf("default Recipe unexpectedly used the root classifier: %p", got)
	}

	namedContext := &RequestContext{}
	namedContext.Routing.SelectRecipe(&config.RoutingRecipe{Name: "privacy"})
	if got := router.classifierForRequest(namedContext); got != nil {
		t.Fatalf("named Recipe unexpectedly used the root classifier: %p", got)
	}
}

func TestRequestModelIsEntrypoint(t *testing.T) {
	router := newEntrypointTestRouter(t)

	cases := []struct {
		model string
		want  bool
	}{
		{model: defaultEntrypointModel, want: true},
		{model: "vllm-sr/privacy", want: true},
		{model: "vllm-sr/default-alias", want: true},
		{model: "auto", want: false},
		{model: "MoM", want: false},
		{model: "model-a", want: false},
		{model: "unknown-model", want: false},
	}
	for _, testCase := range cases {
		if got := router.requestModelIsEntrypoint(testCase.model); got != testCase.want {
			t.Fatalf("requestModelIsEntrypoint(%q) = %v, want %v", testCase.model, got, testCase.want)
		}
	}
}

func TestDecisionCandidatesForRequest(t *testing.T) {
	router := newEntrypointTestRouter(t)

	ctx := &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/privacy", ctx)
	candidates := router.decisionCandidatesForRequest(ctx)
	if len(candidates) != 1 || candidates[0].Name != "privacy_route" {
		t.Fatalf("expected the privacy recipe's decisions as candidates, got %+v", candidates)
	}

	// An Entrypoint may explicitly select the Recipe named default.
	ctx = &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/default-alias", ctx)
	if ctx.Routing.SelectedRecipe() == nil || ctx.Routing.SelectedRecipe().Name != config.DefaultRecipeName {
		t.Fatalf("expected the default recipe to be resolved, got %+v", ctx.Routing.SelectedRecipe())
	}
	if candidates := router.decisionCandidatesForRequest(ctx); len(candidates) != 1 || candidates[0].Name != "default_route" {
		t.Fatalf("expected the default recipe candidates, got %+v", candidates)
	}

	ctx = &RequestContext{}
	router.resolveEntrypointForRequest(defaultEntrypointModel, ctx)
	if candidates := router.decisionCandidatesForRequest(ctx); len(candidates) != 1 || candidates[0].Name != "default_route" {
		t.Fatalf("expected the explicit vllm-sr/auto Entrypoint to use default Recipe candidates, got %+v", candidates)
	}

	// Legacy automatic aliases are inert unless the manifest names them as an
	// Entrypoint explicitly.
	ctx = &RequestContext{}
	router.resolveEntrypointForRequest("auto", ctx)
	if candidates := router.decisionCandidatesForRequest(ctx); len(candidates) != 0 {
		t.Fatalf("legacy auto alias selected Recipe candidates: %+v", candidates)
	}
}

// newEntrypointFlowRouter builds a router with a real classifier so decision
// evaluation runs the full signal → decision → model-selection chain. The
// fixture only uses keyword signals, which need no local model artifacts.
func newEntrypointFlowRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := parseExtProcAuthoringConfig(t, entrypointTestConfigYAML)
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	cfg.DefaultModel = "model-a"
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("failed to build recipe classifiers: %v", err)
	}
	return &OpenAIRouter{
		Config:            cfg,
		Classifier:        classifiers.Default(),
		RecipeClassifiers: classifiers,
	}
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
			name:         "bound entrypoint selects its physical model from the derived recipe",
			model:        "vllm-sr/privacy-fast",
			message:      "my ssn is exposed",
			wantDecision: "privacy_route",
			wantModel:    "model-a",
		},
		{
			// The privacy signal does not even run for the default recipe.
			name:         "default Entrypoint ignores other Recipes' decisions",
			model:        defaultEntrypointModel,
			message:      "my ssn is exposed",
			wantDecision: "",
			wantModel:    "",
		},
		{
			name:         "default Entrypoint matches its Recipe decision",
			model:        defaultEntrypointModel,
			message:      "this is urgent",
			wantDecision: "default_route",
			wantModel:    "model-a",
		},
		{
			name:         "privacy Entrypoint does not escape its Recipe when nothing matches",
			model:        "vllm-sr/privacy",
			message:      "this is urgent",
			wantDecision: "",
			wantModel:    "",
		},
		{
			name:         "second Entrypoint can select the default Recipe",
			model:        "vllm-sr/default-alias",
			message:      "this is urgent",
			wantDecision: "default_route",
			wantModel:    "model-a",
		},
		{
			name:         "explicit model preserves the client selection",
			model:        "model-a",
			message:      "this is urgent",
			wantDecision: "",
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

func TestModelsListingUsesExplicitRoutingMetadata(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
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
	if len(modelList.Data) != 1 {
		t.Fatalf("model count = %d, want 1 explicit Entrypoint", len(modelList.Data))
	}
	for _, model := range modelList.Data {
		if model.ID != "router/flash" {
			t.Fatalf("unexpected hidden routing alias %q", model.ID)
		}
		if model.OwnedBy != "vllm-semantic-router" {
			t.Fatalf("%s owned_by = %q, want vllm-semantic-router", model.ID, model.OwnedBy)
		}
		if model.Routing.Resolution != publicmodels.ResolutionVirtual || !model.Routing.Selectable {
			t.Fatalf("%s routing metadata = %+v, want selectable virtual model", model.ID, model.Routing)
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
version: v0.4
models:
  - name: model-a
    card: {description: default tier}
    connections: [{provider: vllm, endpoint: http://127.0.0.1:8000, model: model-a}]
  - name: model-b
    card: {description: privacy tier}
    connections: [{provider: vllm, endpoint: http://127.0.0.1:8001, model: model-b}]
recipes:
  - name: privacy
    document:
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
entrypoints:
  - name: vllm-sr/privacy
    recipe: privacy
    assignments:
      privacy_route: {models: [{model: model-b}]}
`

func TestPerformDecisionEvaluationRecipesOnlyConfig(t *testing.T) {
	cfg, err := parseExtProcAuthoringConfig(t, entrypointRecipesOnlyConfigYAML)
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	cfg.DefaultModel = "model-a"
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("failed to build recipe classifiers: %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifiers.Default(), RecipeClassifiers: classifiers}

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

func TestSemanticCacheScopeStaysRouteLocalForRecipesOnlyConfig(t *testing.T) {
	cfg, err := parseExtProcAuthoringConfig(t, entrypointRecipesOnlyConfigYAML)
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	cfg.DefaultModel = "model-a"
	cfg.Enabled = true
	router := &OpenAIRouter{Config: cfg}

	privacy, ok := cfg.RecipeByName("privacy")
	if !ok || len(privacy.Profile.Decisions) != 1 {
		t.Fatalf("privacy recipe not normalized: %+v", privacy)
	}
	ctx := &RequestContext{}
	ctx.Routing.SelectRecipe(privacy)
	if router.semanticCacheEnabledForRequest(ctx) {
		t.Fatal("an unmatched recipe request must not fall back to the global cache toggle")
	}
	ctx.VSRSelectedDecision = &privacy.Profile.Decisions[0]
	if router.semanticCacheEnabledForRequest(ctx) != cfg.IsCacheEnabledForDecisionObject(&privacy.Profile.Decisions[0]) {
		t.Fatal("request-scoped lookup must use the selected recipe decision object")
	}
}

func TestConcreteModelBypassesSemanticCache(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
	}}
	ctx := &RequestContext{}
	ctx.Routing.SelectPassthrough()

	if router.semanticCacheEnabledForRequest(ctx) {
		t.Fatal("a concrete backend request must not enter recipe cache state")
	}
}
