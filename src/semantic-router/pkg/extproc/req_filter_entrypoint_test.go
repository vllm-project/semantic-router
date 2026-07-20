package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
