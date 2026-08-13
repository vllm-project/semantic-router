package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// entrypointRulesTestConfigYAML declares one conditional entrypoint mirroring
// the issue's own worked example: a caller matching tenant=A,user=B on the
// chat-completions path gets the more specific "privacy" recipe; any other
// tenant=A caller gets "default"; anyone else is denied.
const entrypointRulesTestConfigYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
      description: default tier
    - name: model-b
      description: privacy tier
  decisions: []
recipes:
  - name: privacy
    routing: {}
entrypoints:
  - model_names: ["vllm-sr/tenant-auto"]
    rules:
      - name: tenant-a-user-b
        matches:
          - path: {type: exact, value: "/v1/chat/completions"}
            headers:
              - {name: x-authz-tenant-id, value: "A"}
              - {name: x-authz-user-id, value: "B"}
        recipe: privacy
      - name: tenant-a-default
        matches:
          - headers: [{name: x-authz-tenant-id, value: "A"}]
        recipe: default
  - model_names: ["vllm-sr/legacy"]
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

func newEntrypointRulesTestRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(entrypointRulesTestConfigYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	return &OpenAIRouter{Config: cfg}
}

func TestResolveEntrypointForRequestConditionalMatched(t *testing.T) {
	router := newEntrypointRulesTestRouter(t)

	ctx := &RequestContext{Headers: map[string]string{
		":path":             "/v1/chat/completions",
		"x-authz-tenant-id": "A",
		"x-authz-user-id":   "B",
	}}
	router.resolveEntrypointForRequest("vllm-sr/tenant-auto", ctx)

	if ctx.Routing.IsDenied() {
		t.Fatalf("expected a matched recipe, got denied (status=%d reason=%q)", ctx.Routing.DeniedStatus(), ctx.Routing.DeniedReason())
	}
	if got := ctx.Routing.SelectedRecipe(); got == nil || got.Name != "privacy" {
		t.Fatalf("expected the privacy recipe (more specific rule), got %+v", got)
	}
}

func TestResolveEntrypointForRequestConditionalLessSpecificMatch(t *testing.T) {
	router := newEntrypointRulesTestRouter(t)

	ctx := &RequestContext{Headers: map[string]string{
		":path":             "/v1/responses", // not the path the specific rule requires
		"x-authz-tenant-id": "A",
		"x-authz-user-id":   "B",
	}}
	router.resolveEntrypointForRequest("vllm-sr/tenant-auto", ctx)

	if ctx.Routing.IsDenied() {
		t.Fatalf("expected a matched recipe via the broader tenant-only rule, got denied (status=%d reason=%q)", ctx.Routing.DeniedStatus(), ctx.Routing.DeniedReason())
	}
	if got := ctx.Routing.SelectedRecipe(); got == nil || got.Name != config.DefaultRecipeName {
		t.Fatalf("expected the default recipe (broader rule), got %+v", got)
	}
}

func TestResolveEntrypointForRequestConditionalDeniedNeverPassthrough(t *testing.T) {
	router := newEntrypointRulesTestRouter(t)

	ctx := &RequestContext{Headers: map[string]string{
		":path":             "/v1/chat/completions",
		"x-authz-tenant-id": "some-other-tenant",
	}}
	router.resolveEntrypointForRequest("vllm-sr/tenant-auto", ctx)

	if !ctx.Routing.IsDenied() {
		t.Fatalf("expected the claimed-but-unmatched alias to be denied, got recipe=%+v passthrough=%v", ctx.Routing.SelectedRecipe(), ctx.Routing.IsPassthrough())
	}
	if ctx.Routing.IsPassthrough() {
		t.Fatal("a claimed conditional entrypoint alias with no matching rule must never become passthrough")
	}
	if ctx.Routing.DeniedStatus() != 404 {
		t.Fatalf("denied status = %d, want 404", ctx.Routing.DeniedStatus())
	}
}

func TestResolveEntrypointForRequestLegacyEntrypointsUnaffected(t *testing.T) {
	// The test config declares both a legacy and a conditional entrypoint:
	// adding conditional support must not perturb the legacy alias's
	// behavior at all, with or without headers present.
	router := newEntrypointRulesTestRouter(t)

	for _, headers := range []map[string]string{nil, {"x-authz-tenant-id": "unrelated"}} {
		ctx := &RequestContext{Headers: headers}
		router.resolveEntrypointForRequest("vllm-sr/legacy", ctx)
		if ctx.Routing.IsDenied() {
			t.Fatalf("legacy entrypoint must never be denied, headers=%v", headers)
		}
		if got := ctx.Routing.SelectedRecipe(); got == nil || got.Name != "privacy" {
			t.Fatalf("legacy entrypoint must always resolve to its fixed recipe regardless of headers, got %+v", got)
		}
	}
}

func TestRunRequestPreRoutingStagesReturnsErrorForDeniedEntrypoint(t *testing.T) {
	router := newEntrypointRulesTestRouter(t)

	ctx := &RequestContext{Headers: map[string]string{
		":path":             "/v1/chat/completions",
		"x-authz-tenant-id": "some-other-tenant",
	}}
	_, resp := router.runRequestPreRoutingStages("vllm-sr/tenant-auto", &FastExtractResult{}, ctx)

	if resp == nil {
		t.Fatal("expected an immediate error response for a denied entrypoint, got nil")
	}
	immediate := resp.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("expected an ImmediateResponse")
	}
	var body struct {
		Error struct {
			Code int `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(immediate.Body, &body); err != nil {
		t.Fatalf("failed to parse error body: %v", err)
	}
	if body.Error.Code != 404 {
		t.Fatalf("error code = %d, want 404", body.Error.Code)
	}
}
