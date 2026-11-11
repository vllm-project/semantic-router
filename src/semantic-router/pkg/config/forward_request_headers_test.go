package config

import (
	"fmt"
	"strings"
	"testing"
)

// forwardHeadersConfig lists GLOBAL on global.router.fallback, RECIPE on the
// recipe's fallback and DEPLOYMENT on a systemone deployment. The backend
// sends its key in X-Gateway-Key, the deployment in X-Decision-Key.
const forwardHeadersConfig = `version: v0.5
providers:
  defaults: {model: fast}
  models:
    - name: fast
      backend_refs:
        - name: fast
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 1
          auth: {header: X-Gateway-Key, api_key_env: FORWARD_TEST_GATEWAY_KEY}
recipes:
  - name: default
    decisions:
      - name: everything
        priority: 10
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: fast}]
    fallback:
      enabled: true
      forward_request_headers: RECIPE
global:
  router:
    fallback:
      forward_request_headers: GLOBAL
  model_catalog:
    deployments:
      gateway-d3:
        provider: systemone
        schema: vllm-sr/4.0
        endpoint: https://decisions.example.com/v1/systemone
        model: vLLM-SR-d3-27B
        auth: {header: X-Decision-Key, prefix: "", api_key_env: FORWARD_TEST_DECISION_KEY}
        forward_request_headers: DEPLOYMENT
`

func forwardHeadersDocument(global, recipe, deployment []string) []byte {
	list := func(names []string) string {
		quoted := make([]string, len(names))
		for i, name := range names {
			quoted[i] = fmt.Sprintf("%q", name)
		}
		return "[" + strings.Join(quoted, ", ") + "]"
	}
	return []byte(strings.NewReplacer("GLOBAL", list(global), "RECIPE", list(recipe), "DEPLOYMENT", list(deployment)).Replace(forwardHeadersConfig))
}

// Both forward_request_headers lists refuse every name a call the Router
// makes may not copy from the client, and say why.
func TestForwardRequestHeadersRefusesWhatNoCallMayCarry(t *testing.T) {
	t.Setenv("FORWARD_TEST_GATEWAY_KEY", "gateway-key")
	t.Setenv("FORWARD_TEST_DECISION_KEY", "decision-key")
	allowed := []string{"x-user-id", "X-Request-Id"}
	if _, err := ParseYAMLBytes(forwardHeadersDocument(allowed, allowed, allowed)); err != nil {
		t.Fatalf("an attribution header and the request id: %v", err)
	}

	refusals := []struct{ name, want string }{
		{":authority", `cannot include pseudo-header ":authority"`},
		{" ", "entries cannot be empty"},
		{"x user", `lists "x user", which is not an HTTP header name`},
		{"Authorization", `cannot include credential header "Authorization"`},
		{"proxy-authorization", `cannot include credential header "proxy-authorization"`},
		{"Cookie", `cannot include credential header "Cookie"`},
		{"x-api-key", `cannot include credential header "x-api-key"`},
		{"api-key", `cannot include credential header "api-key"`},
		{"Ocp-Apim-Subscription-Key", `cannot include credential header "Ocp-Apim-Subscription-Key"`},
		{"x-user-openai-key", `cannot include credential header "x-user-openai-key"`},
		{"Connection", `cannot include "Connection": the Router sets the framing and body headers of its own request`},
		{"keep-alive", `cannot include "keep-alive": the Router sets`},
		{"TE", `cannot include "TE": the Router sets`},
		{"trailer", `cannot include "trailer": the Router sets`},
		{"transfer-encoding", `cannot include "transfer-encoding": the Router sets`},
		{"upgrade", `cannot include "upgrade": the Router sets`},
		{"proxy-connection", `cannot include "proxy-connection": the Router sets`},
		{"Host", `cannot include "Host": the Router sets`},
		{"content-length", `cannot include "content-length": the Router sets`},
		{"content-type", `cannot include "content-type": the Router sets`},
		{"x-vsr-selected-model", `cannot include "x-vsr-selected-model": x-vsr- and x-envoy- headers belong to the Router and Envoy`},
		{"X-Envoy-Max-Retries", `cannot include "X-Envoy-Max-Retries": x-vsr- and x-envoy- headers belong`},
		{"x-authz-user-id", `cannot include identity header "x-authz-user-id", which the Router keeps from upstreams`},
	}
	for _, refusal := range refusals {
		for _, tc := range []struct {
			setting                    string
			global, recipe, deployment []string
			path                       string
		}{
			{"global fallback", []string{refusal.name}, allowed, allowed, "global.router.fallback: forward_request_headers "},
			{"recipe fallback", allowed, []string{refusal.name}, allowed, "recipes[default].fallback: forward_request_headers "},
			{"deployment", allowed, allowed, []string{refusal.name}, "global.model_catalog.deployments.gateway-d3: forward_request_headers "},
		} {
			_, err := ParseYAMLBytes(forwardHeadersDocument(tc.global, tc.recipe, tc.deployment))
			if err == nil || !strings.Contains(err.Error(), tc.path) || !strings.Contains(err.Error(), refusal.want) {
				t.Errorf("%s lists %q: got %v, want %q after %q", tc.setting, refusal.name, err, refusal.want, tc.path)
			}
		}
	}

	for _, tc := range []struct {
		setting                    string
		global, recipe, deployment []string
		want                       string
	}{
		{"a backend's own auth header", []string{"x-gateway-key"}, allowed, allowed, `global.router.fallback: forward_request_headers cannot include credential header "x-gateway-key"`},
		{"the deployment's own auth header", allowed, allowed, []string{"X-DECISION-KEY"}, `global.model_catalog.deployments.gateway-d3: forward_request_headers cannot include credential header "X-DECISION-KEY"`},
		{"a name twice", allowed, []string{"x-user-id", "X-User-Id"}, allowed, `recipes[default].fallback: forward_request_headers lists "X-User-Id" twice`},
	} {
		if _, err := ParseYAMLBytes(forwardHeadersDocument(tc.global, tc.recipe, tc.deployment)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want %q", tc.setting, err, tc.want)
		}
	}
}

// Only a remote API's calls are made for a request with its headers; the
// model runtime is the Router's own.
func TestForwardRequestHeadersAppliesToRemoteDeployments(t *testing.T) {
	t.Setenv("FORWARD_TEST_GATEWAY_KEY", "gateway-key")
	t.Setenv("FORWARD_TEST_DECISION_KEY", "decision-key")
	document := strings.Replace(string(forwardHeadersDocument(nil, nil, nil)), "    deployments:\n", `    deployments:
      local-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        forward_request_headers: [x-user-id]
`, 1)
	want := "global.model_catalog.deployments.local-kai: forward_request_headers applies to systemone, openai_chat and openai_embeddings deployments"
	if _, err := ParseYAMLBytes([]byte(document)); err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("got %v, want %q", err, want)
	}
}
