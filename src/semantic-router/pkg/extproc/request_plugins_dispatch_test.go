//go:build !windows

package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// dispatchPluginsConfig routes every request that says "route me" to model m
// on the given wire format, with the decision's plugins as YAML.
func dispatchPluginsConfig(apiFormat, provider, endpoint, plugins string) string {
	return fmt.Sprintf(`
version: v0.5
providers:
  defaults:
    model: m
  models:
    - name: m
      provider_model_id: m
      api_format: %s
      backend_refs:
        - name: b
          provider: %s
          endpoint: %s
recipes:
  - name: default
    signals:
      keywords:
        - name: kw
          operator: OR
          keywords: ["route me"]
    decisions:
      - name: plugins_route
        rules:
          operator: AND
          conditions:
            - type: keyword
              name: kw
        modelRefs:
          - model: m
        plugins:
%s
`, apiFormat, provider, endpoint, plugins)
}

func dispatchPluginsRouter(t *testing.T, document string) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(document))
	require.NoError(t, err)
	router, err := buildOpenAIRouterFromConfig(cfg)
	require.NoError(t, err)
	if router.CredentialResolver == nil {
		router.CredentialResolver = newTestCredentialResolver(cfg)
	}
	return router
}

// sendThroughPlugins runs the request headers and body phases and returns the
// last processing response with the request context.
func sendThroughPlugins(t *testing.T, router *OpenAIRouter, path, body string) (*ext_proc.ProcessingResponse, *RequestContext) {
	t.Helper()
	ctx := &RequestContext{Headers: map[string]string{}}
	require.NoError(t, router.processRequestHeaders(NewMockStream(nil), newRequestHeaders("POST", path), ctx))
	stream := NewMockStream(nil)
	require.NoError(t, router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx))
	require.NotEmpty(t, stream.Responses)
	return stream.Responses[len(stream.Responses)-1], ctx
}

// forwardedJSON is the provider-bound body of a forwarded request, decoded.
func forwardedJSON(t *testing.T, response *ext_proc.ProcessingResponse) map[string]any {
	t.Helper()
	if immediate := response.GetImmediateResponse(); immediate != nil {
		t.Fatalf("answered with %d instead of forwarded: %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var decoded map[string]any
	require.NoError(t, json.Unmarshal(body, &decoded), "forwarded body: %s", body)
	return decoded
}

func TestRequestParamsCapLowersAnAnthropicThinkingBudgetThatNoLongerFits(t *testing.T) {
	router := dispatchPluginsRouter(t, dispatchPluginsConfig("anthropic", "anthropic", "http://127.0.0.1:8000", `          - type: request_params
            configuration:
              max_tokens_limit: 4096`))
	for _, tc := range []struct {
		name       string
		budget     int
		wantBudget float64
	}{
		{name: "budget above the cap", budget: 8000, wantBudget: 4095},
		{name: "budget equal to the cap", budget: 4096, wantBudget: 4095},
		{name: "budget below the cap is kept", budget: 2048, wantBudget: 2048},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response, _ := sendThroughPlugins(t, router, "/v1/messages", fmt.Sprintf(
				`{"model":"vllm-sr/auto","max_tokens":16000,"thinking":{"type":"enabled","budget_tokens":%d},"messages":[{"role":"user","content":"route me"}]}`,
				tc.budget))
			body := forwardedJSON(t, response)
			require.Equal(t, float64(4096), body["max_tokens"])
			require.Equal(t, map[string]any{"type": "enabled", "budget_tokens": tc.wantBudget}, body["thinking"])
		})
	}

	// A cap at Anthropic's 1024 minimum leaves no valid budget below it.
	tight := dispatchPluginsRouter(t, dispatchPluginsConfig("anthropic", "anthropic", "http://127.0.0.1:8000", `          - type: request_params
            configuration:
              max_tokens_limit: 1024`))
	response, _ := sendThroughPlugins(t, tight, "/v1/messages",
		`{"model":"vllm-sr/auto","max_tokens":16000,"thinking":{"type":"enabled","budget_tokens":8000},"messages":[{"role":"user","content":"route me"}]}`)
	require.NotNil(t, response.GetImmediateResponse())
	require.EqualValues(t, http.StatusBadRequest, response.GetImmediateResponse().GetStatus().GetCode())
	require.JSONEq(t, `{"type":"error","error":{"type":"invalid_request_error","message":"Anthropic reasoning budget must be at least 1024 and less than max_tokens"}}`,
		string(response.GetImmediateResponse().GetBody()))
}

func TestToolsPolicyDropsANamedToolChoiceForABlockedTool(t *testing.T) {
	router := dispatchPluginsRouter(t, dispatchPluginsConfig("openai", "vllm", "http://127.0.0.1:8000/v1", `          - type: tools
            configuration:
              enabled: true
              policy:
                mode: filtered
                block_tools: [admin_delete]`))
	tools := `[{"type":"function","function":{"name":"admin_delete","parameters":{"type":"object","properties":{}}}},` +
		`{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]`
	request := func(choice string) string {
		return `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"route me"}],"tools":` + tools +
			`,"tool_choice":{"type":"function","function":{"name":"` + choice + `"}}}`
	}

	body := forwardedJSON(t, first(sendThroughPlugins(t, router, "/v1/chat/completions", request("admin_delete"))))
	require.Equal(t, []string{"lookup"}, forwardedToolNames(t, body))
	require.NotContains(t, body, "tool_choice")

	body = forwardedJSON(t, first(sendThroughPlugins(t, router, "/v1/chat/completions", request("lookup"))))
	require.Equal(t, []string{"lookup"}, forwardedToolNames(t, body))
	require.Equal(t, map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}}, body["tool_choice"])
}

func first(response *ext_proc.ProcessingResponse, _ *RequestContext) *ext_proc.ProcessingResponse {
	return response
}

func forwardedToolNames(t *testing.T, body map[string]any) []string {
	t.Helper()
	var names []string
	for _, tool := range body["tools"].([]any) {
		names = append(names, tool.(map[string]any)["function"].(map[string]any)["name"].(string))
	}
	return names
}

func TestPromptCacheMarksTheFallbackRequest(t *testing.T) {
	router := dispatchPluginsRouter(t, `
version: v0.5
providers:
  defaults:
    model: primary
  models:
    - name: primary
      provider_model_id: primary
      api_format: anthropic
      backend_refs:
        - name: primary-backend
          provider: anthropic
          endpoint: http://127.0.0.1:8001
    - name: backup
      provider_model_id: backup
      api_format: anthropic
      backend_refs:
        - name: backup-backend
          provider: anthropic
          endpoint: http://127.0.0.1:8002
recipes:
  - name: default
    fallback:
      enabled: true
      max_attempts: 2
    signals:
      keywords:
        - name: kw
          operator: OR
          keywords: ["route me"]
    decisions:
      - name: plugins_route
        rules:
          operator: AND
          conditions:
            - type: keyword
              name: kw
        modelRefs:
          - model: primary
          - model: backup
        plugins:
          - type: prompt_cache
            configuration:
              enabled: true
              targets: [instructions]
`)
	response, ctx := sendThroughPlugins(t, router, "/v1/messages",
		`{"model":"vllm-sr/auto","max_tokens":64,"system":"stable instructions","messages":[{"role":"user","content":"route me"}]}`)
	require.Equal(t, "primary", ctx.VSRSelectedModel)
	require.Equal(t, map[string]any{"type": "ephemeral", "ttl": "5m"},
		cacheControlOfSystem(t, response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()),
		"the primary request carries the marker")

	var fallbackBody []byte
	router.fallbackCaller = func(_ context.Context, model string, body []byte, _ map[string]string) ([]byte, int, error) {
		if model != "backup" {
			return nil, http.StatusInternalServerError, fmt.Errorf("unexpected fallback model %q", model)
		}
		fallbackBody = body
		return []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"backup","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}`), http.StatusOK, nil
	}
	answer, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "503"}}}},
	}, ctx)
	require.NoError(t, err)
	require.NotNil(t, answer.GetImmediateResponse(), "the backup should answer")
	require.Equal(t, map[string]any{"type": "ephemeral", "ttl": "5m"}, cacheControlOfSystem(t, fallbackBody),
		"the fallback request carries the marker too")
}

// cacheControlOfSystem is the marker on the last system block of a Messages
// body, or nil when the system prompt is a plain string without one.
func cacheControlOfSystem(t *testing.T, body []byte) any {
	t.Helper()
	var wire struct {
		System any `json:"system"`
	}
	require.NoError(t, json.Unmarshal(body, &wire), "body: %s", body)
	blocks, ok := wire.System.([]any)
	if !ok || len(blocks) == 0 {
		return nil
	}
	return blocks[len(blocks)-1].(map[string]any)["cache_control"]
}
