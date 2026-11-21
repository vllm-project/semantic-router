package extproc

import (
	"context"
	"fmt"
	"strconv"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

// twoHarnessRouter serves Claude Code and Codex from one gateway: each
// harness has its own entrypoint onto the same recipe and a Chat backend.
func twoHarnessRouter(t *testing.T, global, claudeCode, codex string) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`
version: v0.5
providers:
  defaults:
    model: worker
  models:
    - name: worker
      provider_model_id: worker
      api_format: openai
      backend_refs: [{name: b, provider: openai, endpoint: "https://api.example.com/v1"}]
entrypoints:
  - model_names: [vllm-sr/claude-code]
    recipe: default
    default: true
%s
  - model_names: [vllm-sr/codex]
    recipe: default
%s
recipes:
  - name: default
    decisions:
      - name: agents
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: worker}]
global:
  router:
    allow_lossy_translation: %s
`, claudeCode, codex, global)))
	require.NoError(t, err)
	router, err := buildOpenAIRouterFromConfig(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = router.Close() })
	router.CredentialResolver = newTestCredentialResolver(cfg)
	return router
}

// deferredToolSearch is a Claude Code turn after tool search: a deferred tool
// and the tool_reference the search returned.
const deferredToolSearch = `{"model":%q,"max_tokens":64,"messages":[
	{"role":"user","content":"fetch it"},
	{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"ToolSearch","input":{"query":"select:WebFetch"}}]},
	{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"tool_reference","tool_name":"WebFetch"}]},{"type":"text","text":"Tool loaded."}]}],
	"tools":[{"name":"ToolSearch","input_schema":{"type":"object"}},{"name":"WebFetch","input_schema":{"type":"object"},"defer_loading":true}]}`

func sendMessages(t *testing.T, router *OpenAIRouter, body string) *ext_proc.ProcessingResponse {
	t.Helper()
	return sendRequest(t, router, "/v1/messages", body, &RequestContext{Headers: map[string]string{}})
}

func sendRequest(t *testing.T, router *OpenAIRouter, path, body string, ctx *RequestContext) *ext_proc.ProcessingResponse {
	t.Helper()
	require.NoError(t, router.processRequestHeaders(NewMockStream(nil), newRequestHeaders("POST", path), ctx))
	stream := NewMockStream(nil)
	require.NoError(t, router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx))
	require.Len(t, stream.Responses, 1)
	return stream.Responses[0]
}

func requireRefused(t *testing.T, reply *ext_proc.ProcessingResponse, want string) {
	t.Helper()
	immediate := reply.GetImmediateResponse()
	require.NotNil(t, immediate, "the request was dispatched")
	require.EqualValues(t, 400, immediate.GetStatus().GetCode())
	require.Contains(t, string(immediate.GetBody()), want)
}

// One gateway serves both harnesses: Claude Code's entrypoint accepts its
// deferred tools and the tool references tool search returns, while the same
// turn sent to Codex's entrypoint is still refused.
func TestEntrypointAcceptsDeferredToolsOnlyForItsHarness(t *testing.T) {
	router := twoHarnessRouter(t, "[]", "    allow_lossy_translation: [tools.defer_loading]", "")

	reply := sendMessages(t, router, fmt.Sprintf(deferredToolSearch, "vllm-sr/claude-code"))
	require.Nil(t, reply.GetImmediateResponse(), "Claude Code's entrypoint refused its deferred tools: %s", reply.GetImmediateResponse().GetBody())

	requireRefused(t, sendMessages(t, router, fmt.Sprintf(deferredToolSearch, "vllm-sr/codex")), "Anthropic content type is unsupported")
	requireRefused(t, sendMessages(t, router, `{"model":"vllm-sr/codex","max_tokens":64,"messages":[{"role":"user","content":"hi"}],`+
		`"tools":[{"name":"WebFetch","input_schema":{"type":"object"},"defer_loading":true}]}`), "tools.defer_loading")
}

// An entrypoint's list replaces the global one, so an empty list narrows a
// loss the gateway accepts for every other entrypoint.
func TestEmptyEntrypointListNarrowsTheGlobalLosses(t *testing.T) {
	router := twoHarnessRouter(t, "[tools.defer_loading]", "", "    allow_lossy_translation: []")

	reply := sendMessages(t, router, fmt.Sprintf(deferredToolSearch, "vllm-sr/claude-code"))
	require.Nil(t, reply.GetImmediateResponse(), "the global list no longer reached Claude Code's entrypoint")
	requireRefused(t, sendMessages(t, router, fmt.Sprintf(deferredToolSearch, "vllm-sr/codex")), "Anthropic content type is unsupported")
}

// A Responses request names the Codex entrypoint the same way.
func TestResponsesDeferredToolsFollowTheEntrypoint(t *testing.T) {
	router := twoHarnessRouter(t, "[]", "    allow_lossy_translation: [tools.defer_loading]", "")
	body := `{"model":%q,"input":"hi","tools":[{"type":"function","name":"WebFetch","parameters":{"type":"object"},"defer_loading":true}]}`
	for model, refused := range map[string]bool{"vllm-sr/claude-code": false, "vllm-sr/codex": true} {
		ctx := &RequestContext{Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1}
		router.resolveIngressEntrypoint([]byte(fmt.Sprintf(body, model)), ctx)
		engine, err := router.protocolEngine(ctx)
		require.NoError(t, err)
		_, _, _, err = engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(fmt.Sprintf(body, model)))
		require.Equal(t, refused, err != nil, "%s: decode error = %v", model, err)
	}
}

// Losses checked after routing follow the entrypoint too: signed thinking
// reaches a Chat backend without its signature only through the entrypoint
// that accepts reasoning.signature.
func TestSignedThinkingFollowsTheEntrypoint(t *testing.T) {
	router := twoHarnessRouter(t, "[]", "", "    allow_lossy_translation: [reasoning.signature]")
	history := func(model string) string {
		return fmt.Sprintf(`{"model":%q,"max_tokens":64,"messages":[
	{"role":"user","content":"What is 2+2?"},
	{"role":"assistant","content":[{"type":"thinking","thinking":"simple arithmetic","signature":%s},{"type":"text","text":"4"}]},
	{"role":"user","content":"And 3+3?"}]}`, model, strconv.Quote("EqQBCkYIBxgCKkBexample"))
	}
	reply := sendMessages(t, router, history("vllm-sr/codex"))
	require.Nil(t, reply.GetImmediateResponse(), "the entrypoint that accepts the loss refused it: %s", reply.GetImmediateResponse().GetBody())
	requireRefused(t, sendMessages(t, router, history("vllm-sr/claude-code")), "signed thinking")
}

// A Looper hop names a concrete model, so it takes the losses of the client
// request it is made for: every hop of a ratings decision reached through
// vllm-sr/codex accepts none, though the global list accepts deferred tools.
func TestLooperHopKeepsItsClientEntrypoint(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  defaults:
    model: model-a
  models:
    - name: model-a
      backend_refs: [{name: a, provider: vllm, endpoint: "127.0.0.1:8000"}]
    - name: model-b
      backend_refs: [{name: b, provider: vllm, endpoint: "127.0.0.1:8001"}]
entrypoints:
  - model_names: [vllm-sr/claude-code]
    recipe: default
    default: true
  - model_names: [vllm-sr/codex]
    recipe: default
    allow_lossy_translation: []
recipes:
  - name: default
    decisions:
      - name: compare
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: model-a}, {model: model-b}]
        algorithm:
          type: ratings
          on_error: skip
          ratings: {max_concurrent: 1}
global:
  router:
    allow_lossy_translation: [tools.defer_loading]
`))
	require.NoError(t, err)
	router, err := buildOpenAIRouterFromConfig(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = router.Close() })
	router.CredentialResolver = newTestCredentialResolver(cfg)
	recorder := &hopSessionRecorder{router: router}
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true
	router.hopCaller = &graph.SessionCaller{
		Engine: routing.NewEngine(recorder, opts),
		Upstream: &capturingHopUpstream{responseBody: []byte(`{"id":"hop","object":"chat.completion","created":1,"model":"model-a",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}`)},
	}
	client := &RequestContext{TraceContext: context.Background(), Headers: map[string]string{}}
	request, rejected := router.prepareProtocolRequest([]byte(`{"model":"vllm-sr/codex","messages":[{"role":"user","content":"hi"}]}`), client)
	require.Nil(t, rejected)
	require.Empty(t, router.acceptedLosses(client))

	_, err = router.handleLooperExecution(context.Background(), request, cfg.GetDecisionByName("compare"), client)
	require.NoError(t, err)
	require.NotEmpty(t, recorder.sessions, "the decision made no hop")
	for _, session := range recorder.sessions {
		require.Empty(t, router.acceptedLosses(session.ctx), "a hop took the global losses")
	}

	concrete := &RequestContext{Headers: map[string]string{}}
	router.resolveIngressEntrypoint([]byte(`{"model":"model-a"}`), concrete)
	require.Equal(t, []string{"tools.defer_loading"}, router.acceptedLosses(concrete))
}

// A cached answer is served only under the losses it was made with, so two
// entrypoints that accept different losses never share one.
func TestResponseCacheKeepsEntrypointLossesApart(t *testing.T) {
	router := twoHarnessRouter(t, "[]", "    allow_lossy_translation: [tools.defer_loading]", "")
	fingerprint := func(model string) string {
		ctx := &RequestContext{Headers: map[string]string{}, SourceFormat: llmprotocol.AnthropicMessagesV1}
		router.resolveIngressEntrypoint([]byte(`{"model":"`+model+`"}`), ctx)
		return responseCachePolicyFingerprint(router.Config, ctx)
	}
	require.NotEqual(t, fingerprint("vllm-sr/claude-code"), fingerprint("vllm-sr/codex"))

	// A global list alone keeps the keys a cache already holds.
	global := twoHarnessRouter(t, "[tools.defer_loading]", "", "")
	ctx := &RequestContext{Headers: map[string]string{}, SourceFormat: llmprotocol.AnthropicMessagesV1}
	global.resolveIngressEntrypoint([]byte(`{"model":"vllm-sr/codex"}`), ctx)
	require.Equal(t, fingerprint("vllm-sr/codex"), responseCachePolicyFingerprint(global.Config, ctx))
}

// A token count names an entrypoint like a turn, and is decoded under its
// losses.
func TestCountTokensFollowsTheEntrypoint(t *testing.T) {
	router := twoHarnessRouter(t, "[]", "    allow_lossy_translation: [tools.defer_loading]", "")
	body := `{"model":%q,"messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"name":"WebFetch","input_schema":{"type":"object"},"defer_loading":true}]}`
	count := func(model string) *ext_proc.ProcessingResponse {
		return sendRequest(t, router, "/v1/messages/count_tokens", fmt.Sprintf(body, model), &RequestContext{Headers: map[string]string{}})
	}
	require.EqualValues(t, 200, count("vllm-sr/claude-code").GetImmediateResponse().GetStatus().GetCode())
	requireRefused(t, count("vllm-sr/codex"), "tools.defer_loading")
}

// A model the listener routes elsewhere as unserved takes the losses of the
// entrypoint it is routed to.
func TestUnservedModelTakesTheLossesOfItsRoute(t *testing.T) {
	router := twoHarnessRouter(t, "[]", "    allow_lossy_translation: [tools.defer_loading]", "")
	body := fmt.Sprintf(deferredToolSearch, "claude-opus-unserved")
	for route, refused := range map[string]bool{"vllm-sr/claude-code": false, "vllm-sr/codex": true} {
		reply := sendRequest(t, router, "/v1/messages", body, &RequestContext{Headers: map[string]string{}, UnservedModelRoute: route})
		if refused {
			requireRefused(t, reply, "Anthropic content type is unsupported")
		} else {
			require.Nil(t, reply.GetImmediateResponse(), "%s refused: %s", route, reply.GetImmediateResponse().GetBody())
		}
	}
}
