package extproc

import (
	"encoding/json"
	"fmt"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Each client format states a reasoning preference in its own fields; an
// output limit alone states none.
func TestClientStatesReasoningPreferencePerFormat(t *testing.T) {
	const messages = `"messages":[{"role":"user","content":"hi"}]`
	for _, test := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
		states bool
	}{
		{"messages without thinking", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":100,` + messages + `}`, false},
		{"messages effort without thinking", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":100,"output_config":{"effort":"low"},` + messages + `}`, false},
		{"messages adaptive thinking", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":100,"thinking":{"type":"adaptive"},` + messages + `}`, true},
		{"messages enabled thinking", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":2048,"thinking":{"type":"enabled","budget_tokens":1024},` + messages + `}`, true},
		{"messages disabled thinking", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":100,"thinking":{"type":"disabled"},` + messages + `}`, true},
		{"chat without reasoning", llmprotocol.OpenAIChatV1, `{"model":"m","max_completion_tokens":100,` + messages + `}`, false},
		{"chat reasoning_effort none", llmprotocol.OpenAIChatV1, `{"model":"m","max_tokens":100,"reasoning_effort":"none",` + messages + `}`, true},
		{"chat reasoning_effort minimal", llmprotocol.OpenAIChatV1, `{"model":"m","max_tokens":100,"reasoning_effort":"minimal",` + messages + `}`, true},
		{"chat reasoning budget", llmprotocol.OpenAIChatV1, `{"model":"m","max_tokens":100,"reasoning_budget_tokens":64,` + messages + `}`, true},
		{"chat enable_thinking switch", llmprotocol.OpenAIChatV1, `{"model":"m","max_tokens":100,"chat_template_kwargs":{"enable_thinking":false},` + messages + `}`, true},
		{"chat thinking switch", llmprotocol.OpenAIChatV1, `{"model":"m","max_tokens":100,"chat_template_kwargs":{"thinking":true},` + messages + `}`, true},
		{"chat other template arguments", llmprotocol.OpenAIChatV1, `{"model":"m","max_tokens":100,"chat_template_kwargs":{"custom_flag":true},` + messages + `}`, false},
		{"responses without reasoning", llmprotocol.OpenAIResponsesV1, `{"model":"m","max_output_tokens":100,"input":"hi"}`, false},
		{"responses reasoning effort", llmprotocol.OpenAIResponsesV1, `{"model":"m","max_output_tokens":100,"reasoning":{"effort":"minimal"},"input":"hi"}`, true},
		{"responses reasoning summary", llmprotocol.OpenAIResponsesV1, `{"model":"m","max_output_tokens":100,"reasoning":{"summary":"auto"},"input":"hi"}`, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(test.format, []byte(test.body))
			require.NoError(t, err)
			require.Equal(t, test.states, clientStatesReasoningPreference(request))
			limit := unstatedReasoningOutputLimit(request)
			if test.states {
				require.Nil(t, limit)
				return
			}
			require.NotNil(t, limit)
			require.EqualValues(t, 100, *limit)
		})
	}
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, []byte(`{"model":"m",`+messages+`}`))
	require.NoError(t, err)
	require.Nil(t, unstatedReasoningOutputLimit(request), "a request without an output limit")
}

func reasoningBudgetRouter(t *testing.T, backend llmprotocol.WireFormat, family string, threshold int) *OpenAIRouter {
	t.Helper()
	apiFormat, provider := config.APIFormatOpenAI, "vllm"
	switch backend {
	case llmprotocol.OpenAIResponsesV1:
		apiFormat = config.APIFormatResponses
	case llmprotocol.AnthropicMessagesV1:
		apiFormat, provider = config.APIFormatAnthropic, "anthropic"
	}
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`
version: v0.5
providers:
  defaults:
    model: worker
    reasoning_min_output_tokens: %d
  models:
    - name: worker
      provider_model_id: worker
      api_format: %s
      reasoning:
        family: %s
      backend_refs: [{name: b, provider: %s, endpoint: "https://api.example.com/v1"}]
recipes:
  - name: default
    decisions:
      - name: thinking
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: worker, use_reasoning: true}]
global:
  router:
    headers:
      debug: any
`, threshold, apiFormat, family, provider)))
	require.NoError(t, err)
	router, err := buildOpenAIRouterFromConfig(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = router.Close() })
	router.CredentialResolver = newTestCredentialResolver(cfg)
	return router
}

// routeReasoningRequest runs a client request through the request phases and
// returns the body the backend receives and the request context.
func routeReasoningRequest(t *testing.T, router *OpenAIRouter, client llmprotocol.WireFormat, body string) (map[string]json.RawMessage, *RequestContext) {
	t.Helper()
	path := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: "/v1/chat/completions", llmprotocol.OpenAIResponsesV1: "/v1/responses", llmprotocol.AnthropicMessagesV1: "/v1/messages",
	}[client]
	ctx := &RequestContext{Headers: map[string]string{}}
	require.NoError(t, router.processRequestHeaders(NewMockStream(nil), &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"}, {Key: ":path", Value: path}, {Key: headers.VSRDebug, Value: "true"},
		}}},
	}, ctx))
	stream := NewMockStream(nil)
	require.NoError(t, router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx))
	require.Len(t, stream.Responses, 1)
	require.Nil(t, stream.Responses[0].GetImmediateResponse(), "the request was refused")
	var sent map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(stream.Responses[0].GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &sent))
	return sent, ctx
}

// traceReasoning runs the backend's answer through the response headers phase
// and returns the reasoning member of x-vsr-trace.
func traceReasoning(t *testing.T, router *OpenAIRouter, ctx *RequestContext) string {
	t.Helper()
	stream := NewMockStream(nil)
	require.NoError(t, router.processResponseHeaders(stream, &ext_proc.ProcessingRequest{}, &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: "200"}, {Key: "content-type", Value: "application/json"},
		}}},
	}, ctx))
	require.Len(t, stream.Responses, 1)
	set := headerValuesByName(stream.Responses[0].GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders())
	return traceMembersOf(t, set[headers.VSRTrace])[headers.TraceReasoning]
}

// sentReasoning reads the reasoning control each backend format received:
// the qwen3 chat template switch on Chat and Responses, the thinking type on
// Messages.
func sentReasoning(t *testing.T, backend llmprotocol.WireFormat, sent map[string]json.RawMessage) string {
	t.Helper()
	if backend == llmprotocol.AnthropicMessagesV1 {
		var thinking struct {
			Type string `json:"type"`
		}
		require.NoError(t, json.Unmarshal(sent["thinking"], &thinking))
		return thinking.Type
	}
	var kwargs map[string]bool
	require.NoError(t, json.Unmarshal(sent["chat_template_kwargs"], &kwargs))
	return fmt.Sprintf("enable_thinking=%t", kwargs["enable_thinking"])
}

// A fallback candidate runs with the reasoning its lane chose unless the
// request's output limit is too small for it, as the primary does.
func TestFallbackCandidateReasoningFollowsTheOutputLimit(t *testing.T) {
	router := reasoningBudgetRouter(t, llmprotocol.OpenAIChatV1, "qwen3", 512)
	small, large := int64(100), int64(1000)
	for _, test := range []struct {
		name  string
		limit *int64
		want  bool
	}{{"small limit", &small, false}, {"large limit", &large, true}, {"preference or no limit", nil, true}} {
		ctx := &RequestContext{
			VSREligibleModelRefs:         []config.ModelRef{{Model: "worker", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(true)}}},
			unstatedReasoningOutputLimit: test.limit,
		}
		require.Equal(t, test.want, router.candidateReasoningChoice(ctx, "worker"), test.name)
	}
}

// A decision lane turns reasoning on, and the backend counts reasoning against
// the output limit. A request that states no reasoning preference and sets a
// limit below providers.defaults.reasoning_min_output_tokens reaches the
// backend with the model's reasoning-off control, and the trace and Router
// Replay say so. A larger or absent limit, an explicit preference, or an unset
// threshold keeps the reasoning the decision chose.
func TestSmallOutputLimitTurnsTheDecisionsReasoningOff(t *testing.T) {
	const question = `"messages":[{"role":"user","content":"What is 2+2?"}]`
	messages, chat, responses := llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1
	for _, test := range []struct {
		name      string
		client    llmprotocol.WireFormat
		backend   llmprotocol.WireFormat
		family    string
		threshold int
		body      string
		want      string
		off       bool
	}{
		{
			"messages without thinking, small limit", messages, chat, "qwen3", 512,
			`{"model":"vllm-sr/auto","max_tokens":100,` + question + `}`, "enable_thinking=false", true,
		},
		{
			"messages without thinking, large limit", messages, chat, "qwen3", 512,
			`{"model":"vllm-sr/auto","max_tokens":1000,` + question + `}`, "enable_thinking=true", false,
		},
		{
			"messages at the threshold", messages, chat, "qwen3", 512,
			`{"model":"vllm-sr/auto","max_tokens":512,` + question + `}`, "enable_thinking=true", false,
		},
		{
			"messages adaptive thinking, small limit", messages, chat, "qwen3", 512,
			`{"model":"vllm-sr/auto","max_tokens":100,"thinking":{"type":"adaptive"},` + question + `}`, "enable_thinking=true", false,
		},
		{
			"messages enabled thinking, limit below the threshold", messages, chat, "qwen3", 4096,
			`{"model":"vllm-sr/auto","max_tokens":2048,"thinking":{"type":"enabled","budget_tokens":1024},` + question + `}`, "enable_thinking=true", false,
		},
		{
			"messages without thinking, threshold unset", messages, chat, "qwen3", 0,
			`{"model":"vllm-sr/auto","max_tokens":100,` + question + `}`, "enable_thinking=true", false,
		},
		{
			"messages to a family that cannot turn reasoning off", messages, chat, "qwen3-always-on", 512,
			`{"model":"vllm-sr/auto","max_tokens":100,` + question + `}`, "enable_thinking=true", false,
		},
		{
			"messages to a messages backend", messages, messages, "claude-effort-adaptive", 512,
			`{"model":"vllm-sr/auto","max_tokens":100,` + question + `}`, "disabled", true,
		},
		{
			"chat with reasoning_effort, small limit", chat, chat, "qwen3", 512,
			`{"model":"vllm-sr/auto","max_tokens":100,"reasoning_effort":"low",` + question + `}`, "enable_thinking=true", false,
		},
		{
			"chat with a thinking switch, small limit", chat, chat, "qwen3", 512,
			`{"model":"vllm-sr/auto","max_tokens":100,"chat_template_kwargs":{"enable_thinking":true},` + question + `}`, "enable_thinking=true", false,
		},
		{
			"chat without an output limit", chat, chat, "qwen3", 512,
			`{"model":"vllm-sr/auto",` + question + `}`, "enable_thinking=true", false,
		},
		{
			"chat without reasoning, small limit", chat, chat, "qwen3", 512,
			`{"model":"vllm-sr/auto","max_completion_tokens":100,` + question + `}`, "enable_thinking=false", true,
		},
		{
			"responses without reasoning, small limit", responses, responses, "qwen3", 512,
			`{"model":"vllm-sr/auto","store":false,"max_output_tokens":100,"input":"What is 2+2?"}`, "enable_thinking=false", true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := reasoningBudgetRouter(t, test.backend, test.family, test.threshold)
			sent, ctx := routeReasoningRequest(t, router, test.client, test.body)
			require.Equal(t, test.want, sentReasoning(t, test.backend, sent), "backend body: %v", sent)
			wantTrace, wantReplay := "on", "on"
			if test.off {
				wantTrace, wantReplay = "off;reason=output_budget", "off"
			}
			require.Equal(t, wantTrace, traceReasoning(t, router, ctx))
			require.Equal(t, wantReplay, buildReplayRoutingRecord(ctx, "vllm-sr/auto", "worker", "thinking").ReasoningMode)
		})
	}
}
