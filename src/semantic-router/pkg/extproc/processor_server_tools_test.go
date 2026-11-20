package extproc

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// The blocks Anthropic's web search leaves in a turn, as Claude Code 2.1.325
// receives them.
const (
	serverToolCall     = `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"current Go release"}}`
	serverToolResult   = `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","title":"Go","url":"https://go.dev/doc/devel/release","encrypted_content":"EqQBCioIBxgC","page_age":null}]}`
	serverToolCitation = `{"type":"web_search_result_location","cited_text":"go1.27.4","url":"https://go.dev/doc/devel/release","title":"Go","encrypted_index":"Eo8BCioIBxgC"}`
	serverToolCited    = `{"type":"text","text":"Go 1.27.4 is current.","citations":[` + serverToolCitation + `]}`
)

// serverToolRouter serves one model on Anthropic's API, one through an
// Anthropic-compatible gateway, one on a Chat backend and one on vLLM's
// Messages API, the last two unable to run server tools.
func serverToolRouter() *OpenAIRouter {
	model := func(endpoint, format string) config.ModelParams {
		return config.ModelParams{PreferredEndpoints: []string{endpoint}, APIFormat: format}
	}
	cfg := &config.RouterConfig{BackendModels: config.BackendModels{
		DefaultModel: "claude",
		ModelConfig: map[string]config.ModelParams{
			"claude":    model("anthropic", config.APIFormatAnthropic),
			"gateway":   model("gateway", config.APIFormatAnthropic),
			"qwen-chat": model("vllm-chat", config.APIFormatOpenAI),
			"qwen-msg":  model("vllm-msg", config.APIFormatAnthropic),
		},
		VLLMEndpoints: []config.VLLMEndpoint{
			{Name: "anthropic", Address: "127.0.0.1", Port: 8000, ProviderProfileName: "anthropic"},
			{Name: "gateway", Address: "127.0.0.1", Port: 8003, ProviderProfileName: "gateway"},
			{Name: "vllm-chat", Address: "127.0.0.1", Port: 8001, ProviderProfileName: "vllm"},
			{Name: "vllm-msg", Address: "127.0.0.1", Port: 8002, ProviderProfileName: "vllm"},
		},
		ProviderProfiles: map[string]config.ProviderProfile{
			"anthropic": {Type: "anthropic", BaseURL: "http://127.0.0.1:8000"},
			"gateway":   {Type: "anthropic-compatible", BaseURL: "http://127.0.0.1:8003"},
			"vllm":      {Type: "vllm", BaseURL: "http://127.0.0.1:8001"},
		},
	}}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
}

// serverToolRequest decodes a Messages request the way ingress does, with
// the web search tool declared, history from an earlier search, or both.
func serverToolRequest(t *testing.T, declare, history bool) *llmprotocol.Request {
	t.Helper()
	tools := `{"name":"lookup","input_schema":{"type":"object"}}`
	if declare {
		tools = `{"type":"web_search_20250305","name":"web_search","max_uses":8},` + tools
	}
	messages := `{"role":"user","content":"Which Go release is current?"}`
	if history {
		messages += `,{"role":"assistant","content":[` + serverToolCall + `,` + serverToolResult + `,` + serverToolCited + `]},` +
			`{"role":"user","content":"And the one before it?"}`
	}
	body := `{"model":"vllm-sr/auto","max_tokens":256,"tools":[` + tools + `],"messages":[` + messages + `]}`
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
	require.NoError(t, err)
	return &request
}

func dispatchedBody(t *testing.T, response *ext_proc.ProcessingResponse) string {
	t.Helper()
	body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	require.NotEmpty(t, body, "the request was not dispatched: %v", response)
	return string(body)
}

// A request that declares Anthropic's web search, with an earlier search in
// its history, reaches Anthropic's API with the tool and the blocks as sent.
func TestServerToolRequestReachesAnAnthropicBackend(t *testing.T) {
	router := serverToolRouter()
	request := serverToolRequest(t, true, true)
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)

	response, err := router.handleSpecifiedModelRouting(request, "claude", "", ctx)
	require.NoError(t, err)
	body := dispatchedBody(t, response)
	for _, sent := range []string{`"type":"web_search_20250305"`, `"max_uses":8`, `"type":"server_tool_use"`, `"encrypted_content":"EqQBCioIBxgC"`, `"encrypted_index":"Eo8BCioIBxgC"`} {
		require.Contains(t, body, sent)
	}
}

// Every backend that cannot run the tool refuses the request before it is
// sent, names the tool and why, and Router Replay keeps the refusal as the
// client got it. Selection passes over such a backend for one that runs it.
func TestServerToolRequestRoutedElsewhereIsRefusedAndRecorded(t *testing.T) {
	for _, model := range []string{"qwen-chat", "qwen-msg"} {
		t.Run(model, func(t *testing.T) {
			router := serverToolRouter()
			request := serverToolRequest(t, true, false)
			ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
			ctx.VSRSelectedDecision = &config.Decision{Name: "default"}
			ctx.VSRSelectedModel = model
			enableReplay(router, ctx)

			_, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
			var protocolError *llmprotocol.ProtocolError
			require.ErrorAs(t, err, &protocolError)
			require.Equal(t, "unsupported_server_tool", protocolError.Code)
			for _, part := range []string{`"web_search"`, "web_search_20250305", "Anthropic backend", `"` + model + `"`, "provider vllm"} {
				require.Contains(t, protocolError.Message, part)
			}

			response, err := router.requestBodyReply(nil, err, ctx)
			require.NoError(t, err)
			immediate := response.GetImmediateResponse()
			require.EqualValues(t, http.StatusBadRequest, immediate.GetStatus().GetCode())
			require.Contains(t, string(immediate.GetBody()), `"type":"invalid_request_error"`)
			require.Contains(t, string(immediate.GetBody()), "web_search_20250305")
			require.NoError(t, router.handleProcessReceiveError(ctx, io.EOF))
			requireReplayEnd(t, router, ctx, routerreplay.LifecycleFailed, "request_rejected", http.StatusBadRequest)
			record, _ := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.Equal(t, model, record.SelectedModel)
			require.Contains(t, record.ResponseBody, "web_search_20250305")
		})
	}

	router := serverToolRouter()
	request := serverToolRequest(t, true, false)
	for model, runs := range map[string]bool{"claude": true, "gateway": true, "qwen-chat": false, "qwen-msg": false} {
		err := router.candidateCapabilityMismatch(config.ModelRef{Model: model}, request, nil, nil, nil, false)
		require.Equal(t, runs, err == nil, "%s: %v", model, err)
	}
	refs := func(models ...string) *selection.SelectionContext {
		input := &selection.SelectionContext{DecisionName: "search"}
		for _, model := range models {
			input.CandidateModels = append(input.CandidateModels, config.ModelRef{Model: model})
		}
		return input
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	eligible, err := router.capabilityEligibleSelectionContext(refs("qwen-chat", "claude", "qwen-msg"), nil, ctx)
	require.NoError(t, err)
	require.Equal(t, []config.ModelRef{{Model: "claude"}}, eligible.CandidateModels)
	_, err = router.capabilityEligibleSelectionContext(refs("qwen-chat", "qwen-msg"), nil, routingTestContext(llmprotocol.AnthropicMessagesV1, request))
	var protocolError *llmprotocol.ProtocolError
	require.ErrorAs(t, err, &protocolError)
	require.Equal(t, "unsupported_server_tool", protocolError.Code)
}

// Fallback moves past a candidate that cannot run the tool, to the next one,
// which receives the tool as sent.
func TestServerToolFallbackSkipsABackendThatCannotRunIt(t *testing.T) {
	router := serverToolRouter()
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, serverToolRequest(t, true, false))
	_, result, err := router.prepareFallbackCandidate(t.Context(), &config.ModelRef{Model: "qwen-chat"}, ctx, false)
	var protocolError *llmprotocol.ProtocolError
	require.ErrorAs(t, err, &protocolError)
	require.Equal(t, "unsupported_server_tool", protocolError.Code)
	require.True(t, result.CanFallback)
	candidate, _, err := router.prepareFallbackCandidate(t.Context(), &config.ModelRef{Model: "claude"}, ctx, false)
	require.NoError(t, err)
	require.Contains(t, string(candidate.body), `"type":"web_search_20250305"`)
}

// A later turn routed to a backend that cannot read the blocks an earlier
// search left goes without them: the answer's text stays and the loss is
// reported.
func TestServerToolHistoryIsLeftOutForAnotherBackend(t *testing.T) {
	router := serverToolRouter()
	request := serverToolRequest(t, false, true)
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)

	response, err := router.handleSpecifiedModelRouting(request, "qwen-chat", "", ctx)
	require.NoError(t, err)
	body := dispatchedBody(t, response)
	require.Contains(t, body, "Go 1.27.4 is current.")
	for _, dropped := range []string{"server_tool_use", "web_search_tool_result", "encrypted"} {
		require.NotContains(t, body, dropped)
	}
	found := false
	for _, diagnostic := range ctx.ProtocolDiagnostics {
		found = found || diagnostic.Field == "content.server_tool" && diagnostic.Action == llmprotocol.DiagnosticDropped
	}
	require.True(t, found, "the dropped blocks were not reported: %+v", ctx.ProtocolDiagnostics)
}

// Anthropic's reply to a search reaches the client with every block, buffered
// and streamed, and the Router's own record of the answer keeps them.
func TestServerToolReplyReachesTheClient(t *testing.T) {
	content := `[` + serverToolCall + `,` + serverToolResult + `,` + serverToolCited + `]`
	t.Run("buffered", func(t *testing.T) {
		router := serverToolRouter()
		request := serverToolRequest(t, true, false)
		ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
		_, err := router.handleSpecifiedModelRouting(request, "claude", "", ctx)
		require.NoError(t, err)
		_, err = router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "200"}, {Key: "content-type", Value: "application/json"}}},
		}}, ctx)
		require.NoError(t, err)
		upstream := `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":` + content +
			`,"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":2908,"output_tokens":57,"server_tool_use":{"web_search_requests":1}}}`
		response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(upstream), EndOfStream: true},
		}, ctx)
		require.NoError(t, err)
		client := upstream
		if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			client = string(mutation.GetBody())
		}
		var reply struct {
			Content json.RawMessage `json:"content"`
		}
		require.NoError(t, json.Unmarshal([]byte(client), &reply), client)
		require.JSONEq(t, content, string(reply.Content))
	})
	t.Run("streamed", func(t *testing.T) {
		router := &OpenAIRouter{}
		ctx := &RequestContext{
			SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
			RequestModel: "public-model", TraceContext: t.Context(),
		}
		frame := func(event string) string {
			var typed struct {
				Type string `json:"type"`
			}
			require.NoError(t, json.Unmarshal([]byte(event), &typed))
			return "event: " + typed.Type + "\ndata: " + event + "\n\n"
		}
		body := strings.Join([]string{
			frame(`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2908,"output_tokens":16}}}`),
			frame(`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`),
			frame(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"current Go release\"}"}}`),
			frame(`{"type":"content_block_stop","index":0}`),
			frame(`{"type":"content_block_start","index":1,"content_block":` + serverToolResult + `}`),
			frame(`{"type":"content_block_stop","index":1}`),
			frame(`{"type":"content_block_start","index":2,"content_block":{"citations":[],"type":"text","text":""}}`),
			frame(`{"type":"content_block_delta","index":2,"delta":{"type":"citations_delta","citation":` + serverToolCitation + `}}`),
			frame(`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Go 1.27.4 is current."}}`),
			frame(`{"type":"content_block_stop","index":2}`),
			frame(`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":57,"server_tool_use":{"web_search_requests":1}}}`),
			frame(`{"type":"message_stop"}`),
		}, "")
		response := router.handleSemanticStreamingResponseBody([]byte(body), true, ctx)
		require.False(t, ctx.StreamingAborted)
		require.True(t, ctx.StreamingComplete)
		// A Messages client of a Messages backend gets the backend's own frames.
		client := body
		if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			client = string(mutation.GetBody())
		}
		for _, sent := range []string{`"type":"server_tool_use"`, `"partial_json":"{\"query\":\"current Go release\"}"`, `"type":"web_search_tool_result"`, `"type":"citations_delta"`, `"encrypted_index":"Eo8BCioIBxgC"`} {
			require.Contains(t, client, sent)
		}
		require.NotContains(t, client, "event: error")
		require.NotNil(t, ctx.SemanticResponse)
		var answer []llmprotocol.Content
		for _, item := range ctx.SemanticResponse.Output {
			answer = append(answer, item.Content...)
		}
		require.Len(t, answer, 3)
		require.Equal(t, llmprotocol.ContentServerTool, answer[0].Kind)
		require.JSONEq(t, serverToolCall, string(answer[0].Raw))
		require.Equal(t, llmprotocol.ContentServerTool, answer[1].Kind)
		require.JSONEq(t, `[`+serverToolCitation+`]`, string(answer[2].Raw))
	})
}

// A prompt cache marker on a server tool, the policy's or the caller's,
// reaches Anthropic on that tool and counts as the request's marker.
func TestServerToolKeepsPromptCacheMarkers(t *testing.T) {
	for name, marker := range map[string]string{"policy": "", "caller": `,"cache_control":{"type":"ephemeral"}`} {
		t.Run(name, func(t *testing.T) {
			body := `{"model":"vllm-sr/auto","max_tokens":256,"tools":[{"name":"lookup","input_schema":{"type":"object"}},` +
				`{"type":"web_search_20250305","name":"web_search","max_uses":8` + marker + `}],` +
				`"messages":[{"role":"user","content":"Which Go release is current?"}]}`
			request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
			require.NoError(t, err)
			ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
			ctx.VSRSelectedDecision = promptCacheTestDecision(map[string]interface{}{"enabled": true, "ttl": "1h"})

			response, err := serverToolRouter().handleSpecifiedModelRouting(&request, "claude", "", ctx)
			require.NoError(t, err)
			var wire promptCacheWireRequest
			require.NoError(t, json.Unmarshal([]byte(dispatchedBody(t, response)), &wire))
			require.Len(t, wire.Tools, 2)
			require.Nil(t, wire.Tools[0].CacheControl)
			require.NotNil(t, wire.Tools[1].CacheControl)
			require.Equal(t, "web_search", wire.Tools[1].Name)
			if name == "policy" {
				require.Equal(t, "1h", wire.Tools[1].CacheControl.TTL)
				assertPromptCacheReceipt(t, ctx, promptCacheActionInserted, 1, 0)
			} else {
				require.Empty(t, wire.Tools[1].CacheControl.TTL)
				assertPromptCacheReceipt(t, ctx, promptCacheActionPreserved, 0, 1)
			}
		})
	}
}

// A fallback that a search answers reaches a streaming client with the
// search's blocks, rendered from the buffered reply.
func TestServerToolFallbackReplyStreamsToTheClient(t *testing.T) {
	router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-2"})
	ctx.SemanticRequest = serverToolRequest(t, true, false)
	ctx.SemanticRequest.Stream = true
	ctx.SourceFormat, ctx.TargetFormat = llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1
	ctx.UpstreamStatusCode = http.StatusBadGateway
	ctx.ExpectStreamingResponse = true
	reply := `{"id":"msg_1","type":"message","role":"assistant","model":"model-fallback-2","content":[` +
		serverToolCall + `,` + serverToolResult + `,` + serverToolCited + `],"stop_reason":"end_turn","stop_sequence":null,` +
		`"usage":{"input_tokens":2908,"output_tokens":57,"server_tool_use":{"web_search_requests":1}}}`
	router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
		return []byte(reply), http.StatusOK, nil
	}

	immediate := router.handleUpstreamTransportError([]byte("upstream 502"), ctx).GetImmediateResponse()
	require.NotNil(t, immediate)
	require.EqualValues(t, http.StatusOK, immediate.GetStatus().GetCode(), string(immediate.GetBody()))
	body := string(immediate.GetBody())
	for _, sent := range []string{`"type":"server_tool_use"`, `"type":"web_search_tool_result"`, `"encrypted_content":"EqQBCioIBxgC"`, `"type":"citations_delta"`} {
		require.Contains(t, body, sent)
	}
	assertClientStreamDecodes(t, protocolcodec.NewBuiltinEngine(), llmprotocol.AnthropicMessagesV1, immediate.GetBody())
}
