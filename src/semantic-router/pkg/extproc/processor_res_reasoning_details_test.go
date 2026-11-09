package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestOpenRouterReasoningDetailsSurviveRouterResponseAndDispatch(t *testing.T) {
	details := `[{"type":"reasoning.text","text":"thought","signature":"sig","id":"r1","format":"anthropic-claude-v1","index":0},{"type":"reasoning.encrypted","data":"opaque","id":null,"format":"openai-responses-v1","index":1}]`
	toolCall := `{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}`
	router := &OpenAIRouter{}
	for _, streamed := range []bool{false, true} {
		t.Run(map[bool]string{false: "buffered", true: "streamed"}[streamed], func(t *testing.T) {
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1, RequestModel: "public", TraceContext: t.Context(), SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public", Stream: streamed}}
			if streamed {
				body := `data: {"id":"r","model":"m","choices":[{"index":0,"delta":{"reasoning":"thought","reasoning_details":` + details + `},"finish_reason":null}]}` + "\n\n" +
					`data: {"id":"r","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}` + "\n\n" +
					`data: {"id":"r","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\ndata: [DONE]\n\n"
				result := router.handleSemanticStreamingResponseBody([]byte(body), true, ctx)
				require.Nil(t, result.GetImmediateResponse())
				require.False(t, ctx.StreamingAborted)
			} else {
				body := []byte(`{"id":"r","model":"m","provider":"OpenRouter","choices":[{"index":0,"message":{"role":"assistant","reasoning":"thought","reasoning_details":` + details + `,"tool_calls":[` + toolCall + `]},"finish_reason":"tool_calls"}]}`)
				result := router.handleNonStreamingResponseBody(body, ctx, 0)
				require.Nil(t, result.GetImmediateResponse())
				reply := string(result.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
				require.Contains(t, reply, `"reasoning_details"`)
				require.Contains(t, reply, `"reasoning_content":"thought"`, "clients render the plain reasoning text")
			}
			require.NotNil(t, ctx.SemanticResponse)
			engine, err := router.protocolEngine()
			require.NoError(t, err)
			response, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, *ctx.SemanticResponse, llmprotocol.Envelope{})
			require.NoError(t, err)
			var reply struct {
				Choices []struct {
					Message json.RawMessage `json:"message"`
				} `json:"choices"`
			}
			require.NoError(t, json.Unmarshal(response.Body, &reply))
			followUp := []byte(`{"model":"public","messages":[{"role":"user","content":"lookup"},` + string(reply.Choices[0].Message) + `,{"role":"tool","tool_call_id":"call-1","content":"result"}]}`)
			requestCtx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1, TraceContext: t.Context()}
			request, immediate := router.prepareProtocolRequest(followUp, requestCtx)
			require.Nil(t, immediate)
			request.Model = "selected-model"
			request.Generation++
			upstream, err := router.encodeDispatchRequest(requestCtx)
			require.NoError(t, err)
			var wire struct {
				Messages []struct {
					Details   json.RawMessage `json:"reasoning_details"`
					Reasoning *string         `json:"reasoning_content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(upstream, &wire))
			require.JSONEq(t, details, string(wire.Messages[1].Details))
			require.Nil(t, wire.Messages[1].Reasoning, "returned history must not forward the text twice")
		})
	}
}

// A Messages client of an OpenRouter backend gets the reasoning text and a
// visible omission instead of an error for a reply the provider already made.
func TestOpenRouterReasoningDetailsDegradeForMessagesClient(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1, RequestModel: "public", TraceContext: t.Context()}
	body := []byte(`{"id":"r","model":"m","provider":"OpenRouter","choices":[{"index":0,"message":{"role":"assistant","content":"answer","reasoning":"thought","reasoning_details":[{"type":"reasoning.text","text":"thought","signature":"sig","format":"anthropic-claude-v1","index":0},{"type":"reasoning.encrypted","data":"opaque","format":"anthropic-claude-v1","index":1}]},"finish_reason":"stop"}]}`)
	result := router.handleNonStreamingResponseBody(body, ctx, 0)
	require.Nil(t, result.GetImmediateResponse())
	reply := string(result.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	require.Contains(t, reply, `"thinking":"thought"`)
	require.NotContains(t, reply, "opaque")
	require.Contains(t, protocolDiagnosticsHeader(result.GetResponseBody().GetResponse().GetHeaderMutation()), ";reasoning_details")
}

func reasoningContinuityHistory(t *testing.T) llmprotocol.Request {
	t.Helper()
	body := []byte(`{"model":"model","messages":[{"role":"user","content":"hello"},{"role":"assistant","reasoning_details":[{"type":"reasoning.encrypted","data":"c2ln","id":"call-1","index":0}],"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call-1","content":"result"}]}`)
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	require.NoError(t, err)
	return request
}

// Internal Looper calls are built from openai-go parameters, which drop the
// blocks, so a tool turn that carries them fails before any call is built.
func TestLooperRefusesReasoningContinuityHistory(t *testing.T) {
	request := reasoningContinuityHistory(t)
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	decision := &config.Decision{Name: "panel", ModelRefs: []config.ModelRef{{Model: "model"}}, Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM}}
	built, rejected := router.buildLooperRequest(&request, decision, &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, TraceContext: t.Context()})
	require.Nil(t, built)
	require.EqualValues(t, 400, rejected.GetImmediateResponse().GetStatus().GetCode())
}

// The recovery followup keeps only the retrieval calls and is sent through a
// Looper client, so it must fail without sending a request that lacks the blocks.
func TestContextRecoveryRefusesReasoningContinuity(t *testing.T) {
	for name, test := range map[string]struct {
		history bool
		message string
	}{
		"history":   {history: true, message: `"content":null`},
		"this_turn": {message: `"content":null,"reasoning_details":[{"type":"reasoning.encrypted","data":"c2ln","id":"retrieve-1","index":0}]`},
	} {
		t.Run(name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				writer.WriteHeader(http.StatusInternalServerError)
			}))
			defer server.Close()
			decision := &config.Decision{
				Name: "recoverable",
				Plugins: []config.DecisionPlugin{{
					Type: config.DecisionPluginContextCompression,
					Configuration: config.MustStructuredPayload(map[string]interface{}{
						"enabled": true, "recovery": map[string]interface{}{"enabled": true, "store": "redis"},
					}),
				}},
			}
			router := &OpenAIRouter{
				Config:              &config.RouterConfig{Looper: config.LooperConfig{Endpoint: server.URL}},
				CompressionRecovery: &recoveryStoreStub{},
			}
			semantic := testNeutralRequest("model", "question")
			if test.history {
				history := reasoningContinuityHistory(t)
				semantic = &history
			}
			ctx := &RequestContext{
				RequestID: "request-1", RequestModel: "model", VSRSelectedModel: "model",
				VSRSelectedDecisionName: decision.Name, VSRSelectedDecision: decision,
				ContextCompressionRecoveryKeys: []string{"key-1"},
				Headers:                        map[string]string{"x-authz-user-id": "user-1"},
				SourceFormat:                   llmprotocol.OpenAIChatV1, SemanticRequest: semantic,
			}
			scope := router.contextCompressionScope(ctx)
			router.CompressionRecovery.(*recoveryStoreStub).entries = map[string]contextcompression.RecoveryEntry{
				scope + ":key-1": {Key: "key-1", Scope: scope, Content: "full original context"},
			}
			response := []byte(`{"id":"initial","object":"chat.completion","created":1,"model":"model","usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13},"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant",` + test.message + `,"tool_calls":[{"id":"retrieve-1","type":"function","function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"key-1\"}"}}]}}]}`)
			_, err := router.handleContextRecoveryFollowup(context.Background(), response, ctx)
			require.ErrorContains(t, err, "reasoning continuity")
			require.Zero(t, calls.Load(), "no followup may be sent without the blocks")
		})
	}
}

// Chat-to-Chat requests replay the client bytes when nothing changed them, so a
// follow-up that returns the text alias beside the blocks must be re-encoded.
func TestOpenRouterReasoningFollowUpForwardsOnlyTheBlocks(t *testing.T) {
	details := `[{"type":"reasoning.text","text":"thought","signature":"sig","index":0},{"type":"reasoning.encrypted","data":"c2ln","id":"call-1","index":0}]`
	toolCalls := `[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]`
	for _, alias := range []string{"reasoning_content", "reasoning"} {
		t.Run(alias, func(t *testing.T) {
			followUp := []byte(`{"model":"public","messages":[{"role":"user","content":"lookup"},{"role":"assistant","` + alias + `":"thought","reasoning_details":` + details + `,"tool_calls":` + toolCalls + `},{"role":"tool","tool_call_id":"call-1","content":"result"}]}`)
			router := &OpenAIRouter{}
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1, TraceContext: t.Context()}
			_, immediate := router.prepareProtocolRequest(followUp, ctx)
			require.Nil(t, immediate)
			upstream, err := router.encodeDispatchRequest(ctx)
			require.NoError(t, err)
			var wire struct {
				Messages []map[string]json.RawMessage `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(upstream, &wire))
			require.JSONEq(t, details, string(wire.Messages[1]["reasoning_details"]))
			require.JSONEq(t, toolCalls, string(wire.Messages[1]["tool_calls"]))
			require.NotContains(t, wire.Messages[1], "reasoning")
			require.NotContains(t, wire.Messages[1], "reasoning_content")
		})
	}
	plain := []byte(`{"model":"public","messages":[{"role":"user","content":"lookup"}]}`)
	ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1, TraceContext: t.Context()}
	_, immediate := (&OpenAIRouter{}).prepareProtocolRequest(plain, ctx)
	require.Nil(t, immediate)
	require.NotEmpty(t, ctx.ProtocolEnvelope.Request, "a request without dropped text keeps its replay")
}
