package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOpenRouterReasoningDetailsSurviveRouterResponseAndDispatch(t *testing.T) {
	details := `[{"type":"reasoning.text","text":"thought","signature":"sig","id":"r1","format":"anthropic-claude-v1","index":0},{"type":"reasoning.encrypted","data":"opaque","id":null,"format":"openai-responses-v1","index":1}]`
	toolCall := `{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}`
	router := &OpenAIRouter{}
	for _, streamed := range []bool{false, true} {
		t.Run(map[bool]string{false: "buffered", true: "streamed"}[streamed], func(t *testing.T) {
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1, RequestModel: "public", TraceContext: t.Context(), SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public", Stream: streamed}}
			if streamed {
				body := `data: {"id":"r","model":"m","choices":[{"index":0,"delta":{"reasoning_details":` + details + `},"finish_reason":null}]}` + "\n\n" +
					`data: {"id":"r","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}` + "\n\n" +
					`data: {"id":"r","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\ndata: [DONE]\n\n"
				result := router.handleSemanticStreamingResponseBody([]byte(body), true, ctx)
				require.Nil(t, result.GetImmediateResponse())
				require.False(t, ctx.StreamingAborted)
			} else {
				body := []byte(`{"id":"r","model":"m","provider":"OpenRouter","choices":[{"index":0,"message":{"role":"assistant","reasoning_details":` + details + `,"tool_calls":[` + toolCall + `]},"finish_reason":"tool_calls"}]}`)
				result := router.handleNonStreamingResponseBody(body, ctx, 0)
				require.Nil(t, result.GetImmediateResponse())
				require.Contains(t, string(result.GetResponseBody().GetResponse().GetBodyMutation().GetBody()), "reasoning_details")
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
					Details json.RawMessage `json:"reasoning_details"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(upstream, &wire))
			require.JSONEq(t, details, string(wire.Messages[1].Details))
		})
	}
}
