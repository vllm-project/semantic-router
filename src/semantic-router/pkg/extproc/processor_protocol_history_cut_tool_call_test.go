package extproc

import (
	"encoding/json"
	"fmt"
	"slices"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const sendEmailParameters = `{"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}`

// cutCallHistories are a client's next request after the output limit cut a
// send_email call, kept as the Router passed it on: the model, the call's
// arguments, JSON-encoded, and what follows the call in its turn.
var cutCallHistories = map[llmprotocol.WireFormat]string{
	llmprotocol.OpenAIChatV1: `{"model":%q,"messages":[` +
		`{"role":"user","content":"Tell the team the review moves to ten."},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"send_email","arguments":%s}}%s]},` +
		`{"role":"tool","tool_call_id":"call_1","content":"The email was cut off; nothing was sent."},` +
		`{"role":"user","content":"Send a shorter one."}],` +
		`"tools":[{"type":"function","function":{"name":"send_email","parameters":` + sendEmailParameters + `}}]}`,
	llmprotocol.OpenAIResponsesV1: `{"model":%q,"store":false,"input":[` +
		`{"type":"message","role":"user","content":"Tell the team the review moves to ten."},` +
		`{"type":"function_call","call_id":"call_1","name":"send_email","arguments":%s}%s,` +
		`{"type":"function_call_output","call_id":"call_1","output":"The email was cut off; nothing was sent."},` +
		`{"type":"message","role":"user","content":"Send a shorter one."}],` +
		`"tools":[{"type":"function","name":"send_email","parameters":` + sendEmailParameters + `}]}`,
}

// A streamed call the output limit cut ends as the backend left it, inside a
// string here, and the client sends it back so.
const cutCallArguments = `{"to": "team@example.com", "body": "Dear team, the review moves to ten.`

func dispatchCutCallHistory(t *testing.T, client, backend llmprotocol.WireFormat, later string) (*ext_proc.ProcessingResponse, *RequestContext, *OpenAIRouter) {
	t.Helper()
	router, model := routingTestRouterForFormat(backend)
	router.Cache = cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false})
	ctx := &RequestContext{Headers: map[string]string{}, RequestID: "history-cut-tool-call", SourceFormat: client}
	arguments, err := json.Marshal(cutCallArguments)
	require.NoError(t, err)
	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(fmt.Sprintf(cutCallHistories[client], model, arguments, later))},
	}, ctx)
	require.NoError(t, err)
	return response, ctx, router
}

// A Chat or Responses client keeps the call an output limit cut in the history
// of its next request, as agents do. The Router refused that request before
// routing with invalid_tool_call; as the last call of its assistant turn the
// call is now accepted, and every backend receives the members completed
// before the cut. The request's diagnostics report the closure.
func TestHistoryCallCutByTheOutputLimitReachesEveryBackendClosed(t *testing.T) {
	for client := range cutCallHistories {
		for _, backend := range extProcMatrixFormats {
			t.Run(string(client)+"_client_"+string(backend)+"_backend", func(t *testing.T) {
				response, ctx, router := dispatchCutCallHistory(t, client, backend, "")
				if immediate := router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse(); immediate != nil {
					t.Fatalf("request was refused: %s", immediate.GetBody())
				}
				dispatched := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
				require.NotContains(t, string(dispatched), "Dear team", "the backend received the cut arguments")
				sent, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(backend, dispatched)
				require.NoError(t, err, string(dispatched))
				var arguments []string
				for _, message := range sent.Messages {
					for _, content := range message.Content {
						if content.ToolCall != nil {
							arguments = append(arguments, content.ToolCall.Arguments)
						}
					}
				}
				require.Len(t, arguments, 1, string(dispatched))
				require.JSONEq(t, `{"to":"team@example.com"}`, arguments[0])
				require.True(t, slices.ContainsFunc(ctx.ProtocolDiagnostics, func(diagnostic llmprotocol.Diagnostic) bool {
					return diagnostic.Field == "tool_call.arguments" && diagnostic.Action == llmprotocol.DiagnosticApproximated
				}), "diagnostics: %+v", ctx.ProtocolDiagnostics)
			})
		}
	}
}

// The client learns of the closure from the reply's x-vsr-warnings, which the
// response headers carry, buffered and streamed alike.
func TestHistoryCallCutByTheOutputLimitIsReportedInWarnings(t *testing.T) {
	response, ctx, router := dispatchCutCallHistory(t, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, "")
	require.Nil(t, router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse())
	headers, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "200"}, {Key: "content-type", Value: "application/json"}}},
	}}, ctx)
	require.NoError(t, err)
	require.Contains(t, protocolDiagnosticsHeader(headers.GetResponseHeaders().GetResponse().GetHeaderMutation()), `;field="tool_call.arguments"`)
}

// Only the last call of a turn can be the one an output limit cut, so a cut
// call that another call of its turn follows is still refused before routing.
func TestHistoryCallCutBeforeTheEndOfItsTurnIsRefused(t *testing.T) {
	later := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:      `,{"id":"call_2","type":"function","function":{"name":"send_email","arguments":"{}"}}`,
		llmprotocol.OpenAIResponsesV1: `,{"type":"function_call","call_id":"call_2","name":"send_email","arguments":"{}"}`,
	}
	for client := range cutCallHistories {
		t.Run(string(client), func(t *testing.T) {
			response, ctx, router := dispatchCutCallHistory(t, client, llmprotocol.OpenAIChatV1, later[client])
			immediate := router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse()
			require.NotNil(t, immediate, "the request was routed")
			require.EqualValues(t, 400, immediate.GetStatus().GetCode())
			require.Contains(t, string(immediate.GetBody()), `"invalid_tool_call"`)
		})
	}
}
