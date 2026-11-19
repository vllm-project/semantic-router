package extproc

import (
	"fmt"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// outputLimitToolRouter routes every request to a Chat backend through a
// decision whose request_params cap the output at 50 tokens.
func outputLimitToolRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  defaults:
    model: worker
  models:
    - name: worker
      provider_model_id: worker
      api_format: openai
      backend_refs: [{name: b, provider: openai, endpoint: "https://api.example.com/v1"}]
recipes:
  - name: default
    decisions:
      - name: tools
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: worker}]
        plugins:
          - type: request_params
            configuration:
              max_tokens_limit: 50
`))
	require.NoError(t, err)
	router, err := buildOpenAIRouterFromConfig(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = router.Close() })
	router.CredentialResolver = newTestCredentialResolver(cfg)
	return router
}

const sendEmailSchema = `{"type":"object","properties":{"body":{"type":"string"}},"required":["body"]}`

// outputLimitToolRequests ask each client's API for a send_email call within
// 256 output tokens, which the decision caps at 50.
var outputLimitToolRequests = map[llmprotocol.WireFormat]struct{ path, body string }{
	llmprotocol.OpenAIChatV1: {"/v1/chat/completions", `{"model":"vllm-sr/auto","max_tokens":256,"stream":%t,` +
		`"tools":[{"type":"function","function":{"name":"send_email","parameters":` + sendEmailSchema + `}}],` +
		`"messages":[{"role":"user","content":"Tell the team the review moves to ten."}]}`},
	llmprotocol.OpenAIResponsesV1: {"/v1/responses", `{"model":"vllm-sr/auto","store":false,"max_output_tokens":256,"stream":%t,` +
		`"tools":[{"type":"function","name":"send_email","parameters":` + sendEmailSchema + `}],` +
		`"input":"Tell the team the review moves to ten."}`},
	llmprotocol.AnthropicMessagesV1: {"/v1/messages", `{"model":"vllm-sr/auto","max_tokens":256,"stream":%t,` +
		`"tools":[{"name":"send_email","input_schema":` + sendEmailSchema + `}],` +
		`"messages":[{"role":"user","content":"Tell the team the review moves to ten."}]}`},
}

// sendOutputLimitToolTurn runs a client's request through the request phases
// and returns the body sent to the backend.
func sendOutputLimitToolTurn(t *testing.T, router *OpenAIRouter, client llmprotocol.WireFormat, stream bool) ([]byte, *RequestContext) {
	t.Helper()
	request := outputLimitToolRequests[client]
	ctx := &RequestContext{Headers: map[string]string{}}
	require.NoError(t, router.processRequestHeaders(NewMockStream(nil), newRequestHeaders("POST", request.path), ctx))
	replies := NewMockStream(nil)
	require.NoError(t, router.processRequestBody(replies, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(fmt.Sprintf(request.body, stream)), EndOfStream: true},
	}, ctx))
	require.Len(t, replies.Responses, 1)
	require.Nil(t, replies.Responses[0].GetImmediateResponse(), "the request was refused")
	return replies.Responses[0].GetRequestBody().GetResponse().GetBodyMutation().GetBody(), ctx
}

// answerOutputLimitToolTurn runs the backend's reply, read by read, through
// the response phases and returns what the client receives and the
// x-vsr-warnings sent with the last read.
func answerOutputLimitToolTurn(t *testing.T, router *OpenAIRouter, ctx *RequestContext, contentType string, reads ...string) (string, string) {
	t.Helper()
	stream := NewMockStream(nil)
	require.NoError(t, router.processResponseHeaders(stream, &ext_proc.ProcessingRequest{}, &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: "200"}, {Key: "content-type", Value: contentType},
		}}},
	}, ctx))
	var received strings.Builder
	var warnings string
	for index, read := range reads {
		require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(read), EndOfStream: index == len(reads)-1},
		}, ctx))
		response := stream.Responses[len(stream.Responses)-1]
		require.Nil(t, response.GetImmediateResponse(), "the reply was refused")
		if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			read = string(mutation.GetBody())
		}
		received.WriteString(read)
		warnings = protocolDiagnosticsHeader(response.GetResponseBody().GetResponse().GetHeaderMutation())
	}
	return received.String(), warnings
}

func chatToolCallReply(finish string, completionTokens int) string {
	return fmt.Sprintf(`{"id":"chatcmpl-limit","object":"chat.completion","created":1,"model":"worker","choices":[{"index":0,`+
		`"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",`+
		`"function":{"name":"send_email","arguments":"{}"}}]},"finish_reason":%q}],`+
		`"usage":{"prompt_tokens":20,"completion_tokens":%d,"total_tokens":%d}}`, finish, completionTokens, 20+completionTokens)
}

// chatToolCallStream is the stream of a call the output limit cut inside its
// arguments, ending with finish.
func chatToolCallStream(finish string) []string {
	chunk := func(fields string) string {
		return `data: {"id":"chatcmpl-limit","object":"chat.completion.chunk","created":1,"model":"worker",` + fields + "}\n\n"
	}
	return []string{
		chunk(`"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"send_email","arguments":""}}]},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"body\": \"Dear team, the review moves to ten."}}]},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{},"finish_reason":"` + finish + `"}]`),
		chunk(`"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":50,"total_tokens":70}`),
		"data: [DONE]\n\n",
	}
}

// vLLM's buffered Chat reply says tool_calls whenever its tool parser found a
// call, also one the output limit cut, which some parsers close as {}; its
// stream of the same answer ends with length. Once the reply's output reached
// the limit the Router sent the backend, its decision's cap and not the
// client's 256, every client gets the reply as one the limit cut, buffered and
// streamed alike, where it got a complete-looking call. Below the limit the
// tool-call stop stands.
func TestToolCallStopAtTheRoutersOutputLimitReachesEveryClientAsALimitStop(t *testing.T) {
	router := outputLimitToolRouter(t)
	type stops struct{ limit, toolCall []string }
	buffered := map[llmprotocol.WireFormat]stops{
		llmprotocol.OpenAIChatV1: {
			limit:    []string{`"finish_reason":"length"`, `"arguments":"{}"`},
			toolCall: []string{`"finish_reason":"tool_calls"`},
		},
		llmprotocol.OpenAIResponsesV1: {
			limit:    []string{`"status":"incomplete"`, `"incomplete_details":{"reason":"max_output_tokens"}`, `"type":"function_call"`},
			toolCall: []string{`"status":"completed"`},
		},
		llmprotocol.AnthropicMessagesV1: {
			limit:    []string{`"stop_reason":"max_tokens"`, `"input":{}`},
			toolCall: []string{`"stop_reason":"tool_use"`},
		},
	}
	streamed := map[llmprotocol.WireFormat]stops{
		llmprotocol.OpenAIChatV1:        {limit: []string{`"finish_reason":"length"`}, toolCall: []string{`"finish_reason":"tool_calls"`}},
		llmprotocol.OpenAIResponsesV1:   {limit: []string{`"type":"response.incomplete"`}, toolCall: []string{`"type":"response.completed"`}},
		llmprotocol.AnthropicMessagesV1: {limit: []string{`"stop_reason":"max_tokens"`}, toolCall: []string{`"stop_reason":"tool_use"`}},
	}
	for client := range buffered {
		t.Run(string(client), func(t *testing.T) {
			sent, ctx := sendOutputLimitToolTurn(t, router, client, false)
			require.Contains(t, string(sent), `"max_completion_tokens":50`)
			body, warnings := answerOutputLimitToolTurn(t, router, ctx, "application/json", chatToolCallReply("tool_calls", 50))
			for _, fragment := range buffered[client].limit {
				require.Contains(t, body, fragment)
			}
			require.NotContains(t, body, buffered[client].toolCall[0])
			require.Contains(t, warnings, `;field="response.stop_reason"`)

			_, ctx = sendOutputLimitToolTurn(t, router, client, false)
			body, warnings = answerOutputLimitToolTurn(t, router, ctx, "application/json", chatToolCallReply("tool_calls", 30))
			for _, fragment := range buffered[client].toolCall {
				require.Contains(t, body, fragment)
			}
			require.NotContains(t, body, buffered[client].limit[0])
			require.NotContains(t, warnings, "response.stop_reason")

			// The stream ends with length, as vLLM's does, or with tool_calls,
			// as some of its tool parsers end a call the limit cut.
			for _, finish := range []string{"length", "tool_calls"} {
				_, ctx = sendOutputLimitToolTurn(t, router, client, true)
				body, _ = answerOutputLimitToolTurn(t, router, ctx, "text/event-stream", chatToolCallStream(finish)...)
				for _, fragment := range streamed[client].limit {
					require.Contains(t, body, fragment, "stream ending with %s", finish)
				}
				require.NotContains(t, body, streamed[client].toolCall[0], "stream ending with %s", finish)
				require.NotContains(t, body, "invalid_stream_tool_arguments", "stream ending with %s", finish)
			}
		})
	}
}
