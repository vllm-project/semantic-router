package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

func chatBackendReply(body string) *parity.Upstream {
	return &parity.Upstream{Status: http.StatusOK, Headers: [][]string{{"content-type", "application/json"}}, Body: body}
}

// A history whose get_weather call is followed by a user turn and no result
// never reaches the backend: each client gets the 400 its own API gives,
// naming the call. With the result in place the same history is served.
func TestUnansweredToolCallIsRefusedInTheClientsTerms(t *testing.T) {
	const nextTurn = `{"role":"user","content":"And in Rome?"}]}`
	for _, test := range []struct {
		name, path, history, result string
		code, param, message        string
	}{
		{
			name: "chat completions", path: "/v1/chat/completions",
			history: `{"model":"vllm-sr/auto","tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}],"messages":[` +
				`{"role":"user","content":"What is the weather in Paris?"},` +
				`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},`,
			result: `{"role":"tool","tool_call_id":"call_1","content":"sunny"},`,
			code:   "unanswered_tool_call", param: "messages",
			message: "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. " +
				"The following tool_call_ids did not have response messages: call_1",
		},
		{
			name: "responses", path: "/v1/responses",
			history: `{"model":"vllm-sr/auto","store":false,"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"}}],"input":[` +
				`{"role":"user","content":"What is the weather in Paris?"},` +
				`{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"},`,
			result: `{"type":"function_call_output","call_id":"call_1","output":"sunny"},`,
			code:   "unanswered_tool_call", param: "input",
			message: "No tool output found for function call call_1.",
		},
		{
			name: "anthropic messages", path: "/v1/messages",
			history: `{"model":"vllm-sr/auto","max_tokens":64,"tools":[{"name":"get_weather","input_schema":{"type":"object"}}],"messages":[` +
				`{"role":"user","content":"What is the weather in Paris?"},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},`,
			result: `{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}]},{"role":"assistant","content":"It is sunny."},`,
			message: "messages.1: `tool_use` ids were found without `tool_result` blocks immediately after: toolu_1. " +
				"Each `tool_use` block must have a corresponding `tool_result` block in the next message.",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			record := runClientErrorCase(t, context.Background(), test.path, test.history+nextTurn, nil)
			require.Nil(t, record.Upstream, "the refused request reached a backend")
			require.Equal(t, http.StatusBadRequest, record.Response.Status, "%s", record.Response.Body)
			decoded := decodeClientErrorBody(t, record)
			require.Equal(t, "invalid_request_error", decoded.Error.Type)
			require.Equal(t, test.message, decoded.Error.Message)
			if test.code == "" {
				require.Equal(t, "error", decoded.Type, "an Anthropic envelope")
				require.Nil(t, decoded.Error.Code)
			} else {
				require.Equal(t, test.code, *decoded.Error.Code)
				require.Equal(t, test.param, *decoded.Error.Param)
			}

			record = runClientErrorCase(t, context.Background(), test.path, test.history+test.result+nextTurn,
				chatBackendReply(string(extProcResponseFixture(llmprotocol.OpenAIChatV1))))
			require.Equal(t, http.StatusOK, record.Response.Status, "%s", record.Response.Body)
			require.NotNil(t, record.Upstream)
			require.Contains(t, string(record.Upstream.Body), `"sunny"`, "the backend lost the tool result")
		})
	}
}

// A Responses request that continues a stored response is judged with that
// history: the call the stored response ended with is answered by a
// function_call_output in the request's input, and refused without one.
func TestResponsesContinuationIsJudgedWithItsStoredHistory(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(clientErrorsConfig))
	require.NoError(t, err)
	recorder := parity.NewRecorder(newParityRouter(t, cfg), routing.DefaultOptions)
	send := func(name, body string, upstream *parity.Upstream) *parity.Record {
		record := recorder.Run(context.Background(), parity.Case{
			Name:     name,
			Request:  parity.CaseRequest{Method: "POST", Path: "/v1/responses", Headers: [][]string{{"content-type", "application/json"}}, Body: body},
			Upstream: upstream,
		})
		require.Empty(t, record.Error)
		require.NotNil(t, record.Response)
		return record
	}
	tools := `"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"}}]`

	first := send("tool call", `{"model":"vllm-sr/auto",`+tools+`,"input":"What is the weather in Paris?"}`,
		chatBackendReply(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"chat-model","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},`+
			`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))
	require.Equal(t, http.StatusOK, first.Response.Status, "%s", first.Response.Body)
	var stored struct {
		ID     string `json:"id"`
		Output []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(first.Response.Body, &stored), "%s", first.Response.Body)
	require.Len(t, stored.Output, 1)
	require.Equal(t, "function_call", stored.Output[0].Type)
	callID := stored.Output[0].CallID

	refused := send("continuation without the output",
		`{"model":"vllm-sr/auto","store":false,"previous_response_id":"`+stored.ID+`",`+tools+`,"input":"And in Rome?"}`, nil)
	require.Nil(t, refused.Upstream, "the refused continuation reached a backend")
	require.Equal(t, http.StatusBadRequest, refused.Response.Status, "%s", refused.Response.Body)
	decoded := decodeClientErrorBody(t, refused)
	require.Equal(t, "unanswered_tool_call", *decoded.Error.Code)
	require.Equal(t, "input", *decoded.Error.Param)
	require.Equal(t, "No tool output found for function call "+callID+".", decoded.Error.Message)

	answered := send("continuation with the output",
		`{"model":"vllm-sr/auto","store":false,"previous_response_id":"`+stored.ID+`",`+tools+`,"input":[`+
			`{"type":"function_call_output","call_id":"`+callID+`","output":"sunny"}]}`,
		chatBackendReply(string(extProcResponseFixture(llmprotocol.OpenAIChatV1))))
	require.Equal(t, http.StatusOK, answered.Response.Status, "%s", answered.Response.Body)
	require.NotNil(t, answered.Upstream)
	var dispatched struct {
		Messages []struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID string `json:"id"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(answered.Upstream.Body, &dispatched), "%s", answered.Upstream.Body)
	require.Len(t, dispatched.Messages, 3, "%s", answered.Upstream.Body)
	require.Equal(t, callID, dispatched.Messages[1].ToolCalls[0].ID)
	require.Equal(t, "tool", dispatched.Messages[2].Role)
	require.Equal(t, callID, dispatched.Messages[2].ToolCallID)
}
