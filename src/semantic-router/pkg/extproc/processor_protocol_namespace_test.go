package extproc

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

const namespacedFlatToolName = "mcp__echo__echo"

func namespacedToolContext(t *testing.T, backend llmprotocol.WireFormat) *RequestContext {
	t.Helper()
	return &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		TargetFormat: backend,
		RequestModel: "public-model",
		TraceContext: t.Context(),
		// Ingress assigns this to every Responses request.
		ResponseObjectState: &ResponseObjectState{GeneratedResponseID: "resp_router_owned"},
		SemanticRequest: &llmprotocol.Request{
			Generation: 2,
			Tools: []llmprotocol.Tool{
				{Name: "exec_command", InputSchema: json.RawMessage(`{"type":"object"}`)},
				{Name: namespacedFlatToolName, Namespace: "mcp__echo", LocalName: "echo", InputSchema: json.RawMessage(`{"type":"object"}`)},
			},
		},
	}
}

func namespacedBackendResponse(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"` + namespacedFlatToolName + `","arguments":"{\"text\":\"hi\"}"}}]}}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"id":"response_1","type":"message","role":"assistant","model":"source-model","stop_reason":"tool_use","stop_sequence":null,"content":[{"type":"tool_use","id":"call_1","name":"` + namespacedFlatToolName + `","input":{"text":"hi"}}],"usage":{"input_tokens":2,"output_tokens":1}}`)
	default:
		return []byte(`{"id":"response_1","model":"source-model","status":"completed","output":[{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"` + namespacedFlatToolName + `","arguments":"{\"text\":\"hi\"}"}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
	}
}

func namespacedBackendStream(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"` + namespacedFlatToolName + `","arguments":"{\"text\":"}}]},"finish_reason":null}]}

data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"hi\"}"}}]},"finish_reason":"tool_calls"}]}

data: {"id":"response_1","model":"source-model","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}

data: [DONE]

`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`event: message_start
data: {"type":"message_start","message":{"id":"response_1","type":"message","role":"assistant","model":"source-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_1","name":"` + namespacedFlatToolName + `","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"text\":\"hi\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`)
	default:
		item := `"type":"function_call","id":"fc_1","call_id":"call_1","name":"` + namespacedFlatToolName + `"`
		return []byte(`event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"response_1","object":"response","created_at":1,"model":"source-model","status":"in_progress","output":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{` + item + `,"status":"in_progress","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":2,"output_index":0,"item_id":"fc_1","delta":"{\"text\":\"hi\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","sequence_number":3,"output_index":0,"item_id":"fc_1","arguments":"{\"text\":\"hi\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":4,"output_index":0,"item":{` + item + `,"status":"completed","arguments":"{\"text\":\"hi\"}"}}

event: response.completed
data: {"type":"response.completed","sequence_number":5,"response":{"id":"response_1","object":"response","created_at":1,"model":"source-model","status":"completed","output":[{` + item + `,"status":"completed","arguments":"{\"text\":\"hi\"}"}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}}

`)
	}
}

func requireNamespacedSemanticCall(t *testing.T, response *llmprotocol.Response) {
	t.Helper()
	require.NotNil(t, response)
	require.Len(t, response.Output, 1)
	require.Len(t, response.Output[0].Content, 1)
	call := response.Output[0].Content[0].ToolCall
	require.NotNil(t, call)
	require.Equal(t, "echo", call.Name)
	require.Equal(t, "mcp__echo", call.Namespace)
	require.JSONEq(t, `{"text":"hi"}`, call.Arguments)
}

func TestNamespacedToolCallsReachResponsesClientBuffered(t *testing.T) {
	for _, backend := range extProcMatrixFormats {
		t.Run(string(backend), func(t *testing.T) {
			ctx := namespacedToolContext(t, backend)
			router := &OpenAIRouter{}
			response := router.handleNonStreamingResponseBody(namespacedBackendResponse(backend), ctx, 0)
			body := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
			require.NotEmpty(t, body, "the backend body names a flat tool and must not be forwarded")
			require.Contains(t, string(body), `"namespace":"mcp__echo"`)
			require.Contains(t, string(body), `"name":"echo"`)
			require.NotContains(t, string(body), namespacedFlatToolName)
			requireNamespacedSemanticCall(t, ctx.SemanticResponse)
		})
	}
}

func TestNamespacedToolCallsReachResponsesClientStreamed(t *testing.T) {
	for _, backend := range extProcMatrixFormats {
		t.Run(string(backend), func(t *testing.T) {
			ctx := namespacedToolContext(t, backend)
			router := &OpenAIRouter{}
			response := router.handleSemanticStreamingResponseBody(namespacedBackendStream(backend), true, ctx)
			body := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
			require.NotEmpty(t, body, "the backend stream names a flat tool and must not be forwarded")
			require.GreaterOrEqual(t, bytes.Count(body, []byte(`"namespace":"mcp__echo"`)), 3, string(body))
			require.NotContains(t, string(body), namespacedFlatToolName)
			requireNamespacedSemanticCall(t, ctx.SemanticResponse)
		})
	}
}

func TestPlainToolsAddNoResponseMutation(t *testing.T) {
	ctx := namespacedToolContext(t, llmprotocol.OpenAIResponsesV1)
	ctx.SemanticRequest.Tools = ctx.SemanticRequest.Tools[:1]
	ctx.ResponseObjectState = nil
	require.Nil(t, clientResponseMutation(ctx, llmprotocol.OpenAIResponsesV1))
	require.Nil(t, clientStreamMutation(ctx, llmprotocol.OpenAIResponsesV1))
}

// Request history and a restored response name one tool the same way.
func TestReplayRecordsARestoredCallByItsFlatName(t *testing.T) {
	step := replayToolCallStep("response", "assistant", "openai.responses.v1", llmprotocol.ToolCall{
		ID: "call_1", Name: "echo", Namespace: "mcp__echo", Arguments: "{}",
	})
	require.Equal(t, namespacedFlatToolName, step.ToolName)
	plain := replayToolCallStep("response", "assistant", "openai.responses.v1", llmprotocol.ToolCall{ID: "call_2", Name: "exec_command"})
	require.Equal(t, "exec_command", plain.ToolName)
}

func TestNamespacedDispatchNeverCarriesTheNamespace(t *testing.T) {
	body := []byte(`{"model":"public-model","input":[
		{"type":"function_call","call_id":"call_1","namespace":"mcp__echo","name":"echo","arguments":"{}"},
		{"type":"function_call_output","call_id":"call_1","output":"done"}
	],"tools":[{"type":"namespace","name":"mcp__echo","description":"Echo tools.","tools":[{"type":"function","name":"echo","parameters":{"type":"object"}}]}]}`)
	for _, backend := range extProcMatrixFormats {
		t.Run(string(backend), func(t *testing.T) {
			engine := protocolcodec.NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
			require.NoError(t, err)
			request.Sampling.MaxOutputTokens = llmprotocol.Int64(1024)
			ctx := &RequestContext{
				SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: backend,
				RequestModel: "public-model", SemanticRequest: &request, ProtocolEnvelope: envelope,
			}
			dispatched, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx)
			require.NoError(t, err)
			require.NotContains(t, string(dispatched), `"namespace"`)
			require.GreaterOrEqual(t, bytes.Count(dispatched, []byte(namespacedFlatToolName)), 2, string(dispatched))
		})
	}
}

func TestNamespacedFallbackRequestCloneKeepsItsOrigin(t *testing.T) {
	ctx := namespacedToolContext(t, llmprotocol.OpenAIChatV1)
	cloned, err := cloneSemanticRequestForReplay(ctx.SemanticRequest)
	require.NoError(t, err)
	namespaces := llmprotocol.RequestToolNamespaces(cloned.Tools)
	restored := namespaces.RestoreCall(&llmprotocol.ToolCall{Name: namespacedFlatToolName})
	require.Equal(t, "mcp__echo", restored.Namespace)
	require.Equal(t, "echo", restored.Name)
}

func TestNamespacedToolCallsReachResponsesClientThroughLooper(t *testing.T) {
	chatBody := namespacedBackendResponse(llmprotocol.OpenAIChatV1)
	t.Run("buffered", func(t *testing.T) {
		ctx := namespacedToolContext(t, llmprotocol.OpenAIChatV1)
		semantic, body, err := prepareBufferedLooperResponse(
			protocolcodec.NewBuiltinEngine(),
			&looper.Response{Body: chatBody, BufferedBody: chatBody, Model: "public-model", ContentType: "application/json"},
			ctx, llmprotocol.OpenAIResponsesV1,
		)
		require.NoError(t, err)
		require.Contains(t, string(body), `"namespace":"mcp__echo"`)
		require.NotContains(t, string(body), namespacedFlatToolName)
		requireNamespacedSemanticCall(t, semantic)
	})
}

// A stored call keeps its namespace, so a previous_response_id continuation sends
// the backend the same flat name the original turn declared.
func TestNamespacedRetainedCallFlattensOnContinuation(t *testing.T) {
	var parsed responseapi.ResponseAPIResponse
	require.NoError(t, json.Unmarshal([]byte(`{"output":[{"type":"function_call","id":"fc_1","call_id":"call_1","namespace":"mcp__echo","name":"echo","arguments":"{}"}]}`), &parsed))
	var parsedInput responseapi.InputItem
	require.NoError(t, json.Unmarshal([]byte(`{"type":"function_call","call_id":"call_0","namespace":"mcp__echo","name":"echo","arguments":"{}"}`), &parsedInput))

	history, err := materializeStoredResponseHistory(protocolcodec.NewBuiltinEngine(), []*responseapi.StoredResponse{{
		Input: []responseapi.InputItem{
			parsedInput,
			{Type: responseapi.ItemTypeFunctionCallOutput, CallID: "call_0", Output: json.RawMessage(`"done"`)},
		},
		Output: parsed.Output,
	}})
	require.NoError(t, err)
	var names []string
	for _, message := range history.Messages {
		for _, content := range message.Content {
			if content.ToolCall != nil {
				names = append(names, content.ToolCall.Name)
			}
		}
	}
	require.Equal(t, []string{namespacedFlatToolName, namespacedFlatToolName}, names)
}

// A stored call to mcp__a/b__c flattens to the name the request declares for
// mcp__a__b/c, so the continuation fails closed instead of merging the two.
func TestNamespacedRetainedCallCollidingWithTheRequestFailsClosed(t *testing.T) {
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(`{
		"model":"m","previous_response_id":"resp_1","input":"next",
		"tools":[{"type":"namespace","name":"mcp__a__b","tools":[{"type":"function","name":"c","parameters":{"type":"object"}}]}]
	}`))
	require.NoError(t, err)
	var parsed responseapi.ResponseAPIResponse
	require.NoError(t, json.Unmarshal([]byte(`{"output":[{"type":"function_call","id":"fc_1","call_id":"call_1","namespace":"mcp__a","name":"b__c","arguments":"{}"}]}`), &parsed))
	ctx := &RequestContext{ResponseObjectState: &ResponseObjectState{
		ConversationHistory: []*responseapi.StoredResponse{{Output: parsed.Output}},
	}}

	_, err = (&OpenAIRouter{}).materializeResponseObjectContext(&request, ctx)
	assertResponseObjectStateError(t, err, llmprotocol.ErrorInvalidRequest, "tool_namespace_collision")
}
