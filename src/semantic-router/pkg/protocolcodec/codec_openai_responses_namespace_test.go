package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// codexNamespaceRequest has the shape Codex CLI 0.166.0 sends with multi_agent
// on and an MCP server configured: namespaces beside plain function tools, and a
// history call that names its namespace.
const codexNamespaceRequest = `{
	"model":"vllm-sr/auto",
	"store":false,
	"input":[
		{"type":"message","role":"user","content":[{"type":"input_text","text":"Echo hi."}]},
		{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","namespace":"mcp__echo","name":"echo","arguments":"{\"text\":\"hi\"}"},
		{"type":"function_call_output","id":"fco_1","call_id":"call_1","output":"echo: hi"}
	],
	"tools":[
		{"type":"function","name":"exec_command","description":"Run a command.","strict":false,"parameters":{"type":"object"}},
		{"type":"namespace","name":"multi_agent_v1","description":"Tools for spawning and managing sub-agents.","tools":[
			{"type":"function","name":"spawn_agent","description":"Start a child.","strict":false,"parameters":{"type":"object"}}
		]},
		{"type":"namespace","name":"mcp__echo","description":"Tools in the mcp__echo namespace.","tools":[
			{"type":"function","name":"echo","description":"Echo the text.","strict":false,"parameters":{"type":"object"}},
			{"type":"custom","name":"raw_echo","description":"Echo free-form input."}
		]}
	]
}`

func decodeNamespaceRequest(t *testing.T, body string) llmprotocol.Request {
	t.Helper()
	request, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	require.NoError(t, err)
	return request
}

func namespaceRequestWithTools(tools string) string {
	return `{"model":"m","input":"hello","tools":` + tools + `}`
}

func TestResponsesNamespaceToolsDecodeToFlatTools(t *testing.T) {
	request := decodeNamespaceRequest(t, codexNamespaceRequest)
	names := make([]string, 0, len(request.Tools))
	for _, tool := range request.Tools {
		names = append(names, tool.Name)
	}
	require.Equal(t, []string{"exec_command", "multi_agent_v1__spawn_agent", "mcp__echo__echo", "mcp__echo__raw_echo"}, names)

	require.Empty(t, request.Tools[0].Namespace)
	spawn := request.Tools[1]
	require.Equal(t, "multi_agent_v1", spawn.Namespace)
	require.Equal(t, "spawn_agent", spawn.LocalName)
	require.Equal(t, "Tools for spawning and managing sub-agents.\n\nStart a child.", spawn.Description)
	require.JSONEq(t, `{"type":"object"}`, string(spawn.InputSchema))
	require.Equal(t, llmprotocol.ToolKindCustom, request.Tools[3].Kind)
	require.Equal(t, "raw_echo", request.Tools[3].LocalName)
}

func TestResponsesNamespaceHistoryCallsFlattenToTheDeclaredName(t *testing.T) {
	request := decodeNamespaceRequest(t, codexNamespaceRequest)
	var call *llmprotocol.ToolCall
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.ToolCall != nil {
				call = content.ToolCall
			}
		}
	}
	require.NotNil(t, call)
	require.Equal(t, "mcp__echo__echo", call.Name)
	require.Empty(t, call.Namespace, "a backend never sees the namespace")
}

func TestResponsesNamespaceCustomHistoryCallFlattens(t *testing.T) {
	request := decodeNamespaceRequest(t, `{"model":"m","input":[
		{"type":"custom_tool_call","call_id":"call_1","namespace":"ns","name":"raw","input":"x"},
		{"type":"custom_tool_call_output","call_id":"call_1","output":"y"}
	]}`)
	require.Equal(t, "ns__raw", request.Messages[0].Content[0].ToolCall.Name)
	require.Equal(t, llmprotocol.ToolKindCustom, request.Messages[0].Content[0].ToolCall.Kind)
}

// Flattening shortens a long namespace or name, so the limit applies to both
// as the client sent them, for function and custom calls alike.
func TestResponsesNamespaceHistoryCallsKeepTheToolNameLimit(t *testing.T) {
	long := strings.Repeat("a", 257)
	for name, items := range map[string]string{
		"function name":      `{"type":"function_call","call_id":"call_1","namespace":"ns","name":"` + long + `","arguments":"{}"},{"type":"function_call_output","call_id":"call_1","output":"done"}`,
		"function namespace": `{"type":"function_call","call_id":"call_1","namespace":"` + long + `","name":"tool","arguments":"{}"},{"type":"function_call_output","call_id":"call_1","output":"done"}`,
		"custom name":        `{"type":"custom_tool_call","call_id":"call_1","namespace":"ns","name":"` + long + `","input":"x"},{"type":"custom_tool_call_output","call_id":"call_1","output":"done"}`,
		"custom namespace":   `{"type":"custom_tool_call","call_id":"call_1","namespace":"` + long + `","name":"tool","input":"x"},{"type":"custom_tool_call_output","call_id":"call_1","output":"done"}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(`{"model":"m","input":[`+items+`]}`))
			requireProtocolErrorCode(t, err, "tool_call_limit")
		})
	}
	atLimit := strings.Repeat("a", 256)
	_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(`{"model":"m","input":[`+
		`{"type":"function_call","call_id":"call_1","namespace":"`+atLimit+`","name":"`+atLimit+`","arguments":"{}"},`+
		`{"type":"function_call_output","call_id":"call_1","output":"done"}]}`))
	require.NoError(t, err)
}

func TestResponsesNamespaceIsRejectedOutsideToolCalls(t *testing.T) {
	_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(
		`{"model":"m","input":[{"type":"message","role":"user","namespace":"ns","content":"hi"}]}`,
	))
	require.Error(t, err)
}

func TestResponsesNamespaceToolsFailClosed(t *testing.T) {
	function := `{"type":"function","name":"tool","parameters":{"type":"object"}}`
	tests := []struct {
		name  string
		tools string
		code  string
	}{
		{"nested namespace", `[{"type":"namespace","name":"a","tools":[{"type":"namespace","name":"b","tools":[` + function + `]}]}]`, "unsupported_tool"},
		{"hosted entry", `[{"type":"namespace","name":"a","tools":[{"type":"web_search"}]}]`, "unsupported_tool"},
		{"image generation entry", `[{"type":"namespace","name":"a","tools":[{"type":"image_generation"}]}]`, "unsupported_tool"},
		{"no entries", `[{"type":"namespace","name":"a","tools":[]}]`, "invalid_tool"},
		{"no namespace name", `[{"type":"namespace","name":" ","tools":[` + function + `]}]`, "invalid_tool"},
		{"no entry name", `[{"type":"namespace","name":"a","tools":[{"type":"function","parameters":{"type":"object"}}]}]`, "invalid_tool"},
		{"deferred entry", `[{"type":"namespace","name":"a","tools":[{"type":"function","name":"tool","defer_loading":true,"parameters":{"type":"object"}}]}]`, "unsupported_tools_defer_loading"},
		{"collision across namespaces", `[
			{"type":"namespace","name":"mcp__a__b","tools":[{"type":"function","name":"c","parameters":{"type":"object"}}]},
			{"type":"namespace","name":"mcp__a","tools":[{"type":"function","name":"b__c","parameters":{"type":"object"}}]}
		]`, "tool_namespace_collision"},
		{"collision with a plain tool", `[
			{"type":"function","name":"a__tool","parameters":{"type":"object"}},
			{"type":"namespace","name":"a","tools":[` + function + `]}
		]`, "tool_namespace_collision"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(namespaceRequestWithTools(test.tools)))
			requireProtocolErrorCode(t, err, test.code)
		})
	}
}

// mcp__a with b__c and mcp__a__b with c both flatten to mcp__a__b__c, so a
// history call must not share a flat name with a different tool either.
func TestResponsesNamespaceHistoryCallsFailClosedOnACollision(t *testing.T) {
	call := func(id, kind, namespace, name string) string {
		namespaceField := ""
		if namespace != "" {
			namespaceField = `"namespace":"` + namespace + `",`
		}
		if kind == "custom_tool_call" {
			return `{"type":"custom_tool_call","call_id":"` + id + `",` + namespaceField + `"name":"` + name + `","input":"x"},` +
				`{"type":"custom_tool_call_output","call_id":"` + id + `","output":"done"}`
		}
		return `{"type":"function_call","call_id":"` + id + `",` + namespaceField + `"name":"` + name + `","arguments":"{}"},` +
			`{"type":"function_call_output","call_id":"` + id + `","output":"done"}`
	}
	declared := `[{"type":"namespace","name":"mcp__a__b","tools":[{"type":"function","name":"c","parameters":{"type":"object"}}]}]`
	tests := []struct {
		name  string
		input string
		tools string
	}{
		{"two history calls", call("call_1", "function_call", "mcp__a", "b__c") + "," + call("call_2", "function_call", "mcp__a__b", "c"), "[]"},
		{"history call and declared tool", call("call_1", "function_call", "mcp__a", "b__c"), declared},
		{"custom history call and declared tool", call("call_1", "custom_tool_call", "mcp__a", "b__c"), declared},
		{"plain and namespaced history calls", call("call_1", "function_call", "", "mcp__a__b__c") + "," + call("call_2", "function_call", "mcp__a__b", "c"), "[]"},
		{"namespaced history call and plain tool", call("call_1", "function_call", "mcp__a", "b__c"), `[{"type":"function","name":"mcp__a__b__c","parameters":{"type":"object"}}]`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(
				`{"model":"m","input":[`+test.input+`],"tools":`+test.tools+`}`,
			))
			requireProtocolErrorCode(t, err, "tool_namespace_collision")
		})
	}

	sameTool := decodeNamespaceRequest(t, `{"model":"m","input":[`+
		call("call_1", "function_call", "mcp__a__b", "c")+","+call("call_2", "function_call", "mcp__a__b", "c")+
		`],"tools":`+declared+`}`)
	require.True(t, sameTool.ToolNameOrigins.Flattened(), "a namespaced request keeps its origins for retained history")
	require.Nil(t, decodeNamespaceRequest(t, `{"model":"m","input":[`+call("call_1", "function_call", "", "tool")+`]}`).ToolNameOrigins)
}

func TestResponsesPlainDuplicateToolsKeepTheirOwnError(t *testing.T) {
	function := `{"type":"function","name":"tool","parameters":{"type":"object"}}`
	_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(
		llmprotocol.OpenAIResponsesV1, []byte(namespaceRequestWithTools(`[`+function+`,`+function+`]`)),
	)
	requireProtocolErrorCode(t, err, "duplicate_tool")
}

// A namespaced body must not reach a backend through same-format byte replay.
func TestResponsesNamespaceRequestIsNotReplayedToAResponsesBackend(t *testing.T) {
	historyOnly := `{"model":"m","input":[
		{"type":"function_call","call_id":"call_1","namespace":"ns","name":"tool","arguments":"{}"},
		{"type":"function_call_output","call_id":"call_1","output":"done"}
	]}`
	for name, body := range map[string]string{"declared tools": codexNamespaceRequest, "history only": historyOnly} {
		t.Run(name, func(t *testing.T) {
			engine := NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
			require.NoError(t, err)
			encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
			require.NoError(t, err)
			require.NotContains(t, string(encoded.Body), `"namespace"`)
		})
	}
}

func TestResponsesRequestWithoutNamespaceStillReplaysItsBytes(t *testing.T) {
	// A tool property called namespace is not a Responses namespace.
	body := `{"model":"m",  "input":"hi","tools":[{"type":"function","name":"get_pods","parameters":{"type":"object","properties":{"namespace":{"type":"string"}}}}]}`
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	require.NoError(t, err)
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	require.NoError(t, err)
	require.Equal(t, body, string(encoded.Body))
}

func TestResponsesNamespaceToolsReachEveryBackendFlat(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(format), func(t *testing.T) {
			engine := NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(strings.Replace(
				codexNamespaceRequest, `,
			{"type":"custom","name":"raw_echo","description":"Echo free-form input."}`, "", 1,
			)))
			require.NoError(t, err)
			request.Store = nil
			request.Sampling.MaxOutputTokens = maxTokensPointer(1024)
			encoded, err := engine.EncodeRequest(format, request, envelope)
			require.NoError(t, err)
			require.Contains(t, string(encoded.Body), "multi_agent_v1__spawn_agent")
			require.Contains(t, string(encoded.Body), "mcp__echo__echo")
			require.NotContains(t, string(encoded.Body), `"namespace"`)
			require.NotContains(t, string(encoded.Body), `"name":"echo"`)
		})
	}
}

func maxTokensPointer(value int64) *int64 { return &value }

// A custom tool in a namespace follows the custom-tool rule of each backend.
func TestResponsesNamespaceCustomToolFollowsEachBackend(t *testing.T) {
	body := namespaceRequestWithTools(`[{"type":"namespace","name":"ns","tools":[{"type":"custom","name":"raw","description":"Free form."}]}]`)
	encode := func(format llmprotocol.WireFormat) (RequestResult, error) {
		engine := NewBuiltinEngine()
		request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
		require.NoError(t, err)
		request.Sampling.MaxOutputTokens = maxTokensPointer(1024)
		return engine.EncodeRequest(format, request, envelope)
	}
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		encoded, err := encode(format)
		require.NoError(t, err, string(format))
		require.Contains(t, string(encoded.Body), `"ns__raw"`)
	}
	_, err := encode(llmprotocol.AnthropicMessagesV1)
	requireProtocolErrorCode(t, err, "unsupported_capability")
}

func TestResponsesNamespaceCustomCallRendersWithItsNamespace(t *testing.T) {
	namespaces := llmprotocol.RequestToolNamespaces([]llmprotocol.Tool{{Kind: llmprotocol.ToolKindCustom, Name: "ns__raw", Namespace: "ns", LocalName: "raw"}})
	body := `{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[
		{"id":"call_1","type":"custom","custom":{"name":"ns__raw","input":"free form"}}
	]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	translated, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, []byte(body), func(response *llmprotocol.Response) error {
		namespaces.RestoreResponse(response)
		return nil
	})
	require.NoError(t, err)
	require.Contains(t, string(translated.Body), `"type":"custom_tool_call"`)
	require.Contains(t, string(translated.Body), `"namespace":"ns"`)
	require.Contains(t, string(translated.Body), `"name":"raw"`)
	require.NotContains(t, string(translated.Body), "ns__raw")
}

// A named choice cannot reach a flat tool through its inner name or a namespace.
func TestResponsesNamedToolChoiceInsideANamespaceFailsClosed(t *testing.T) {
	tools := `"tools":[{"type":"namespace","name":"ns","tools":[{"type":"function","name":"tool","parameters":{"type":"object"}}]}]`
	for _, choice := range []string{
		`{"type":"function","name":"tool"}`,
		`{"type":"function","name":"tool","namespace":"ns"}`,
	} {
		_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1,
			[]byte(`{"model":"m","input":"hello",`+tools+`,"tool_choice":`+choice+`}`))
		require.Error(t, err, choice)
	}
}

func namespaceRestoreMutation() (ResponseMutation, StreamEventMutation) {
	namespaces := llmprotocol.RequestToolNamespaces([]llmprotocol.Tool{
		{Name: "mcp__echo__echo", Namespace: "mcp__echo", LocalName: "echo"},
	})
	return func(response *llmprotocol.Response) error {
			namespaces.RestoreResponse(response)
			return nil
		}, func(event *llmprotocol.Event) error {
			event.ToolCall = namespaces.RestoreCall(event.ToolCall)
			return nil
		}
}

func TestResponsesNamespaceCallsRenderBuffered(t *testing.T) {
	backends := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: `{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"mcp__echo__echo","arguments":"{\"text\":\"hi\"}"}}
		]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		llmprotocol.OpenAIResponsesV1: `{"id":"r1","object":"response","created_at":1,"model":"m","status":"completed","output":[
			{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"mcp__echo__echo","arguments":"{\"text\":\"hi\"}"}
		],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`,
		llmprotocol.AnthropicMessagesV1: `{"id":"m1","type":"message","role":"assistant","model":"m","stop_reason":"tool_use","stop_sequence":null,"content":[
			{"type":"tool_use","id":"call_1","name":"mcp__echo__echo","input":{"text":"hi"}}
		],"usage":{"input_tokens":1,"output_tokens":1}}`,
	}
	for format, body := range backends {
		t.Run(string(format), func(t *testing.T) {
			mutation, _ := namespaceRestoreMutation()
			translated, err := NewBuiltinEngine().TranslateResponse(format, llmprotocol.OpenAIResponsesV1, []byte(body), mutation)
			require.NoError(t, err)
			requireNamespacedOutputCall(t, translated.Body)
		})
	}
}

func requireNamespacedOutputCall(t *testing.T, body []byte) {
	t.Helper()
	var response struct {
		Output []struct {
			Type      string `json:"type"`
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
			CallID    string `json:"call_id"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(body, &response), string(body))
	require.Len(t, response.Output, 1, string(body))
	require.Equal(t, "function_call", response.Output[0].Type)
	require.Equal(t, "echo", response.Output[0].Name)
	require.Equal(t, "mcp__echo", response.Output[0].Namespace)
	require.Equal(t, "call_1", response.Output[0].CallID)
}

func TestResponsesNamespaceCallsRenderStreamed(t *testing.T) {
	streams := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"mcp__echo__echo","arguments":""}}]},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"text\":\"hi\"}"}}]},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}

data: [DONE]

`,
		llmprotocol.OpenAIResponsesV1: `event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"r1","object":"response","created_at":1,"model":"m","status":"in_progress","output":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"function_call","id":"fc_1","status":"in_progress","call_id":"call_1","name":"mcp__echo__echo","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":2,"output_index":0,"item_id":"fc_1","delta":"{\"text\":\"hi\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","sequence_number":3,"output_index":0,"item_id":"fc_1","name":"mcp__echo__echo","arguments":"{\"text\":\"hi\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":4,"output_index":0,"item":{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"mcp__echo__echo","arguments":"{\"text\":\"hi\"}"}}

event: response.completed
data: {"type":"response.completed","sequence_number":5,"response":{"id":"r1","object":"response","created_at":1,"model":"m","status":"completed","output":[{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"mcp__echo__echo","arguments":"{\"text\":\"hi\"}"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}

`,
		llmprotocol.AnthropicMessagesV1: `event: message_start
data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_1","name":"mcp__echo__echo","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"text\":\"hi\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`,
	}
	for format, wire := range streams {
		t.Run(string(format), func(t *testing.T) {
			_, mutation := namespaceRestoreMutation()
			stream, err := NewBuiltinEngine().NewStreamWithMutation(format, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{PublicModel: "m"}, mutation)
			require.NoError(t, err)
			frames, _, _, err := stream.Push([]byte(wire))
			require.NoError(t, err)
			final, _, _, err := stream.Finalize(nil)
			require.NoError(t, err)
			requireNamespacedCallFrames(t, append(frames, final...))
		})
	}
}

// A Chat stream may name the tool only after the call's first fragment; every
// item event and the completed response still carry the namespace.
func TestResponsesNamespaceSurvivesALateStreamedName(t *testing.T) {
	wire := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"arguments":""}}]},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"mcp__echo__echo","arguments":"{\"text\":\"hi\"}"}}]},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}

data: [DONE]

`
	_, mutation := namespaceRestoreMutation()
	stream, err := NewBuiltinEngine().NewStreamWithMutation(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{PublicModel: "m"}, mutation)
	require.NoError(t, err)
	frames, _, _, err := stream.Push([]byte(wire))
	require.NoError(t, err)
	final, _, _, err := stream.Finalize(nil)
	require.NoError(t, err)
	requireNamespacedCallFrames(t, append(frames, final...))
}

func TestStreamToolIdentityRejectsAChangedNamespace(t *testing.T) {
	state := &streamState{policy: llmprotocol.DefaultPolicy()}
	current := llmprotocol.ToolCall{ID: "call_1", Name: "echo", Namespace: "mcp__echo"}
	_, err := state.mergeStreamToolIdentity(current, llmprotocol.ToolCall{ID: "call_1", Name: "echo", Namespace: "mcp__other"})
	requireProtocolErrorCode(t, err, "stream_tool_identity_mismatch")
	merged, err := state.mergeStreamToolIdentity(llmprotocol.ToolCall{ID: "call_1"}, llmprotocol.ToolCall{Name: "echo", Namespace: "mcp__echo"})
	require.NoError(t, err)
	require.Equal(t, "mcp__echo", merged.Namespace)
}

func requireNamespacedCallFrames(t *testing.T, frames [][]byte) {
	t.Helper()
	joined := string(joinFrames(frames))
	seen := map[string]bool{}
	for _, line := range strings.Split(joined, "\n") {
		data, found := strings.CutPrefix(line, "data: ")
		if !found {
			continue
		}
		var event struct {
			Type string `json:"type"`
			Item struct {
				Type      string `json:"type"`
				Name      string `json:"name"`
				Namespace string `json:"namespace"`
			} `json:"item"`
			Response struct {
				Output []struct {
					Name      string `json:"name"`
					Namespace string `json:"namespace"`
				} `json:"output"`
			} `json:"response"`
		}
		require.NoError(t, json.Unmarshal([]byte(data), &event), data)
		switch event.Type {
		case "response.output_item.added", "response.output_item.done":
			require.Equal(t, "function_call", event.Item.Type, data)
			require.Equal(t, "echo", event.Item.Name, data)
			require.Equal(t, "mcp__echo", event.Item.Namespace, data)
			seen[event.Type] = true
		case "response.completed":
			require.Len(t, event.Response.Output, 1, data)
			require.Equal(t, "echo", event.Response.Output[0].Name, data)
			require.Equal(t, "mcp__echo", event.Response.Output[0].Namespace, data)
			seen[event.Type] = true
		}
	}
	require.Len(t, seen, 3, joined)
	require.NotContains(t, joined, "mcp__echo__echo")
}

func joinFrames(frames [][]byte) []byte {
	var joined []byte
	for _, frame := range frames {
		joined = append(joined, frame...)
	}
	return joined
}

// A cached public response is decoded again as Responses before it is rendered.
func TestResponsesNamespaceSurvivesAPublicResponseRoundTrip(t *testing.T) {
	body := `{"id":"r1","object":"response","created_at":1,"model":"m","status":"completed","output":[
		{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","namespace":"mcp__echo","name":"echo","arguments":"{}"}
	],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	engine := NewBuiltinEngine()
	decoded, envelope, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, []byte(body))
	require.NoError(t, err)
	decoded.Generation++
	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, decoded, envelope)
	require.NoError(t, err)
	requireNamespacedOutputCall(t, encoded.Body)

	frames, _, err := engine.EncodeResponseStream(llmprotocol.OpenAIResponsesV1, decoded, llmprotocol.StreamContext{PublicModel: "m"})
	require.NoError(t, err)
	require.Contains(t, string(frames), `"namespace":"mcp__echo"`)
}
