package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
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

// Issue 7200: a successfully finished call with no argument bytes still has
// object-valued arguments, including at every streaming completion boundary.
// Drive the ext_proc adapter and then execute the mock tool, not just the codec.
func TestResponsesEmptyArgumentToolStreamRoundTrip(t *testing.T) {
	const optionalSchema = `{"type":"object","properties":{"timezone":{"type":"string"}},"additionalProperties":false}`
	const requiredSchema = `{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}`
	for _, test := range []struct {
		name, schema string
		members      []string
	}{
		{"omitted", optionalSchema, []string{""}},
		{"empty", optionalSchema, []string{`,"arguments":""`}},
		{"whitespace fragments", optionalSchema, []string{`,"arguments":" "`, `,"arguments":"\t"`, `,"arguments":"\r\n"`}},
		// The transport normalizes an empty object; schema validation belongs
		// to the tool executor, not the protocol bridge.
		{"required field omitted", requiredSchema, []string{""}},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes([]byte(clientErrorsConfig))
			require.NoError(t, err)
			processor := &extprocStreamProcessor{router: newParityRouter(t, cfg)}
			recorder := parity.NewRecorder(processor, routing.DefaultOptions)
			tools := `"tools":[{"type":"function","name":"get_time","strict":false,"parameters":` + test.schema + `}]`
			send := func(name, input string, chunks []string) *parity.Record {
				record := recorder.Run(t.Context(), parity.Case{
					Name: name,
					Request: parity.CaseRequest{
						Method: "POST", Path: "/v1/responses", Headers: [][]string{{"content-type", "application/json"}},
						Body: `{"model":"vllm-sr/auto","store":false,"stream":true,` + tools + `,"input":` + input + `}`,
					},
					Upstream: &parity.Upstream{Status: http.StatusOK, Headers: [][]string{{"content-type", "text/event-stream"}}, Chunks: chunks},
				})
				require.Empty(t, record.Error)
				require.NotNil(t, record.Response)
				require.Equal(t, http.StatusOK, record.Response.Status)
				require.NotNil(t, record.Upstream)
				require.Equal(t, "/v1/chat/completions", record.Upstream.Header.Get(":path"))
				require.NotEmpty(t, processor.takeSent(), "the ext_proc adapter was not exercised")
				return record
			}

			first := send("empty argument call", `"What time is it?"`, emptyArgumentChatToolStream(test.members, "tool_calls", 3))
			var request struct {
				Stream bool `json:"stream"`
				Tools  []struct {
					Function struct {
						Name       string          `json:"name"`
						Parameters json.RawMessage `json:"parameters"`
					} `json:"function"`
				} `json:"tools"`
			}
			require.NoError(t, json.Unmarshal(first.Upstream.Body, &request))
			require.True(t, request.Stream)
			require.Len(t, request.Tools, 1)
			require.Equal(t, "get_time", request.Tools[0].Function.Name)
			require.JSONEq(t, test.schema, string(request.Tools[0].Function.Parameters))
			terminal := assertEmptyArgumentResponsesCompletion(t, responsesToolEvents(t, parityStreamBody(first)), "completed")
			call := terminal.Response.Output[0]

			// Execute a mock optional-only function with the arguments the client
			// actually received, then replay its output alongside the call.
			var arguments map[string]any
			require.NoError(t, json.Unmarshal([]byte(call.Arguments), &arguments))
			require.NotNil(t, arguments)
			require.Empty(t, arguments)
			mockGetTime := func(arguments map[string]any) string {
				require.Empty(t, arguments)
				return "10:00 UTC"
			}
			mockOutput := mockGetTime(arguments)
			input, err := json.Marshal([]any{
				map[string]any{"role": "user", "content": "What time is it?"},
				map[string]any{"type": "function_call", "id": call.ID, "call_id": call.CallID, "name": call.Name, "arguments": call.Arguments},
				map[string]any{"type": "function_call_output", "call_id": call.CallID, "output": mockOutput},
			})
			require.NoError(t, err)
			continued := send("mock tool output", string(input), []string{
				`data: {"id":"chatcmpl-continued","model":"chat-model","choices":[{"index":0,"delta":{"role":"assistant","content":"It is 10:00 UTC."},"finish_reason":"stop"}]}` + "\n\n",
				"data: [DONE]\n\n",
			})
			var dispatched struct {
				Messages []struct {
					Role       string `json:"role"`
					Content    string `json:"content"`
					ToolCallID string `json:"tool_call_id"`
					ToolCalls  []struct {
						ID       string `json:"id"`
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(continued.Upstream.Body, &dispatched))
			require.Len(t, dispatched.Messages, 3)
			require.Equal(t, "assistant", dispatched.Messages[1].Role)
			require.Len(t, dispatched.Messages[1].ToolCalls, 1)
			require.Equal(t, call.CallID, dispatched.Messages[1].ToolCalls[0].ID)
			require.Equal(t, "{}", dispatched.Messages[1].ToolCalls[0].Function.Arguments)
			require.Equal(t, "tool", dispatched.Messages[2].Role)
			require.Equal(t, call.CallID, dispatched.Messages[2].ToolCallID)
			require.Equal(t, mockOutput, dispatched.Messages[2].Content)
			events := responsesToolEvents(t, parityStreamBody(continued))
			var answer strings.Builder
			for _, event := range events {
				if event.Type == "response.output_text.delta" {
					answer.WriteString(event.Delta)
				}
			}
			require.Equal(t, "It is 10:00 UTC.", answer.String())
			require.Equal(t, "response.completed", events[len(events)-1].Type)
			require.Equal(t, "completed", events[len(events)-1].Response.Status)
		})
	}
}

func emptyArgumentChatToolStream(members []string, finish string, completionTokens int) []string {
	chunk := func(choice string) string {
		return `data: {"id":"chatcmpl-empty","object":"chat.completion.chunk","created":1,"model":"chat-model","choices":[` + choice + "]}\n\n"
	}
	reads := []string{chunk(`{"index":0,"delta":{"role":"assistant"},"finish_reason":null}`)}
	for index, member := range members {
		identity := ""
		name := ""
		if index == 0 {
			identity = `,"id":"call_empty","type":"function"`
			name = `"name":"get_time"`
		} else {
			member = strings.TrimPrefix(member, ",")
		}
		reads = append(reads, chunk(`{"index":0,"delta":{"tool_calls":[{"index":0`+identity+`,"function":{`+name+member+`}}]},"finish_reason":null}`))
	}
	return append(reads,
		chunk(`{"index":0,"delta":{},"finish_reason":"`+finish+`"}`),
		fmt.Sprintf("data: {\"id\":\"chatcmpl-empty\",\"model\":\"chat-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n", completionTokens, 5+completionTokens),
		"data: [DONE]\n\n")
}

type responsesToolItem struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Status    string `json:"status"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type responsesToolEvent struct {
	Type      string            `json:"type"`
	Code      string            `json:"code"`
	ItemID    string            `json:"item_id"`
	Delta     string            `json:"delta"`
	Arguments string            `json:"arguments"`
	Item      responsesToolItem `json:"item"`
	Response  struct {
		Status            string              `json:"status"`
		Output            []responsesToolItem `json:"output"`
		IncompleteDetails struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details"`
	} `json:"response"`
}

func parityStreamBody(record *parity.Record) string {
	var body strings.Builder
	body.Write(record.Response.Body)
	for _, chunk := range record.Response.Chunks {
		body.Write(chunk)
	}
	return body.String()
}

func responsesToolEvents(t *testing.T, body string) []responsesToolEvent {
	t.Helper()
	var events []responsesToolEvent
	for _, line := range strings.Split(body, "\n") {
		if data, ok := strings.CutPrefix(line, "data: "); ok {
			var event responsesToolEvent
			require.NoError(t, json.Unmarshal([]byte(data), &event), "%s", body)
			events = append(events, event)
		}
	}
	require.NotEmpty(t, events, "%s", body)
	return events
}

func assertEmptyArgumentResponsesCompletion(t *testing.T, events []responsesToolEvent, status string) responsesToolEvent {
	t.Helper()
	var joined strings.Builder
	var argsDone, itemDone []responsesToolEvent
	for _, event := range events {
		switch event.Type {
		case "response.function_call_arguments.delta":
			joined.WriteString(event.Delta)
		case "response.function_call_arguments.done":
			argsDone = append(argsDone, event)
		case "response.output_item.done":
			if event.Item.Type == "function_call" {
				itemDone = append(itemDone, event)
			}
		case "response.failed", "error":
			t.Fatalf("tool stream failed: %+v", event)
		}
	}
	require.Equal(t, "{}", joined.String())
	require.Len(t, argsDone, 1)
	require.Equal(t, "{}", argsDone[0].Arguments)
	require.Len(t, itemDone, 1)
	require.Equal(t, "{}", itemDone[0].Item.Arguments)
	terminal := events[len(events)-1]
	require.Equal(t, "response."+status, terminal.Type)
	require.Equal(t, status, terminal.Response.Status)
	require.Len(t, terminal.Response.Output, 1)
	call := terminal.Response.Output[0]
	require.Equal(t, "function_call", call.Type)
	require.Equal(t, status, call.Status)
	require.NotEmpty(t, call.ID)
	require.Equal(t, "get_time", call.Name)
	require.Equal(t, "call_empty", call.CallID)
	require.Equal(t, "{}", call.Arguments)
	require.Equal(t, itemDone[0].Item, call)
	require.Equal(t, argsDone[0].ItemID, call.ID)
	return terminal
}

func TestResponsesEmptyArgumentToolStreamRejectsInvalidArguments(t *testing.T) {
	for _, member := range []string{`,"arguments":"null"`, `,"arguments":null`, `,"arguments":"{broken"`} {
		t.Run(member, func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIChatV1, RequestModel: "public-model", TraceContext: t.Context()}
			var body strings.Builder
			reads := emptyArgumentChatToolStream([]string{member}, "tool_calls", 3)
			for index, read := range reads {
				reply := router.handleSemanticStreamingResponseBody([]byte(read), index == len(reads)-1, ctx)
				body.Write(deliveredBody(t, reply, read))
			}
			require.True(t, ctx.StreamingAborted, "%s", body.String())
			require.Nil(t, ctx.SemanticResponse)
			require.NotContains(t, body.String(), `"type":"response.completed"`)
			events := responsesToolEvents(t, body.String())
			require.Equal(t, "error", events[len(events)-1].Type)
			require.NotEmpty(t, events[len(events)-1].Code)
		})
	}
}

func TestResponsesNullArgumentPlaceholderStream(t *testing.T) {
	for _, test := range []struct {
		name    string
		members []string
		want    string
	}{
		{"object follows", []string{`,"arguments":null`, `,"arguments":"{\"timezone\":"`, `,"arguments":"\"UTC\"}"`}, `{"timezone":"UTC"}`},
		{"empty object follows", []string{`,"arguments":null`, `,"arguments":"{}"`}, "{}"},
		{"nothing follows", []string{`,"arguments":null`}, ""},
		{"whitespace follows", []string{`,"arguments":null`, `,"arguments":" \t"`}, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIChatV1, RequestModel: "public-model", TraceContext: t.Context()}
			var body strings.Builder
			reads := emptyArgumentChatToolStream(test.members, "tool_calls", 3)
			for index, read := range reads {
				reply := router.handleSemanticStreamingResponseBody([]byte(read), index == len(reads)-1, ctx)
				body.Write(deliveredBody(t, reply, read))
				if index <= len(test.members) {
					require.False(t, ctx.StreamingAborted, "placeholder rejected before completion: %s", body.String())
				}
			}
			events := responsesToolEvents(t, body.String())
			terminal := events[len(events)-1]
			if test.want == "" {
				require.True(t, ctx.StreamingAborted)
				require.Equal(t, "error", terminal.Type)
				require.Equal(t, "invalid_stream_tool_arguments", terminal.Code)
				require.NotContains(t, body.String(), `"type":"response.completed"`)
				return
			}
			require.False(t, ctx.StreamingAborted)
			require.Equal(t, "response.completed", terminal.Type)
			require.Len(t, terminal.Response.Output, 1)
			require.Equal(t, test.want, terminal.Response.Output[0].Arguments)
			var deltas strings.Builder
			for _, event := range events {
				switch event.Type {
				case "response.function_call_arguments.delta":
					deltas.WriteString(event.Delta)
				case "response.function_call_arguments.done":
					require.Equal(t, test.want, event.Arguments)
				case "response.output_item.done":
					require.Equal(t, test.want, event.Item.Arguments)
				}
			}
			require.Equal(t, test.want, deltas.String())
		})
	}
}

func TestResponsesEmptyArgumentToolStreamPreservesOutputLimits(t *testing.T) {
	router := outputLimitToolRouter(t)
	for _, test := range []struct {
		name, finish string
		tokens       int
	}{
		{"length below cap", "length", 3},
		{"tool calls at cap", "tool_calls", 50},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := sendOutputLimitToolTurn(t, router, llmprotocol.OpenAIResponsesV1, true)
			body, _ := answerOutputLimitToolTurn(t, router, ctx, "text/event-stream", emptyArgumentChatToolStream([]string{""}, test.finish, test.tokens)...)
			events := responsesToolEvents(t, body)
			terminal := events[len(events)-1]
			require.Equal(t, "response.incomplete", terminal.Type)
			require.Equal(t, "incomplete", terminal.Response.Status)
			require.Equal(t, "max_output_tokens", terminal.Response.IncompleteDetails.Reason)
			require.Len(t, terminal.Response.Output, 1)
			require.Equal(t, "function_call", terminal.Response.Output[0].Type)
			require.NotContains(t, body, `"type":"response.completed"`)
			require.NotContains(t, body, `"type":"response.failed"`)
		})
	}
}
