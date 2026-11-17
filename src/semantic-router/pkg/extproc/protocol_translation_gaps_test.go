package extproc

import (
	"bytes"
	"encoding/json"
	"slices"
	"strconv"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Responses usage always carries an integer cached_tokens, 0 when the backend
// reported no cache-read count, as the Responses schema requires and vLLM's own
// Responses server sends. Codex CLI fails the turn on null or on a details
// object without it, and openai-python reads a missing object as None. A
// cache-write count appears only when the backend reported one.
func TestResponsesClientUsageCarriesIntegerCachedTokens(t *testing.T) {
	buffered := string(extProcResponseFixture(llmprotocol.OpenAIChatV1))
	stream := string(extProcStreamFixture(llmprotocol.OpenAIChatV1))
	writeOnly := func(body string) string {
		return strings.Replace(body, `"total_tokens":3}`, `"total_tokens":3,"prompt_tokens_details":{"cache_write_tokens":1}}`, 1)
	}
	cases := map[string]struct {
		body      string
		streaming bool
		write     string
	}{
		"no-counts/buffered":  {buffered, false, ""},
		"no-counts/stream":    {stream, true, ""},
		"write-only/buffered": {writeOnly(buffered), false, "1"},
		"write-only/stream":   {writeOnly(stream), true, "1"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{
				SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIChatV1,
				RequestModel: "public-model", TraceContext: t.Context(),
			}
			var resource []byte
			if test.streaming {
				resource = responsesTerminalResource(t, clientResponseBody(t,
					(&OpenAIRouter{}).handleSemanticStreamingResponseBody([]byte(test.body), true, ctx)))
			} else {
				resource = clientResponseBody(t, (&OpenAIRouter{}).handleNonStreamingResponseBody([]byte(test.body), ctx, 0, false))
			}
			var wire struct {
				Usage struct {
					Details map[string]json.RawMessage `json:"input_tokens_details"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(resource, &wire); err != nil {
				t.Fatalf("client body is not JSON: %v\n%s", err, resource)
			}
			if got := string(wire.Usage.Details["cached_tokens"]); got != "0" {
				t.Fatalf("cached_tokens = %q, want the integer 0: %s", got, resource)
			}
			if got := string(wire.Usage.Details["cache_write_tokens"]); got != test.write {
				t.Fatalf("cache_write_tokens = %q, want %q: %s", got, test.write, resource)
			}
		})
	}
}

func clientResponseBody(t *testing.T, reply *ext_proc.ProcessingResponse) []byte {
	t.Helper()
	if immediate := reply.GetImmediateResponse(); immediate != nil {
		t.Fatalf("response was replaced: status=%v body=%s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	mutation := reply.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatalf("response body was not translated: %+v", reply)
	}
	if streamed := mutation.GetStreamedResponse(); streamed != nil {
		return streamed.GetBody()
	}
	return mutation.GetBody()
}

func responsesTerminalResource(t *testing.T, stream []byte) []byte {
	t.Helper()
	for _, line := range strings.Split(string(stream), "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var event struct {
			Type     string          `json:"type"`
			Response json.RawMessage `json:"response"`
		}
		if err := json.Unmarshal([]byte(payload), &event); err == nil && event.Type == "response.completed" {
			return event.Response
		}
	}
	t.Fatalf("stream has no response.completed event:\n%s", stream)
	return nil
}

// Every tool call reaches a Chat client whole, named once, and in the order of
// the backend's output: a call that takes no arguments and streams none or an
// empty delta, a call whose input arrives only at its start or only when it is
// done, a call streamed in several deltas, and parallel calls that finish out
// of order.
func TestChatStreamToolCallsJoinAsTheBackendSentThem(t *testing.T) {
	anthropic := func(blocks ...string) string {
		event := func(name, data string) string { return "event: " + name + "\ndata: " + data + "\n\n" }
		stream := event("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`)
		for index, block := range blocks {
			parts := strings.Split(block, "|")
			stream += event("content_block_start", `{"type":"content_block_start","index":`+strconv.Itoa(index)+`,"content_block":{"type":"tool_use","id":"toolu_`+strconv.Itoa(index+1)+`","name":"`+parts[0]+`","input":`+parts[1]+`}}`)
			for _, delta := range parts[2:] {
				stream += event("content_block_delta", `{"type":"content_block_delta","index":`+strconv.Itoa(index)+`,"delta":{"type":"input_json_delta","partial_json":`+strconv.Quote(delta)+`}}`)
			}
			stream += event("content_block_stop", `{"type":"content_block_stop","index":`+strconv.Itoa(index)+`}`)
		}
		return stream + event("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}`) +
			event("message_stop", `{"type":"message_stop"}`)
	}
	ids := []string{"a", "b"}
	call := func(item int, name, arguments, status string) string {
		id := ids[item]
		return `{"type":"function_call","id":"fc_` + id + `","call_id":"call_` + id + `","name":"` + name + `","arguments":` + strconv.Quote(arguments) + `,"status":"` + status + `"}`
	}
	added := func(item int, name string) [2]string {
		return [2]string{"response.output_item.added", `{"output_index":` + strconv.Itoa(item) + `,"item":` + call(item, name, "", "in_progress") + `}`}
	}
	delta := func(item int, text string) [2]string {
		return [2]string{"response.function_call_arguments.delta", `{"output_index":` + strconv.Itoa(item) + `,"item_id":"fc_` + ids[item] + `","delta":` + strconv.Quote(text) + `}`}
	}
	done := func(item int, name, arguments string) [][2]string {
		return [][2]string{
			{"response.function_call_arguments.done", `{"output_index":` + strconv.Itoa(item) + `,"item_id":"fc_` + ids[item] + `","arguments":` + strconv.Quote(arguments) + `}`},
			{"response.output_item.done", `{"output_index":` + strconv.Itoa(item) + `,"item":` + call(item, name, arguments, "completed") + `}`},
		}
	}
	responses := func(output []string, events ...[2]string) string {
		events = append([][2]string{{"response.created", `{"response":{"id":"resp_1","object":"response","created_at":1,"model":"m","status":"in_progress","output":[]}}`}}, events...)
		events = append(events, [2]string{"response.completed", `{"response":{"id":"resp_1","object":"response","created_at":1,"model":"m","status":"completed","output":[` +
			strings.Join(output, ",") + `],"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":0},"output_tokens":5,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":15}}}`})
		var stream strings.Builder
		for sequence, event := range events {
			data := strings.Replace(event[1], "{", `{"type":"`+event[0]+`","sequence_number":`+strconv.Itoa(sequence)+`,`, 1)
			stream.WriteString("event: " + event[0] + "\ndata: " + data + "\n\n")
		}
		return stream.String()
	}
	weather := `{"city":"Paris"}`
	single := func(name, arguments string, deltas ...[2]string) string {
		events := append([][2]string{added(0, name)}, deltas...)
		return responses([]string{call(0, name, arguments, "completed")}, append(events, done(0, name, arguments)...)...)
	}
	events := [][2]string{added(0, "get_time"), added(1, "get_date")}
	events = append(append(events, done(1, "get_date", "{}")...), done(0, "get_time", "{}")...)
	parallel := responses([]string{call(0, "get_time", "{}", "completed"), call(1, "get_date", "{}", "completed")}, events...)
	toolTime := joinedChatToolCall{"toolu_1", "function", "get_time", "{}"}
	toolWeather := joinedChatToolCall{"toolu_1", "function", "get_weather", weather}
	callTime := joinedChatToolCall{"call_a", "function", "get_time", "{}"}
	callWeather := joinedChatToolCall{"call_a", "function", "get_weather", weather}
	streams := map[string]struct {
		backend llmprotocol.WireFormat
		body    string
		want    []joinedChatToolCall
	}{
		"anthropic/no-delta":              {llmprotocol.AnthropicMessagesV1, anthropic("get_time|{}"), []joinedChatToolCall{toolTime}},
		"anthropic/empty-delta":           {llmprotocol.AnthropicMessagesV1, anthropic("get_time|{}|"), []joinedChatToolCall{toolTime}},
		"anthropic/start-input":           {llmprotocol.AnthropicMessagesV1, anthropic("get_weather|" + weather), []joinedChatToolCall{toolWeather}},
		"anthropic/several-deltas":        {llmprotocol.AnthropicMessagesV1, anthropic(`get_weather|{}|{"city":|"Paris"}`, "get_time|{}"), []joinedChatToolCall{toolWeather, {"toolu_2", "function", "get_time", "{}"}}},
		"responses/no-delta":              {llmprotocol.OpenAIResponsesV1, single("get_time", "{}"), []joinedChatToolCall{callTime}},
		"responses/empty-delta":           {llmprotocol.OpenAIResponsesV1, single("get_time", "{}", delta(0, "")), []joinedChatToolCall{callTime}},
		"responses/arguments-on-done":     {llmprotocol.OpenAIResponsesV1, single("get_weather", weather), []joinedChatToolCall{callWeather}},
		"responses/several-deltas":        {llmprotocol.OpenAIResponsesV1, single("get_weather", weather, delta(0, `{"city":`), delta(0, `"Paris"}`)), []joinedChatToolCall{callWeather}},
		"responses/parallel-out-of-order": {llmprotocol.OpenAIResponsesV1, parallel, []joinedChatToolCall{callTime, {"call_b", "function", "get_date", "{}"}}},
	}
	for name, stream := range streams {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: stream.backend, RequestModel: "public-model", TraceContext: t.Context()}
			body := clientResponseBody(t, (&OpenAIRouter{}).handleSemanticStreamingResponseBody([]byte(stream.body), true, ctx))
			if got := joinChatStreamToolCalls(t, body); !slices.Equal(got, stream.want) {
				t.Fatalf("joined tool calls = %+v, want %+v:\n%s", got, stream.want, body)
			}
		})
	}
}

type joinedChatToolCall struct{ ID, Type, Name, Arguments string }

// joinChatStreamToolCalls reads a Chat stream's tool calls the way
// openai-python's ChatCompletionStreamState does: deltas are grouped by index,
// index and type are replaced, and every other string field is appended.
func joinChatStreamToolCalls(t *testing.T, body []byte) []joinedChatToolCall {
	t.Helper()
	calls := []joinedChatToolCall{}
	for _, line := range strings.Split(string(body), "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("invalid chunk %s: %v", payload, err)
		}
		for _, choice := range chunk.Choices {
			for _, delta := range choice.Delta.ToolCalls {
				for len(calls) <= delta.Index {
					calls = append(calls, joinedChatToolCall{})
				}
				call := &calls[delta.Index]
				call.ID += delta.ID
				call.Name += delta.Function.Name
				call.Arguments += delta.Function.Arguments
				if delta.Type != "" {
					call.Type = delta.Type
				}
			}
		}
	}
	return calls
}

// A Chat backend ends a forced named tool call with finish_reason "stop", as
// OpenAI documents. Messages ends every turn that carries tool_use blocks with
// stop_reason "tool_use", and Messages clients run the calls on that signal.
func TestMessagesClientForcedToolCallStopsForToolUse(t *testing.T) {
	buffered := `{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null,` +
		`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"extract","arguments":"{\"x\":1}"}}]},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`
	stream := "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"extract\",\"arguments\":\"{\\\"x\\\":1}\"}}]},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\n" +
		"data: [DONE]\n\n"
	for _, streaming := range []bool{false, true} {
		ctx := &RequestContext{SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1, RequestModel: "public-model", TraceContext: t.Context()}
		var body []byte
		if streaming {
			body = clientResponseBody(t, (&OpenAIRouter{}).handleSemanticStreamingResponseBody([]byte(stream), true, ctx))
		} else {
			body = clientResponseBody(t, (&OpenAIRouter{}).handleNonStreamingResponseBody([]byte(buffered), ctx, 0, false))
		}
		if !bytes.Contains(body, []byte(`"stop_reason":"tool_use"`)) {
			t.Errorf("streaming=%t: a turn that ends in a tool call must stop with tool_use:\n%s", streaming, body)
		}
	}
}

// A Messages backend that stops because the context window filled up reports
// model_context_window_exceeded. Chat and Responses clients get the partial
// answer as a length stop, the way vLLM reports a stop at max_model_len, and
// the projection is reported without refusing the reply.
func TestContextWindowStopReachesOpenAIClientsAsLengthStop(t *testing.T) {
	buffered := `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"a long partial answer"}],` +
		`"stop_reason":"model_context_window_exceeded","stop_sequence":null,"usage":{"input_tokens":199000,"output_tokens":1000}}`
	stream := strings.Replace(string(extProcStreamFixture(llmprotocol.AnthropicMessagesV1)), `"stop_reason":"end_turn"`, `"stop_reason":"model_context_window_exceeded"`, 1)
	want := map[llmprotocol.WireFormat][]string{
		llmprotocol.OpenAIChatV1:      {`"finish_reason":"length"`},
		llmprotocol.OpenAIResponsesV1: {`"status":"incomplete"`, `"incomplete_details":{"reason":"max_output_tokens"}`},
	}
	for client, fragments := range want {
		for _, streaming := range []bool{false, true} {
			t.Run(string(client)+"/stream="+strconv.FormatBool(streaming), func(t *testing.T) {
				ctx := &RequestContext{SourceFormat: client, TargetFormat: llmprotocol.AnthropicMessagesV1, RequestModel: "public-model", TraceContext: t.Context()}
				var body []byte
				if streaming {
					body = clientResponseBody(t, (&OpenAIRouter{}).handleSemanticStreamingResponseBody([]byte(stream), true, ctx))
				} else {
					body = clientResponseBody(t, (&OpenAIRouter{}).handleNonStreamingResponseBody([]byte(buffered), ctx, 0, false))
				}
				for _, fragment := range fragments {
					if !bytes.Contains(body, []byte(fragment)) {
						t.Errorf("want %s in:\n%s", fragment, body)
					}
				}
				if !slices.ContainsFunc(ctx.ProtocolDiagnostics, func(diagnostic llmprotocol.Diagnostic) bool {
					return diagnostic.Field == "response.stop_reason" && diagnostic.Action == llmprotocol.DiagnosticApproximated
				}) {
					t.Errorf("the length stop is not reported as approximated: %+v", ctx.ProtocolDiagnostics)
				}
			})
		}
	}
}

// logprobs:false asks for the default, so it is no reason to refuse a request.
// logprobs:true stays unsupported.
func TestChatLogprobsFalseIsAccepted(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	for value, accepted := range map[string]bool{"false": true, "true": false} {
		ctx := &RequestContext{Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1, RequestID: "logprobs", TraceContext: t.Context()}
		_, immediate := router.prepareProtocolRequest([]byte(`{"model":"`+model+`","messages":[{"role":"user","content":"hi"}],"logprobs":`+value+`}`), ctx)
		if (immediate == nil) != accepted {
			t.Errorf("logprobs:%s accepted=%t, want %t", value, immediate == nil, accepted)
		}
	}
}

// Messages has no image detail and no document file name. Dropping them is
// reported like every other omitted field.
func TestMessagesBackendReportsDroppedImageDetailAndFilename(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	ctx := &RequestContext{Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1, RequestID: "media", TraceContext: t.Context()}
	request, immediate := router.prepareProtocolRequest([]byte(`{"model":"`+model+`","messages":[{"role":"user","content":[`+
		`{"type":"image_url","image_url":{"url":"https://example.com/a.png","detail":"high"}},`+
		`{"type":"file","file":{"file_data":"data:application/pdf;base64,JVBERi0=","filename":"report.pdf"}}]}]}`), ctx)
	if immediate != nil {
		t.Fatalf("request rejected: %s", immediate.GetImmediateResponse().GetBody())
	}
	if _, err := router.prepareProviderDispatch(request, model, "", false, ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := router.encodeDispatchRequest(ctx); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"content.image.detail", "content.file.filename"} {
		if !slices.ContainsFunc(ctx.ProtocolDiagnostics, func(diagnostic llmprotocol.Diagnostic) bool {
			return diagnostic.Field == field && diagnostic.Action == llmprotocol.DiagnosticDropped
		}) {
			t.Errorf("no dropped diagnostic for %s: %+v", field, ctx.ProtocolDiagnostics)
		}
	}
}
