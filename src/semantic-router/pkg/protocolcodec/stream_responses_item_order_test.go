package protocolcodec

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// chatItemOrderStream builds a Chat stream from delta objects. The last delta
// ends the choice with finish, and a usage chunk follows.
func chatItemOrderStream(finish string, deltas ...string) string {
	var out strings.Builder
	for index, delta := range deltas {
		reason := "null"
		if index == len(deltas)-1 {
			reason = `"` + finish + `"`
		}
		out.WriteString(`data: {"id":"chatcmpl-order","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` +
			delta + `,"finish_reason":` + reason + "}]}\n\n")
	}
	out.WriteString(`data: {"id":"chatcmpl-order","object":"chat.completion.chunk","created":1,"model":"m","choices":[],` +
		`"usage":{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14}}` + "\n\n")
	return out.String() + "data: [DONE]\n\n"
}

type responsesItemOrderEvent struct {
	Type        string `json:"type"`
	OutputIndex *int   `json:"output_index"`
	Delta       string `json:"delta"`
	Item        struct {
		ID     string `json:"id"`
		Type   string `json:"type"`
		Status string `json:"status"`
	} `json:"item"`
	Part struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"part"`
	Response struct {
		Output []json.RawMessage `json:"output"`
	} `json:"response"`
}

type responsesItemOrder struct {
	trace    []string
	itemIDs  []string
	terminal []json.RawMessage
}

// translateResponsesItemOrder renders a provider stream for a Responses
// client and lists the stream's item events in order, each as its output
// index and event. A stream that translates is re-decoded under the
// Responses stream contract.
func translateResponsesItemOrder(t *testing.T, source llmprotocol.WireFormat, body string) (responsesItemOrder, error) {
	t.Helper()
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(source, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "m",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, streamErr := stream.Push([]byte(body))
	if streamErr == nil {
		var final [][]byte
		final, _, _, streamErr = stream.Finalize(nil)
		frames = append(frames, final...)
	}
	if streamErr == nil {
		verify, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
			llmprotocol.StreamContext{Context: context.Background(), PublicModel: "m"})
		if err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := verify.Push(slices.Concat(frames...)); err != nil {
			t.Fatalf("client stream breaks the Responses contract: %v\n%s", err, slices.Concat(frames...))
		}
	}
	var order responsesItemOrder
	framer := newSSEFramer(8 << 20)
	for _, wireFrame := range frames {
		parsedFrames, err := framer.Push(wireFrame)
		if err != nil {
			t.Fatal(err)
		}
		for _, frame := range parsedFrames {
			parsed, err := parseSSEFrame(frame, 8<<20)
			if err != nil {
				t.Fatal(err)
			}
			var event responsesItemOrderEvent
			if err := json.Unmarshal(parsed.Data, &event); err != nil {
				t.Fatal(err)
			}
			order.record(event)
		}
	}
	return order, streamErr
}

func (order *responsesItemOrder) record(event responsesItemOrderEvent) {
	name := strings.TrimPrefix(event.Type, "response.")
	switch event.Type {
	case "response.created", "response.in_progress":
		return
	case "response.completed", "response.incomplete":
		order.trace = append(order.trace, name)
		order.terminal = event.Response.Output
		return
	case "response.output_item.added":
		name = "+" + event.Item.Type
		order.itemIDs = append(order.itemIDs, event.Item.ID)
	case "response.output_item.done":
		name = "-" + event.Item.Type + ":" + event.Item.Status
	case "response.content_part.added":
		name = "+part:" + event.Part.Type
	case "response.content_part.done":
		name = "-part:" + event.Part.Type
	}
	if event.Delta != "" {
		name += " " + event.Delta
	}
	if event.OutputIndex != nil {
		name = fmt.Sprintf("%d %s", *event.OutputIndex, name)
	}
	order.trace = append(order.trace, name)
}

func assertResponsesItemOrder(t *testing.T, got responsesItemOrder, want []string) {
	t.Helper()
	if !slices.Equal(got.trace, want) {
		t.Fatalf("Responses stream events:\n%s\nwant:\n%s", strings.Join(got.trace, "\n"), strings.Join(want, "\n"))
	}
	if ids := slices.Compact(slices.Sorted(slices.Values(got.itemIDs))); len(ids) != len(got.itemIDs) {
		t.Fatalf("output items share an ID: %v", got.itemIDs)
	}
}

// A Chat backend ends every output of its choice only when its stream ends.
// A Responses client still gets each item's done events before the next
// item starts, as OpenAI streams them, and the terminal response holds them
// all.
func TestResponsesStreamFromChatClosesEachItemBeforeTheNext(t *testing.T) {
	for _, test := range []struct {
		name   string
		stream string
		want   []string
	}{
		{
			name: "reasoning then message",
			stream: chatItemOrderStream("stop",
				`{"role":"assistant","reasoning_content":"Think."}`, `{"reasoning_content":" More."}`,
				`{"content":"Hi"}`, `{"content":" there"}`),
			want: []string{
				"0 +reasoning", "0 +part:reasoning_text", "0 reasoning_text.delta Think.", "0 reasoning_text.delta  More.",
				"0 reasoning_text.done", "0 -part:reasoning_text", "0 -reasoning:completed",
				"1 +message", "1 +part:output_text", "1 output_text.delta Hi", "1 output_text.delta  there",
				"1 output_text.done", "1 -part:output_text", "1 -message:completed",
				"completed",
			},
		},
		{
			name: "message then function call",
			stream: chatItemOrderStream("tool_calls",
				`{"role":"assistant","content":"Let me check."}`,
				`{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"city\":"}}]}`,
				`{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}`),
			want: []string{
				"0 +message", "0 +part:output_text", "0 output_text.delta Let me check.",
				"0 output_text.done", "0 -part:output_text", "0 -message:completed",
				"1 +function_call", `1 function_call_arguments.delta {"city":`, `1 function_call_arguments.delta "Paris"}`,
				"1 function_call_arguments.done", "1 -function_call:completed",
				"completed",
			},
		},
		{
			name: "one call then the next",
			stream: chatItemOrderStream("tool_calls",
				`{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":""}}]}`,
				`{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"Paris\"}"}}]}`,
				`{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]}`,
				`{"tool_calls":[{"index":1,"function":{"arguments":"\"Rome\"}"}}]}`),
			want: []string{
				"0 +function_call", `0 function_call_arguments.delta {"city":"Paris"}`,
				"0 function_call_arguments.done", "0 -function_call:completed",
				"1 +function_call", `1 function_call_arguments.delta {"city":`, `1 function_call_arguments.delta "Rome"}`,
				"1 function_call_arguments.done", "1 -function_call:completed",
				"completed",
			},
		},
		{
			name: "reasoning, a call, then text",
			stream: chatItemOrderStream("tool_calls",
				`{"role":"assistant","reasoning_content":"Look it up."}`,
				`{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{}"}}]}`,
				`{"content":"Checking."}`),
			want: []string{
				"0 +reasoning", "0 +part:reasoning_text", "0 reasoning_text.delta Look it up.",
				"0 reasoning_text.done", "0 -part:reasoning_text", "0 -reasoning:completed",
				"1 +function_call", "1 function_call_arguments.delta {}", "1 function_call_arguments.done", "1 -function_call:completed",
				"2 +message", "2 +part:output_text", "2 output_text.delta Checking.",
				"2 output_text.done", "2 -part:output_text", "2 -message:completed",
				"completed",
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			order, err := translateResponsesItemOrder(t, llmprotocol.OpenAIChatV1, test.stream)
			if err != nil {
				t.Fatal(err)
			}
			assertResponsesItemOrder(t, order, test.want)
			if len(order.terminal) != len(order.itemIDs) {
				t.Fatalf("terminal output holds %d items, the stream %d", len(order.terminal), len(order.itemIDs))
			}
		})
	}
}

// Content that comes back to an output the stream already closed goes into a
// new item of its kind, after the item the stream is in.
func TestResponsesStreamFromChatResumedContentStartsANewItem(t *testing.T) {
	order, err := translateResponsesItemOrder(t, llmprotocol.OpenAIChatV1, chatItemOrderStream("stop",
		`{"role":"assistant","content":"Hello"}`, `{"reasoning_content":"Check"}`, `{"content":" there"}`))
	if err != nil {
		t.Fatal(err)
	}
	assertResponsesItemOrder(t, order, []string{
		"0 +message", "0 +part:output_text", "0 output_text.delta Hello",
		"0 output_text.done", "0 -part:output_text", "0 -message:completed",
		"1 +reasoning", "1 +part:reasoning_text", "1 reasoning_text.delta Check",
		"1 reasoning_text.done", "1 -part:reasoning_text", "1 -reasoning:completed",
		"2 +message", "2 +part:output_text", "2 output_text.delta  there",
		"2 output_text.done", "2 -part:output_text", "2 -message:completed",
		"completed",
	})
	var texts []string
	for _, raw := range order.terminal {
		var item struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		}
		if err := json.Unmarshal(raw, &item); err != nil || len(item.Content) != 1 {
			t.Fatalf("terminal item = %s", raw)
		}
		texts = append(texts, item.Content[0].Text)
	}
	if !slices.Equal(texts, []string{"Hello", "Check", " there"}) {
		t.Fatalf("terminal item texts = %q", texts)
	}
}

// Fields a Chat backend sends empty open no item and leave the stream in the
// item it is in.
func TestResponsesStreamFromChatEmptyFieldsOpenNoItem(t *testing.T) {
	order, err := translateResponsesItemOrder(t, llmprotocol.OpenAIChatV1, chatItemOrderStream("stop",
		`{"role":"assistant","reasoning_content":"","content":"Hi"}`, `{"content":" there","reasoning_content":""}`, `{"refusal":""}`))
	if err != nil {
		t.Fatal(err)
	}
	assertResponsesItemOrder(t, order, []string{
		"0 +message", "0 +part:output_text", "0 output_text.delta Hi", "0 output_text.delta  there",
		"0 output_text.done", "0 -part:output_text", "0 -message:completed",
		"completed",
	})
}

// A call whose arguments do not form a JSON object yet may still grow, so it
// stays open while the backend streams another call; each closes once its
// arguments are whole and the backend has moved on.
func TestResponsesStreamFromChatKeepsAnUnfinishedCallOpen(t *testing.T) {
	order, err := translateResponsesItemOrder(t, llmprotocol.OpenAIChatV1, chatItemOrderStream("tool_calls",
		`{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"city\":"}},`+
			`{"index":1,"id":"call_2","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]}`,
		`{"tool_calls":[{"index":1,"function":{"arguments":"\"Rome\"}"}},{"index":0,"function":{"arguments":"\"Paris\"}"}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	assertResponsesItemOrder(t, order, []string{
		"0 +function_call", `0 function_call_arguments.delta {"city":`,
		"1 +function_call", `1 function_call_arguments.delta {"city":`, `1 function_call_arguments.delta "Rome"}`,
		"1 function_call_arguments.done", "1 -function_call:completed",
		`0 function_call_arguments.delta "Paris"}`, "0 function_call_arguments.done", "0 -function_call:completed",
		"completed",
	})
}

// Arguments that continue a call after its JSON object closed and the backend
// moved on are not a JSON object; the stream fails before any client event
// shows them.
func TestResponsesStreamFromChatRejectsACallExtendedAfterItClosed(t *testing.T) {
	order, err := translateResponsesItemOrder(t, llmprotocol.OpenAIChatV1, chatItemOrderStream("tool_calls",
		`{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"a\":1}"}}]}`,
		`{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"get_weather","arguments":"{}"}}]}`,
		`{"tool_calls":[{"index":0,"function":{"arguments":",\"b\":2}"}}]}`, `{}`))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_tool_arguments")
	if slices.ContainsFunc(order.trace, func(event string) bool { return strings.Contains(event, `"b"`) }) {
		t.Fatalf("the extended arguments reached the client: %v", order.trace)
	}
}

// A message the output limit ends is incomplete when it is the last item,
// whatever item the backend finished first.
func TestResponsesStreamFromChatCutMessageAfterACallIsIncomplete(t *testing.T) {
	order, err := translateResponsesItemOrder(t, llmprotocol.OpenAIChatV1, chatItemOrderStream("length",
		`{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{}"}}]}`,
		`{"content":"It is"}`))
	if err != nil {
		t.Fatal(err)
	}
	assertResponsesItemOrder(t, order, []string{
		"0 +function_call", "0 function_call_arguments.delta {}", "0 function_call_arguments.done", "0 -function_call:completed",
		"1 +message", "1 +part:output_text", "1 output_text.delta It is",
		"1 output_text.done", "1 -part:output_text", "1 -message:incomplete",
		"incomplete",
	})
}

// Reasoning text is a content part of its item in both directions: a
// Responses backend's reasoning_text parts reach a Responses client as
// content parts, which the client stream's own decoder accepts.
func TestResponsesStreamReasoningTextIsAContentPart(t *testing.T) {
	body, err := os.ReadFile("testdata/responses-reasoning-parts.sse")
	if err != nil {
		t.Fatal(err)
	}
	order, err := translateResponsesItemOrder(t, llmprotocol.OpenAIResponsesV1, string(body))
	if err != nil {
		t.Fatal(err)
	}
	assertResponsesItemOrder(t, order, []string{
		"0 +reasoning", "0 reasoning_summary_part.added", "0 reasoning_summary_text.delta summary",
		"0 +part:reasoning_text", "0 reasoning_text.delta think",
		"0 reasoning_summary_text.done", "0 reasoning_summary_part.done",
		"0 reasoning_text.done", "0 -part:reasoning_text", "0 -reasoning:completed",
		"1 +message", "1 +part:output_text", "1 output_text.delta done",
		"1 output_text.done", "1 -part:output_text", "1 -message:completed",
		"completed",
	})
}
