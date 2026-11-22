package protocolcodec

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// responsesEchoRequest sets every member a Responses object repeats.
const responsesEchoRequest = `{"model":"m","input":"What time is it in Paris?",
	"instructions":"Answer in one sentence.",
	"tools":[{"type":"function","name":"get_time","description":"Current time in a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}}},"strict":false}],
	"tool_choice":{"type":"function","name":"get_time"},"parallel_tool_calls":false,
	"max_output_tokens":64,"temperature":0.2,"top_p":0.9,
	"text":{"format":{"type":"text"},"verbosity":"low"},"reasoning":{"effort":"low","summary":"auto"},
	"metadata":{"session":"s-1"},"truncation":"disabled","user":"user-1","prompt_cache_key":"cache-1"}`

var responsesEchoMembers = map[string]string{
	"instructions":        `"Answer in one sentence."`,
	"tools":               `[{"type":"function","name":"get_time","description":"Current time in a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}}},"strict":false}]`,
	"tool_choice":         `{"type":"function","name":"get_time"}`,
	"parallel_tool_calls": `false`, "max_output_tokens": `64`, "temperature": `0.2`, "top_p": `0.9`,
	"text":      `{"format":{"type":"text"},"verbosity":"low"}`,
	"reasoning": `{"effort":"low","summary":"auto"}`,
	"metadata":  `{"session":"s-1"}`, "truncation": `"disabled"`, "user": `"user-1"`, "prompt_cache_key": `"cache-1"`,
}

var responsesEchoDefaults = map[string]string{
	"instructions": `null`, "tools": `[]`, "tool_choice": `"auto"`, "parallel_tool_calls": `true`,
	"temperature": `null`, "top_p": `null`, "metadata": `{}`,
}

const chatCutReply = `{"id":"chatcmpl-1","object":"chat.completion","created":1792418136,"model":"m",
	"choices":[{"index":0,"message":{"role":"assistant","content":"It is noon"},"finish_reason":"length"}],
	"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`

const chatCutStream = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1792418136,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"It is noon\"},\"finish_reason\":null}]}\n\n" +
	"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1792418136,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n" +
	"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1792418136,\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\n" +
	"data: [DONE]\n\n"

func responsesEchoRender(t *testing.T, body string) llmprotocol.ResponseRenderContext {
	t.Helper()
	_, envelope, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatal(err)
	}
	return envelope.ResponseRender
}

func encodeResponsesObject(t *testing.T, source llmprotocol.WireFormat, reply string, render llmprotocol.ResponseRenderContext) map[string]json.RawMessage {
	t.Helper()
	engine := NewBuiltinEngine()
	response, _, _, err := engine.DecodeResponse(source, []byte(reply))
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, response, llmprotocol.Envelope{ResponseRender: render})
	if err != nil {
		t.Fatal(err)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(encoded.Body, &object); err != nil {
		t.Fatal(err)
	}
	return object
}

type responsesStreamCapture struct {
	objects   map[string]map[string]json.RawMessage
	itemsDone []map[string]json.RawMessage
}

// translateResponsesStream renders a provider stream for a Responses client and
// checks that the client stream satisfies the Responses stream contract.
func translateResponsesStream(t *testing.T, source llmprotocol.WireFormat, render llmprotocol.ResponseRenderContext, body string) responsesStreamCapture {
	t.Helper()
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(source, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "m", ResponseRender: render,
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	frames = append(frames, final...)
	verify, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := verify.Push(slices.Concat(frames...)); err != nil {
		t.Fatalf("client stream breaks the Responses contract: %v\n%s", err, slices.Concat(frames...))
	}
	capture := responsesStreamCapture{objects: map[string]map[string]json.RawMessage{}}
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
			var event struct {
				Type     string                     `json:"type"`
				Response map[string]json.RawMessage `json:"response"`
				Item     map[string]json.RawMessage `json:"item"`
			}
			if err := json.Unmarshal(parsed.Data, &event); err != nil {
				t.Fatal(err)
			}
			if event.Response != nil {
				capture.objects[event.Type] = event.Response
			}
			if event.Type == "response.output_item.done" {
				capture.itemsDone = append(capture.itemsDone, event.Item)
			}
		}
	}
	return capture
}

func assertResponsesMembers(t *testing.T, label string, object map[string]json.RawMessage, want map[string]string) {
	t.Helper()
	for name, value := range want {
		if !jsonEquivalent([]byte(value), object[name]) {
			t.Fatalf("%s %s = %s, want %s", label, name, object[name], value)
		}
	}
}

func outputStatuses(t *testing.T, raw json.RawMessage) []string {
	t.Helper()
	var items []struct {
		Type   string `json:"type"`
		Status string `json:"status"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		t.Fatal(err)
	}
	statuses := make([]string, 0, len(items))
	for _, item := range items {
		statuses = append(statuses, item.Type+":"+item.Status)
	}
	return statuses
}

func TestResponsesObjectRepeatsTheRequestSettings(t *testing.T) {
	render := responsesEchoRender(t, responsesEchoRequest)
	assertResponsesMembers(t, "buffered", encodeResponsesObject(t, llmprotocol.OpenAIChatV1, chatCutReply, render), responsesEchoMembers)

	capture := translateResponsesStream(t, llmprotocol.OpenAIChatV1, render, chatCutStream)
	for _, eventType := range []string{"response.created", "response.in_progress", "response.incomplete"} {
		assertResponsesMembers(t, eventType, capture.objects[eventType], responsesEchoMembers)
	}
}

func TestResponsesObjectKeepsDefaultsForUnsetAndNullSettings(t *testing.T) {
	for name, body := range map[string]string{
		"unset": `{"model":"m","input":"hi"}`,
		"null":  `{"model":"m","input":"hi","instructions":null,"tools":null,"tool_choice":null,"metadata":null,"temperature":null,"top_p":null,"parallel_tool_calls":null,"max_output_tokens":null,"text":null,"reasoning":null}`,
	} {
		t.Run(name, func(t *testing.T) {
			render := responsesEchoRender(t, body)
			if len(render.RequestEcho) != 0 {
				t.Fatalf("request echo = %s, want none", render.RequestEcho)
			}
			object := encodeResponsesObject(t, llmprotocol.OpenAIChatV1, chatCutReply, render)
			assertResponsesMembers(t, "buffered", object, responsesEchoDefaults)
			for _, absent := range []string{"max_output_tokens", "text", "reasoning", "truncation", "user", "prompt_cache_key"} {
				if value, present := object[absent]; present {
					t.Fatalf("%s = %s, want it absent", absent, value)
				}
			}
		})
	}
}

// Overlaying the echo on the response object relies on each repeated member
// having the same name in both objects.
func TestResponsesRequestEchoMembersAreResponseMembers(t *testing.T) {
	members := jsonFieldNames(reflect.TypeOf(responsesResponseWire{}))
	for _, name := range jsonFieldNames(reflect.TypeOf(responsesRequestEchoWire{})) {
		if !slices.Contains(members, name) || slices.Contains([]string{"id", "object", "status", "output", "usage", "model"}, name) {
			t.Fatalf("echo member %q is not a repeated Responses object member", name)
		}
	}
}

func TestResponsesStreamCarriesTheBufferedCreationTime(t *testing.T) {
	responsesStream := strings.ReplaceAll(string(streamFixture(llmprotocol.OpenAIResponsesV1)), `"created_at":100`, `"created_at":1792418136.25`)
	responsesReply := `{"id":"resp_1","object":"response","created_at":1792418136.25,"model":"m","status":"completed",
		"output":[{"type":"message","id":"output_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}],
		"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`
	for _, test := range []struct {
		source        llmprotocol.WireFormat
		reply, stream string
	}{
		{source: llmprotocol.OpenAIChatV1, reply: chatCutReply, stream: chatCutStream},
		{source: llmprotocol.OpenAIResponsesV1, reply: responsesReply, stream: responsesStream},
	} {
		t.Run(string(test.source), func(t *testing.T) {
			buffered := encodeResponsesObject(t, test.source, test.reply, llmprotocol.ResponseRenderContext{})
			if string(buffered["created_at"]) != "1792418136" {
				t.Fatalf("buffered created_at = %s", buffered["created_at"])
			}
			capture := translateResponsesStream(t, test.source, llmprotocol.ResponseRenderContext{}, test.stream)
			if len(capture.objects) < 3 {
				t.Fatalf("stream objects = %v", capture.objects)
			}
			for eventType, object := range capture.objects {
				if string(object["created_at"]) != string(buffered["created_at"]) {
					t.Fatalf("%s created_at = %s, want the buffered %s", eventType, object["created_at"], buffered["created_at"])
				}
			}
		})
	}
}

func TestResponsesMessageCutByTheOutputLimitIsIncompleteBuffered(t *testing.T) {
	for _, test := range []struct {
		name, reply string
		want        []string
	}{
		{name: "cut", reply: chatCutReply, want: []string{"message:incomplete"}},
		{name: "finished", reply: `{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"It is noon"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`, want: []string{"message:completed"}},
		{
			name:  "cut tool call after a message",
			reply: `{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"Let me check.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"city\":\"Par"}}]},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			want:  []string{"message:completed", "function_call:"},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			object := encodeResponsesObject(t, llmprotocol.OpenAIChatV1, test.reply, llmprotocol.ResponseRenderContext{})
			if got := outputStatuses(t, object["output"]); !slices.Equal(got, test.want) {
				t.Fatalf("output statuses = %v, want %v", got, test.want)
			}
		})
	}
}

func TestResponsesMessageCutByTheOutputLimitIsIncompleteStreamed(t *testing.T) {
	anthropicCut := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"It is noon\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	chunk := func(delta, finish string) string {
		return "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":" + delta + ",\"finish_reason\":" + finish + "}]}\n\n"
	}
	usage := "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\ndata: [DONE]\n\n"
	for _, test := range []struct {
		name     string
		source   llmprotocol.WireFormat
		stream   string
		terminal string
		want     []string
	}{
		{name: "chat cut", source: llmprotocol.OpenAIChatV1, stream: chatCutStream, terminal: "response.incomplete", want: []string{"message:incomplete"}},
		{name: "messages cut", source: llmprotocol.AnthropicMessagesV1, stream: anthropicCut, terminal: "response.incomplete", want: []string{"message:incomplete"}},
		{
			name: "chat finished", source: llmprotocol.OpenAIChatV1, terminal: "response.completed", want: []string{"message:completed"},
			stream: chunk(`{"role":"assistant","content":"It is noon"}`, "null") + chunk(`{}`, `"stop"`) + usage,
		},
		{
			name: "chat cut tool call after a message", source: llmprotocol.OpenAIChatV1, terminal: "response.incomplete",
			want: []string{"message:completed", "function_call:completed"},
			stream: chunk(`{"role":"assistant","content":"Let me check."}`, "null") +
				chunk(`{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"city\":\"Par"}}]}`, "null") +
				chunk(`{}`, `"length"`) + usage,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			capture := translateResponsesStream(t, test.source, llmprotocol.ResponseRenderContext{}, test.stream)
			done := make([]string, 0, len(capture.itemsDone))
			for _, item := range capture.itemsDone {
				var kind, status string
				if json.Unmarshal(item["type"], &kind) != nil || json.Unmarshal(item["status"], &status) != nil {
					t.Fatalf("output_item.done item = %v", item)
				}
				done = append(done, kind+":"+status)
			}
			if !slices.Equal(done, test.want) {
				t.Fatalf("output_item.done statuses = %v, want %v", done, test.want)
			}
			terminal := capture.objects[test.terminal]
			if terminal == nil {
				t.Fatalf("stream has no %s: %v", test.terminal, capture.objects)
			}
			if got := outputStatuses(t, terminal["output"]); !slices.Equal(got, test.want) {
				t.Fatalf("%s output statuses = %v, want %v", test.terminal, got, test.want)
			}
		})
	}
}
