package extproc

import (
	"bytes"
	"slices"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Chat backend's buffered reply that the output limit cut inside a tool call
// reaches every client with the limit stop, where the Router answered 502:
// cut inside the arguments, right after the name, and in the second of two
// parallel calls.
func TestBufferedReplyCutInsideAToolCallReachesEveryClient(t *testing.T) {
	reply := func(calls string) string {
		return `{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant",` +
			`"content":null,"tool_calls":[` + calls + `]},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":24,"total_tokens":34}}`
	}
	call := func(id, arguments string) string {
		return `{"id":"` + id + `","type":"function","function":{"name":"get_weather","arguments":` + arguments + `}}`
	}
	cuts := map[string]struct {
		body     string
		chat     string
		messages string
	}{
		"inside the arguments": {reply(call("call_1", `"{\"city\": \"Par"`)), `"arguments":"{\"city\": \"Par"`, `"input":{}`},
		"after the name":       {reply(call("call_1", `""`)), `"arguments":""`, `"input":{}`},
		"in the second call": {
			reply(call("call_1", `"{\"city\": \"Paris\"}"`) + "," + call("call_2", `"{\"city\": \"Lyon\", \"unit\": \"cel"`)),
			`"arguments":"{\"city\": \"Lyon\", \"unit\": \"cel"`, `"input":{"city":"Lyon"}`,
		},
	}
	for name, cut := range cuts {
		want := map[llmprotocol.WireFormat][]string{
			llmprotocol.OpenAIChatV1:        {`"finish_reason":"length"`, cut.chat},
			llmprotocol.OpenAIResponsesV1:   {`"status":"incomplete"`, `"incomplete_details":{"reason":"max_output_tokens"}`, cut.chat},
			llmprotocol.AnthropicMessagesV1: {`"stop_reason":"max_tokens"`, cut.messages},
		}
		for client, fragments := range want {
			t.Run(name+"/"+string(client), func(t *testing.T) {
				ctx := &RequestContext{SourceFormat: client, TargetFormat: llmprotocol.OpenAIChatV1, RequestModel: "public-model", TraceContext: t.Context()}
				body := deliveredBody(t, (&OpenAIRouter{}).handleNonStreamingResponseBody([]byte(cut.body), ctx, 0, false), cut.body)
				for _, fragment := range fragments {
					if !bytes.Contains(body, []byte(fragment)) {
						t.Errorf("want %s in:\n%s", fragment, body)
					}
				}
			})
		}
	}
}

// vLLM's last chunk carries a call's latest fragment and its tool parser's
// flush of the rest as two entries with one index. Every client gets the call
// and the limit stop, where the Router sent a duplicate_stream_tool_index
// error frame.
func TestStreamedToolEntriesThatShareAnIndexReachEveryClient(t *testing.T) {
	stream := "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"get_weather\"}}]},\"logprobs\":null,\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\": \\\"\"}}]},\"logprobs\":null,\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"Par\"}},{\"index\":0,\"function\":{\"arguments\":\"is\\\"}\"}}]},\"logprobs\":null,\"finish_reason\":\"length\",\"stop_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":24,\"total_tokens\":34}}\n\n" +
		"data: [DONE]\n\n"
	want := map[llmprotocol.WireFormat][]string{
		llmprotocol.OpenAIChatV1:        {`"finish_reason":"length"`},
		llmprotocol.OpenAIResponsesV1:   {`"type":"response.incomplete"`, `"arguments":"{\"city\": \"Paris\"}"`},
		llmprotocol.AnthropicMessagesV1: {`"stop_reason":"max_tokens"`, `"partial_json":"{\"city\": \"Paris\"}"`},
	}
	for client, fragments := range want {
		t.Run(string(client), func(t *testing.T) {
			ctx := &RequestContext{SourceFormat: client, TargetFormat: llmprotocol.OpenAIChatV1, RequestModel: "public-model", TraceContext: t.Context()}
			body := deliveredBody(t, (&OpenAIRouter{}).handleSemanticStreamingResponseBody([]byte(stream), true, ctx), stream)
			for _, failure := range []string{`data: {"error"`, "event: error", "event: response.failed"} {
				if bytes.Contains(body, []byte(failure)) {
					t.Fatalf("the stream ended in an error:\n%s", body)
				}
			}
			for _, fragment := range fragments {
				if !bytes.Contains(body, []byte(fragment)) {
					t.Errorf("want %s in:\n%s", fragment, body)
				}
			}
			if client != llmprotocol.OpenAIChatV1 {
				return
			}
			calls := []joinedChatToolCall{{"call_1", "function", "get_weather", `{"city": "Paris"}`}}
			if got := joinChatStreamToolCalls(t, body); !slices.Equal(got, calls) {
				t.Fatalf("joined tool calls = %+v, want %+v", got, calls)
			}
		})
	}
}

// deliveredBody is what the client gets: the Router's rewrite of the body, or
// the backend's bytes when the Router passes them on.
func deliveredBody(t *testing.T, reply *ext_proc.ProcessingResponse, upstream string) []byte {
	t.Helper()
	if immediate := reply.GetImmediateResponse(); immediate != nil {
		t.Fatalf("response was replaced: status=%v body=%s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	mutation := reply.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		return []byte(upstream)
	}
	if streamed := mutation.GetStreamedResponse(); streamed != nil {
		return streamed.GetBody()
	}
	return mutation.GetBody()
}
