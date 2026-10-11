package protocolcodec

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// chatStream builds a Chat stream from delta objects; the last one finishes it.
func chatStream(deltas ...string) string {
	var out strings.Builder
	for index, delta := range deltas {
		finish := "null"
		if index == len(deltas)-1 {
			finish = `"stop"`
		}
		out.WriteString(`data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` + delta + `,"finish_reason":` + finish + "}]}\n\n")
	}
	return out.String() + "data: [DONE]\n\n"
}

// assertChatStreamOrder translates a Chat stream and checks that the parts
// arrive in order and the stream completes without an error event.
func assertChatStreamOrder(t *testing.T, stream string, target llmprotocol.WireFormat, parts ...string) {
	t.Helper()
	out := translateMatchedStopStream(t, llmprotocol.OpenAIChatV1, target, stream)
	if strings.Contains(out, "event: error") {
		t.Fatalf("stream ended in an error event:\n%s", out)
	}
	terminal := map[llmprotocol.WireFormat]string{
		llmprotocol.AnthropicMessagesV1: "event: message_stop",
		llmprotocol.OpenAIResponsesV1:   "event: response.completed",
	}[target]
	assertInOrder(t, out, append(parts, terminal)...)
}

func assertInOrder(t *testing.T, out string, parts ...string) {
	t.Helper()
	last := -1
	for _, part := range parts {
		index := strings.Index(out, part)
		if index <= last {
			t.Fatalf("want %q in order:\n%s", parts, out)
		}
		last = index
	}
}

// vLLM can send the end of the reasoning and the start of the answer in one
// delta (#4555). With either reasoning field, the tail comes before the answer.
func TestChatReasoningTailInMixedDeltaGoesFirst(t *testing.T) {
	for _, field := range []string{"reasoning", "reasoning_content"} {
		stream := chatStream(`{"role":"assistant","`+field+`":"Check"}`, `{"content":"Hello","`+field+`":" again."}`, `{"content":" there"}`)
		for target, want := range map[llmprotocol.WireFormat][]string{
			llmprotocol.AnthropicMessagesV1: {`"thinking":"Check"`, `"thinking":" again."`, `"text":"Hello"`, `"text":" there"`},
			llmprotocol.OpenAIResponsesV1:   {`"delta":"Check"`, `"delta":" again."`, `"delta":"Hello"`, `"delta":" there"`},
		} {
			t.Run(field+"/"+string(target), func(t *testing.T) {
				assertChatStreamOrder(t, stream, target, want...)
			})
		}
	}
}

// A short think block can fit whole in the first delta with text, reasoning
// before the answer (vLLM's basic_parsers.py L134-L145 at 5dd4a628). After
// vLLM's empty role delta, the stream still completes, reasoning first.
func TestChatOpeningDeltaWithReasoningAndAnswerCompletes(t *testing.T) {
	for _, field := range []string{"reasoning", "reasoning_content"} {
		stream := chatStream(`{"role":"assistant","content":""}`, `{"`+field+`":"Short.","content":"Hello"}`, `{"content":" there"}`)
		for target, want := range map[llmprotocol.WireFormat][]string{
			llmprotocol.AnthropicMessagesV1: {`"thinking":"Short."`, `"text":"Hello"`, `"text":" there"`},
			llmprotocol.OpenAIResponsesV1:   {`"delta":"Short."`, `"delta":"Hello"`, `"delta":" there"`},
		} {
			t.Run(field+"/"+string(target), func(t *testing.T) {
				assertChatStreamOrder(t, stream, target, want...)
			})
		}
	}
}

// Mistral's parser starts in content, so its first delta with text can hold
// the answer and then the start of a think block (mistral.py L166-L195 at
// v0.30.0). Reasoning alone in the next delta shows the answer came first.
func TestChatOpeningDeltaWithAnswerBeforeReasoningCompletes(t *testing.T) {
	for _, field := range []string{"reasoning", "reasoning_content"} {
		stream := chatStream(`{"role":"assistant","content":""}`, `{"content":"Hello there","`+field+`":"Check"}`, `{"`+field+`":" again."}`)
		for target, want := range map[llmprotocol.WireFormat][]string{
			llmprotocol.AnthropicMessagesV1: {`"text":"Hello there"`, `"thinking":"Check"`, `"thinking":" again."`},
			llmprotocol.OpenAIResponsesV1:   {`"delta":"Hello there"`, `"delta":"Check"`, `"delta":" again."`},
		} {
			t.Run(field+"/"+string(target), func(t *testing.T) {
				assertChatStreamOrder(t, stream, target, want...)
			})
		}
	}
}

// The decoder holds the answer and reasoning of a first delta that has both
// until the next delta with output; the rest of that delta, and usage in
// between, decodes at once. Every way the stream can go on or end releases
// them, reasoning first unless reasoning alone comes next, and before any
// later event or error. Each row gives main's events and error codes, in
// another order or Push. "|" marks the end of each Push.
func TestChatHeldOpeningDeltaIsReleasedOnEveryExit(t *testing.T) {
	frame := func(fields string) string {
		return `data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":1,"model":"m",` + fields + "}\n\n"
	}
	choice := func(delta, finish string) string {
		return frame(`"choices":[{"index":0,"delta":` + delta + `,"finish_reason":` + finish + `}]`)
	}
	usage := `"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}`
	opening := choice(`{"role":"assistant","reasoning":"Short.","content":"Hello"}`, "null")
	reasoningNext := choice(`{"reasoning":" again."}`, `"stop"`)
	done := "data: [DONE]\n\n"
	for _, tc := range []struct {
		name   string
		frames []string
		want   string
		limits func(*llmprotocol.Limits)
	}{
		{"finish in the same delta", []string{choice(`{"role":"assistant","reasoning":"Short.","content":"Hello"}`, `"stop"`), done}, "thinking:Short. text:Hello | completed |", nil},
		{"answer next", []string{opening, choice(`{"content":" there"}`, `"stop"`), done}, "| thinking:Short. text:Hello text: there | completed |", nil},
		{"reasoning next", []string{opening, reasoningNext, done}, "| text:Hello thinking:Short. thinking: again. | completed |", nil},
		{"tool call next", []string{opening, choice(`{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]}`, `"tool_calls"`), done}, "| thinking:Short. text:Hello tool | completed |", nil},
		{"finish only next", []string{opening, choice(`{}`, `"stop"`), done}, "| thinking:Short. text:Hello | completed |", nil},
		{"empty delta keeps the hold", []string{opening, choice(`{"content":""}`, "null"), reasoningNext, done}, "| | text:Hello thinking:Short. thinking: again. | completed |", nil},
		{"empty delta with usage keeps the hold", []string{opening, frame(`"choices":[{"index":0,"delta":{},"finish_reason":null}],` + usage), reasoningNext, done}, "| usage | text:Hello thinking:Short. thinking: again. | completed |", nil},
		{"usage alone keeps the hold", []string{opening, frame(`"choices":[],` + usage), reasoningNext, done}, "| usage | text:Hello thinking:Short. thinking: again. | completed |", nil},
		{"done without a finish", []string{opening, done}, "| thinking:Short. text:Hello error:stream_item_incomplete", nil},
		{"error chunk", []string{opening, `data: {"error":{"message":"overloaded","type":"server_error"}}` + "\n\n"}, "| thinking:Short. text:Hello failed |", nil},
		{"malformed frame", []string{opening, "data: {\"id\":\n\n"}, "| thinking:Short. text:Hello error:invalid_upstream_json", nil},
		{"stream ends", []string{opening}, "| thinking:Short. text:Hello failed", nil},
		{"annotation waits with the texts", []string{choice(`{"role":"assistant","content":"Hello","reasoning":"Short.","annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com","title":"t","start_index":0,"end_index":5}}]}`, "null"), reasoningNext, done}, "| text:Hello text: thinking:Short. thinking: again. | completed |", nil},
		{"tool call waits with the texts", []string{choice(`{"role":"assistant","content":"Hello","reasoning":"Short.","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]}`, "null"), choice(`{"reasoning":" again."}`, `"tool_calls"`), done}, "| text:Hello thinking:Short. tool thinking: again. | completed |", nil},
		{"annotation without answer text keeps main's order", []string{choice(`{"role":"assistant","reasoning":"R","annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com","title":"t","start_index":0,"end_index":0}}]}`, "null"), choice(`{"reasoning":"R2"}`, `"stop"`), done}, "text: thinking:R | thinking:R2 | completed |", nil},
		{"error in the rest of the held delta", []string{frame(`"choices":[{"index":0,"delta":{"role":"assistant","reasoning":"Short.","content":"Hello"},"finish_reason":null}],"usage":{"prompt_tokens":-1,"completion_tokens":1,"total_tokens":0}`), done}, "error:negative_usage", nil},
		{"error in the held texts comes first", []string{opening, "data: {\"id\":\n\n"}, "| error:text_limit", func(limits *llmprotocol.Limits) { limits.TextBytes = 3 }},
		{"held annotations keep their event budget", []string{choice(`{"role":"assistant","content":"Hello","reasoning":"Short.","annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com","title":"t","start_index":0,"end_index":5}}]}`, "null"), frame(`"choices":[],` + usage), frame(`"choices":[],` + usage), frame(`"choices":[],` + usage)}, "| usage | thinking:Short. text:Hello text: usage | error:stream_event_limit", func(limits *llmprotocol.Limits) { limits.Events = 7 }},
		{"held texts keep their event budget", []string{opening, frame(`"choices":[],` + usage), frame(`"choices":[],` + usage), frame(`"choices":[],` + usage)}, "| thinking:Short. text:Hello usage | usage | error:stream_event_limit", func(limits *llmprotocol.Limits) { limits.Events = 6 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			policy := llmprotocol.DefaultPolicy()
			if tc.limits != nil {
				tc.limits(&policy.Limits)
			}
			decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, policy)
			var trace []string
			record := func(events []llmprotocol.Event, err error) bool {
				for _, event := range events {
					switch event.Type {
					case llmprotocol.EventReasoningDelta:
						trace = append(trace, "thinking:"+event.Delta)
					case llmprotocol.EventOutputTextDelta:
						trace = append(trace, "text:"+event.Delta)
					case llmprotocol.EventToolCallDelta:
						trace = append(trace, "tool")
					case llmprotocol.EventUsageUpdated:
						trace = append(trace, "usage")
					case llmprotocol.EventResponseCompleted:
						trace = append(trace, "completed")
					case llmprotocol.EventResponseFailed:
						trace = append(trace, "failed")
					}
				}
				var protocolError *llmprotocol.ProtocolError
				if errors.As(err, &protocolError) {
					trace = append(trace, "error:"+protocolError.Code)
				} else if err != nil {
					trace = append(trace, "error:"+err.Error())
				}
				return err == nil
			}
			ok := true
			for _, raw := range tc.frames {
				events, _, err := decoder.Push([]byte(raw))
				if ok = record(events, err); !ok {
					break
				}
				trace = append(trace, "|")
			}
			if ok {
				events, _, err := decoder.Finalize(nil)
				record(events, err)
			}
			if got := strings.Join(trace, " "); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// Once the answer is under way, a mixed delta keeps main's order, content first.
func TestChatDeltaOutsideReasoningKeepsContentFirst(t *testing.T) {
	answerStarted := chatStream(`{"role":"assistant","content":"Hello"}`, `{"content":" there","reasoning":"Check"}`, `{"reasoning":" again."}`)
	answerResumed := chatStream(`{"role":"assistant","content":"Hello"}`, `{"reasoning":"Check"}`, `{"content":" there"}`, `{"content":" friend","reasoning":" again."}`)
	for _, tc := range []struct {
		name   string
		stream string
		target llmprotocol.WireFormat
		want   []string
	}{
		{"mixed after the answer started", answerStarted, llmprotocol.AnthropicMessagesV1, []string{`"text":" there"`, `"thinking":"Check"`, `"thinking":" again."`}},
		{"mixed after the answer started", answerStarted, llmprotocol.OpenAIResponsesV1, []string{`"delta":" there"`, `"delta":"Check"`, `"delta":" again."`}},
		// Messages cannot reopen the text block after the reasoning, on main or
		// here, so this row is Responses only.
		{"mixed after the answer resumed", answerResumed, llmprotocol.OpenAIResponsesV1, []string{`"delta":" there"`, `"delta":" friend"`, `"delta":" again."`}},
	} {
		t.Run(tc.name+"/"+string(tc.target), func(t *testing.T) {
			assertChatStreamOrder(t, tc.stream, tc.target, tc.want...)
		})
	}
}

// A buffered vLLM reply puts the reasoning in message.reasoning and the answer
// in message.content; the client should get them in that order too.
func TestVLLMBufferedReasoningPrecedesAnswer(t *testing.T) {
	body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"Qwen/Qwen3.8-27B-FP8","choices":[{"index":0,"message":{"role":"assistant","content":"Hello there, how are you?","reasoning":"Need five words."},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":15,"completion_tokens":30,"total_tokens":45}}`)
	for target, want := range map[llmprotocol.WireFormat][]string{
		llmprotocol.AnthropicMessagesV1: {`"thinking":"Need five words."`, `"text":"Hello there, how are you?"`},
		llmprotocol.OpenAIResponsesV1:   {`"text":"Need five words."`, `"type":"reasoning"`, `"text":"Hello there, how are you?"`, `"type":"message"`},
	} {
		t.Run(string(target), func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
			if err != nil {
				t.Fatal(err)
			}
			assertInOrder(t, string(result.Body), want...)
		})
	}
}

// Chat requests decode assistant history through the same assembler, so a
// replayed turn's reasoning also comes before its answer.
func TestChatRequestAssistantReasoningPrecedesAnswer(t *testing.T) {
	body := []byte(`{"model":"m","max_tokens":50,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"Hello there","reasoning":"Need five words."},{"role":"user","content":"again"}]}`)
	for target, want := range map[llmprotocol.WireFormat][]string{
		llmprotocol.AnthropicMessagesV1: {`"text":"hi"`, `"thinking":"Need five words."`, `"text":"Hello there"`, `"text":"again"`},
		llmprotocol.OpenAIResponsesV1:   {`"text":"hi"`, `"text":"Need five words."`, `"type":"reasoning"`, `"text":"Hello there"`, `"text":"again"`},
	} {
		t.Run(string(target), func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIChatV1, target, body, nil)
			if err != nil {
				t.Fatal(err)
			}
			assertInOrder(t, string(result.Body), want...)
		})
	}
}
