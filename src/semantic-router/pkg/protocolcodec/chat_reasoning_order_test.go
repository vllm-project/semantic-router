package protocolcodec

import (
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

// Outside the reasoning, a mixed delta keeps main's order, content first.
func TestChatDeltaOutsideReasoningKeepsContentFirst(t *testing.T) {
	answerStarted := chatStream(`{"role":"assistant","content":"Hello"}`, `{"content":" there","reasoning":"Check"}`, `{"reasoning":" again."}`)
	opening := chatStream(`{"role":"assistant","content":"Hello there","reasoning":"Check"}`, `{"reasoning":" again."}`)
	answerResumed := chatStream(`{"role":"assistant","content":"Hello"}`, `{"reasoning":"Check"}`, `{"content":" there"}`, `{"content":" friend","reasoning":" again."}`)
	for _, tc := range []struct {
		name   string
		stream string
		target llmprotocol.WireFormat
		want   []string
	}{
		{"mixed after the answer started", answerStarted, llmprotocol.AnthropicMessagesV1, []string{`"text":" there"`, `"thinking":"Check"`, `"thinking":" again."`}},
		{"mixed after the answer started", answerStarted, llmprotocol.OpenAIResponsesV1, []string{`"delta":" there"`, `"delta":"Check"`, `"delta":" again."`}},
		{"mixed opening delta", opening, llmprotocol.AnthropicMessagesV1, []string{`"text":"Hello there"`, `"thinking":"Check"`, `"thinking":" again."`}},
		{"mixed opening delta", opening, llmprotocol.OpenAIResponsesV1, []string{`"delta":"Hello there"`, `"delta":"Check"`, `"delta":" again."`}},
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
