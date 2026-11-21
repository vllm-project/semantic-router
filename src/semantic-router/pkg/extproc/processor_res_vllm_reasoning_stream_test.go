package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// vLLM can send the end of the reasoning and the start of the answer in one
// delta, with the blank line its qwen3 parser leaves in front of the answer.
// The fixture is trimmed from a vLLM 0.30.0 capture (Qwen3.8-27B-FP8, qwen3
// reasoning parser, speculative decoding); streamOllamaThroughRouter replays
// any captured Chat stream through the response stream handler, frame by frame.
// The client gets the answer without that blank line.
func TestVLLMReasoningAndAnswerInOneDeltaReachClient(t *testing.T) {
	for _, tc := range []struct {
		client llmprotocol.WireFormat
		want   []string
		opens  string // the event that opens a block or item; the reply has two
	}{
		{llmprotocol.AnthropicMessagesV1, []string{`"thinking":" words. final"`, `"thinking":".\n"`, `"text":"Hello"`, `"text":" there, how are you"`, "event: message_stop"}, "event: content_block_start"},
		{llmprotocol.OpenAIResponsesV1, []string{`"delta":" words. final"`, `"delta":".\n"`, `"delta":"Hello"`, `"delta":" there, how are you"`, "event: response.completed"}, `"type":"response.output_item.added"`},
		// A Chat client gets vLLM's own frames, less the blank line.
		{llmprotocol.OpenAIChatV1, []string{`"content":"Hello","reasoning":".\n"`, "data: [DONE]"}, ""},
	} {
		t.Run(string(tc.client), func(t *testing.T) {
			received := streamOllamaThroughRouter(t, tc.client, "vllm-chat-reasoning-and-answer-one-delta-stream.sse", false)
			if strings.Contains(received, "event: error") {
				t.Fatalf("client got an error event:\n%s", received)
			}
			if strings.Contains(received, `\n\nHello`) {
				t.Fatalf("client got the blank line in front of the answer:\n%s", received)
			}
			if tc.opens != "" && strings.Count(received, tc.opens) != 2 {
				t.Fatalf("want two %q, one for the reasoning and one for the answer:\n%s", tc.opens, received)
			}
			last := -1
			for _, part := range tc.want {
				index := strings.Index(received, part)
				if index <= last {
					t.Fatalf("want %q in order:\n%s", tc.want, received)
				}
				last = index
			}
		})
	}
}
