package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// thinkingShortAnswerStream is an extended-thinking stream whose visible
// answer is shorter than the provisional output count in message_start, and
// whose message_delta reports the thinking count for the first time.
func thinkingShortAnswerStream(startUsage, deltaUsage string) []byte {
	return thinkingStream(`{"type":"signature_delta","signature":"sig-1"}`, startUsage, deltaUsage)
}

func thinkingStream(thinkingDelta, startUsage, deltaUsage string) []byte {
	return []byte("event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":` + startUsage + `}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":` + thinkingDelta + `}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"42"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":1}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":` + deltaUsage + `}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n")
}

func TestAnthropicStreamTakesTheOutputSplitFromMessageDelta(t *testing.T) {
	tests := []struct {
		name                    string
		start, delta            string
		total, reasoning, other int64
	}{
		{
			name:  "thinking count first reported in message_delta",
			start: `{"input_tokens":12,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":7}`,
			delta: `{"output_tokens":30,"output_tokens_details":{"thinking_tokens":26}}`,
			total: 30, reasoning: 26, other: 4,
		},
		{
			// A Router hop re-encodes message_start with an explicit zero.
			name:  "explicit zero thinking count in message_start",
			start: `{"input_tokens":12,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":7,"output_tokens_details":{"thinking_tokens":0}}`,
			delta: `{"output_tokens":30,"output_tokens_details":{"thinking_tokens":26}}`,
			total: 30, reasoning: 26, other: 4,
		},
		{
			name:  "answer longer than the provisional count",
			start: `{"input_tokens":12,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":7}`,
			delta: `{"output_tokens":60,"output_tokens_details":{"thinking_tokens":26}}`,
			total: 60, reasoning: 26, other: 34,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.AnthropicMessagesV1,
				thinkingShortAnswerStream(test.start, test.delta),
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
			if err != nil {
				t.Fatalf("stream aborted: %v", err)
			}
			for name, count := range map[string]struct {
				got  *int64
				want int64
			}{
				"output total": {response.Usage.OutputTotal.Value, test.total},
				"reasoning":    {response.Usage.OutputReasoning.Value, test.reasoning},
				"other output": {response.Usage.OutputOther.Value, test.other},
				"input total":  {response.Usage.InputTotal.Value, 12},
			} {
				if count.got == nil || *count.got != count.want {
					t.Fatalf("%s = %v, want %d (usage=%+v)", name, count.got, count.want, response.Usage)
				}
			}
		})
	}
}

// Every client format receives the whole answer and a normal terminal event.
// The thinking text is unsigned so that it may cross formats.
func TestAnthropicShortThinkingAnswerCompletesForEveryClientFormat(t *testing.T) {
	body := thinkingStream(`{"type":"thinking_delta","thinking":"short"}`,
		`{"input_tokens":12,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":7}`,
		`{"output_tokens":30,"output_tokens_details":{"thinking_tokens":26}}`,
	)
	terminals := map[llmprotocol.WireFormat]string{
		llmprotocol.AnthropicMessagesV1: "event: message_stop\n",
		llmprotocol.OpenAIChatV1:        "data: [DONE]",
		llmprotocol.OpenAIResponsesV1:   "event: response.completed\n",
	}
	for target, terminal := range terminals {
		t.Run(string(target), func(t *testing.T) {
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
			if err != nil {
				t.Fatal(err)
			}
			frames, _, _, err := stream.Push(body)
			if err != nil {
				t.Fatalf("stream aborted: %v", err)
			}
			final, _, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			output := string(bytes.Join(append(frames, final...), nil))
			if !strings.Contains(output, terminal) || strings.Contains(output, "event: error") {
				t.Fatalf("stream did not complete normally:\n%s", output)
			}
			if target == llmprotocol.AnthropicMessagesV1 &&
				!strings.Contains(output, `"output_tokens":30,"output_tokens_details":{"thinking_tokens":26}`) {
				t.Fatalf("message_delta lost the final output usage:\n%s", output)
			}
		})
	}
}

func TestAnthropicStreamOutputTotalDecreaseStaysStrict(t *testing.T) {
	_, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.AnthropicMessagesV1,
		thinkingShortAnswerStream(
			`{"input_tokens":12,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":7}`,
			`{"output_tokens":5,"output_tokens_details":{"thinking_tokens":3}}`,
		),
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "usage_decreased")
}
