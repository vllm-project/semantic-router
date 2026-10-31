package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicPartialInputUsageUpdatesSettle(t *testing.T) {
	deltas := []struct {
		name  string
		delta string
	}{
		{"output_only_control", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}`},
		{"full_input_control", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":50,"cache_read_input_tokens":20,"cache_creation_input_tokens":30,"output_tokens":15}}`},
		{"ordinary_input_only", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":50,"output_tokens":15}}`},
		{"cache_read_only", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"cache_read_input_tokens":20,"output_tokens":15}}`},
		{"cache_write_only", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"cache_creation_input_tokens":30,"output_tokens":15}}`},
	}
	for _, test := range deltas {
		t.Run(test.name, func(t *testing.T) {
			payload := formatPartialUsageStream(test.delta)
			stream, err := NewBuiltinEngine().NewStream(
				llmprotocol.AnthropicMessagesV1,
				llmprotocol.AnthropicMessagesV1,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
			)
			if err != nil {
				t.Fatal(err)
			}
			_, events, _, err := stream.Push(payload)
			if err != nil {
				t.Fatalf("partial usage update aborted the stream: %v", err)
			}
			_, finalEvents, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			text, terminal := "", 0
			var usage *llmprotocol.Usage
			for _, event := range append(events, finalEvents...) {
				switch event.Type {
				case llmprotocol.EventOutputTextDelta:
					text += event.Delta
				case llmprotocol.EventResponseCompleted:
					terminal++
					usage = event.Usage
				}
			}
			if text != "answer" || terminal != 1 || usage == nil {
				t.Fatalf("stream incomplete: text=%q terminals=%d usage=%+v", text, terminal, usage)
			}
			if usage.InputUncached.Value == nil || *usage.InputUncached.Value != 50 {
				t.Fatalf("input evidence lost: %+v", usage.InputUncached)
			}
			if usage.InputCacheRead.Value == nil || *usage.InputCacheRead.Value != 20 {
				t.Fatalf("cache-read evidence lost: %+v", usage.InputCacheRead)
			}
			if usage.InputCacheWrite.Value == nil || *usage.InputCacheWrite.Value != 30 {
				t.Fatalf("cache-write evidence lost: %+v", usage.InputCacheWrite)
			}
			if usage.OutputTotal.Value == nil || *usage.OutputTotal.Value != 15 {
				t.Fatalf("output evidence wrong: %+v", usage.OutputTotal)
			}
			if usage.Total.Value == nil || *usage.Total.Value != 115 {
				t.Fatalf("total not settled from the merged evidence: %+v", usage.Total)
			}
		})
	}
}

func TestAnthropicPartialUsageKeepsGenuineDecreaseStrict(t *testing.T) {
	payload := formatPartialUsageStream(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":40,"output_tokens":15}}`)
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := stream.Push(payload); err == nil {
		t.Fatal("an explicitly decreased input counter was accepted")
	} else {
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "usage_decreased")
	}
}

func formatPartialUsageStream(delta string) []byte {
	start := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":50,\"cache_read_input_tokens\":20,\"cache_creation_input_tokens\":30}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
	return []byte(start + "event: message_delta\ndata: " + delta + "\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
}
