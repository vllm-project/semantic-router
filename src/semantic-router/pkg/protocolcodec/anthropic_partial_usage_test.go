package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
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

// A Messages client reads input usage from the terminal message_delta, so the
// retained evidence of a partial update must reach it, and so must a Chat or
// Responses backend's input total that arrives without a cache breakdown.
func TestStreamInputUsageReachesMessagesClients(t *testing.T) {
	partial := func(usage string) []byte {
		return formatPartialUsageStream(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":` + usage + `}`)
	}
	tests := []struct {
		name   string
		source llmprotocol.WireFormat
		body   []byte
		want   map[string]int64
	}{
		{
			name: "chat total without cache breakdown", source: llmprotocol.OpenAIChatV1,
			body: []byte("data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"provider-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":3,\"total_tokens\":7}}\n\n" +
				"data: [DONE]\n\n"),
			want: map[string]int64{"input_tokens": 4, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 3},
		},
		{
			name: "responses total without cache breakdown", source: llmprotocol.OpenAIResponsesV1,
			body: []byte("event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"status\":\"in_progress\",\"model\":\"provider-model\",\"output\":[]}}\n\n" +
				"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}\n\n" +
				"event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":2,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}\n\n" +
				"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":3,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"answer\"}\n\n" +
				"event: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"sequence_number\":4,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"text\":\"answer\"}\n\n" +
				"event: response.content_part.done\ndata: {\"type\":\"response.content_part.done\",\"sequence_number\":5,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"answer\",\"annotations\":[]}}\n\n" +
				"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":6,\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\",\"annotations\":[]}]}}\n\n" +
				"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":7,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"status\":\"completed\",\"model\":\"provider-model\",\"output\":[{\"id\":\"msg_1\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":2,\"output_tokens\":4,\"total_tokens\":6}}}\n\n"),
			want: map[string]int64{"input_tokens": 2, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 4},
		},
		{
			name: "ordinary input only", source: llmprotocol.AnthropicMessagesV1,
			body: partial(`{"input_tokens":50,"output_tokens":15}`),
			want: map[string]int64{"input_tokens": 50, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 30, "output_tokens": 15},
		},
		{
			name: "cache read only", source: llmprotocol.AnthropicMessagesV1,
			body: partial(`{"cache_read_input_tokens":20,"output_tokens":15}`),
			want: map[string]int64{"input_tokens": 50, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 30, "output_tokens": 15},
		},
		{
			name: "cache write grows", source: llmprotocol.AnthropicMessagesV1,
			body: partial(`{"cache_creation_input_tokens":35,"output_tokens":15}`),
			want: map[string]int64{"input_tokens": 50, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 35, "output_tokens": 15},
		},
		{
			name: "output only", source: llmprotocol.AnthropicMessagesV1,
			body: partial(`{"output_tokens":15}`),
			want: map[string]int64{"input_tokens": 50, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 30, "output_tokens": 15},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			frames := streamClientFrames(t, test.source, llmprotocol.AnthropicMessagesV1, test.body)
			usage := clientFrameField(t, frames, "message_delta", "usage")
			for field, want := range test.want {
				got, ok := usage[field].(float64)
				if !ok || int64(got) != want {
					t.Fatalf("message_delta usage.%s = %v, want %d (usage=%v)", field, usage[field], want, usage)
				}
			}
		})
	}
}

// A delta that reports only the cache-write TTL split keeps that split and
// settles the aggregate it implies.
func TestAnthropicPartialUsageKeepsCacheWriteTiers(t *testing.T) {
	body := formatPartialUsageStream(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"cache_creation":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":20},"output_tokens":15}}`)
	response, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.AnthropicMessagesV1, body,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatalf("tier-only usage update aborted the stream: %v", err)
	}
	usage := response.Usage
	for name, count := range map[string]struct {
		got  *int64
		want int64
	}{
		"cache write": {usage.InputCacheWrite.Value, 30}, "5m tier": {usage.InputCacheWrite5m.Value, 10},
		"1h tier": {usage.InputCacheWrite1h.Value, 20}, "input total": {usage.InputTotal.Value, 100},
	} {
		if count.got == nil || *count.got != count.want {
			t.Fatalf("%s = %v, want %d (usage=%+v)", name, count.got, count.want, usage)
		}
	}
}

// A Responses client receives the merged input total and its cache split in
// response.completed.
func TestAnthropicPartialInputUsageReachesResponsesClients(t *testing.T) {
	body := formatPartialUsageStream(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"cache_read_input_tokens":25,"output_tokens":15}}`)
	frames := streamClientFrames(t, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, body)
	response := clientFrameField(t, frames, "response.completed", "response")
	usage, _ := response["usage"].(map[string]any)
	details, _ := usage["input_tokens_details"].(map[string]any)
	got := map[string]any{
		"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"], "total_tokens": usage["total_tokens"],
		"cached_tokens": details["cached_tokens"], "cache_write_tokens": details["cache_write_tokens"],
	}
	want := map[string]float64{"input_tokens": 105, "output_tokens": 15, "total_tokens": 120, "cached_tokens": 25, "cache_write_tokens": 30}
	for field, value := range want {
		if got[field] != value {
			t.Fatalf("response.completed usage %s = %v, want %v (usage=%v)", field, got[field], value, usage)
		}
	}
}

func streamClientFrames(t *testing.T, source, target llmprotocol.WireFormat, body []byte) [][]byte {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(source, target,
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
	return append(frames, final...)
}

func clientFrameField(t *testing.T, frames [][]byte, event, field string) map[string]any {
	t.Helper()
	for _, frame := range frames {
		if !bytes.HasPrefix(frame, []byte("event: "+event+"\n")) {
			continue
		}
		_, data, _ := bytes.Cut(frame, []byte("\ndata: "))
		var payload map[string]any
		if err := json.Unmarshal(bytes.TrimSpace(data), &payload); err != nil {
			t.Fatalf("%s frame is not JSON: %s", event, frame)
		}
		value, ok := payload[field].(map[string]any)
		if !ok {
			t.Fatalf("%s frame has no %s object: %s", event, field, frame)
		}
		return value
	}
	t.Fatalf("no %s frame in %q", event, frames)
	return nil
}

func formatPartialUsageStream(delta string) []byte {
	start := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":50,\"cache_read_input_tokens\":20,\"cache_creation_input_tokens\":30}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
	return []byte(start + "event: message_delta\ndata: " + delta + "\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
}
