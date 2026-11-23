package protocolcodec

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// decodeAnthropicEvents pushes Anthropic stream events through the Messages
// decoder and finalizes it, returning the events and the first error.
func decodeAnthropicEvents(t *testing.T, events ...map[string]any) ([]llmprotocol.Event, error) {
	t.Helper()
	decoder := AnthropicMessagesCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
		llmprotocol.DefaultPolicy(),
	)
	var out []llmprotocol.Event
	for _, event := range events {
		frame, err := encodeSSE(event["type"].(string), event)
		if err != nil {
			t.Fatal(err)
		}
		decoded, _, err := decoder.Push(frame)
		out = append(out, decoded...)
		if err != nil {
			return out, err
		}
	}
	decoded, _, err := decoder.Finalize(nil)
	return append(out, decoded...), err
}

func emptyAnswerStart() map[string]any {
	return map[string]any{"type": "message_start", "message": map[string]any{
		"id": "msg_1", "type": "message", "role": "assistant", "model": "provider-model",
		"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": 506, "output_tokens": 1},
	}}
}

func emptyAnswerDelta(stop any) map[string]any {
	return map[string]any{
		"type": "message_delta", "delta": map[string]any{"stop_reason": stop, "stop_sequence": nil},
		"usage": map[string]any{"input_tokens": 506, "output_tokens": 2},
	}
}

// A model that ends its turn with nothing to say streams message_start,
// message_delta with end_turn and message_stop: a completed empty answer with
// its usage. A stream that never says how it ended, or claims a tool call it
// never made, is still an upstream defect.
func TestAnthropicStreamCompletesAnEmptyEndTurnAnswer(t *testing.T) {
	stop := map[string]any{"type": "message_stop"}
	events, err := decodeAnthropicEvents(t, emptyAnswerStart(), emptyAnswerDelta("end_turn"), stop)
	if err != nil {
		t.Fatalf("an empty end_turn answer was refused: %v", err)
	}
	last := events[len(events)-1]
	if last.Type != llmprotocol.EventResponseCompleted || last.StopReason != llmprotocol.StopEndTurn ||
		last.Usage == nil || last.Usage.OutputTotal.Value == nil || *last.Usage.OutputTotal.Value != 2 {
		t.Fatalf("the empty answer did not complete with its usage: %+v", last)
	}

	for name, events := range map[string][]map[string]any{
		"no message_stop":       {emptyAnswerStart(), emptyAnswerDelta("end_turn")},
		"no stop reason":        {emptyAnswerStart(), emptyAnswerDelta(nil), stop},
		"tool use with no call": {emptyAnswerStart(), emptyAnswerDelta("tool_use"), stop},
	} {
		decoded, err := decodeAnthropicEvents(t, events...)
		if err == nil && len(decoded) > 0 && decoded[len(decoded)-1].Type == llmprotocol.EventResponseFailed {
			err = decoded[len(decoded)-1].Error
		}
		if protocolErr := (*llmprotocol.ProtocolError)(nil); !errors.As(err, &protocolErr) {
			t.Errorf("%s: a broken empty stream was not refused: %v", name, err)
		}
	}
}
