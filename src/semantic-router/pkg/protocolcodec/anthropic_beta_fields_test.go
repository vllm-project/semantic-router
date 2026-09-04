package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The context editing echo an Anthropic beta response carries once the
// context-management-2025-06-27 header is in play.
func anthropicBetaContextEdits() map[string]any {
	return map[string]any{"applied_edits": []any{map[string]any{
		"type": "clear_thinking_20251015", "cleared_thinking_turns": 2, "cleared_input_tokens": 1500,
	}}}
}

func anthropicBetaMessage(extra map[string]any) map[string]any {
	message := map[string]any{
		"id": "msg_1", "type": "message", "role": "assistant", "model": "provider-model",
		"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": 3, "output_tokens": 0},
	}
	for key, value := range extra {
		message[key] = value
	}
	return message
}

func anthropicBetaDecoder(t *testing.T) llmprotocol.StreamDecoder {
	t.Helper()
	return AnthropicMessagesCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
		llmprotocol.DefaultPolicy(),
	)
}

func anthropicBetaFrame(t *testing.T, event string, data map[string]any) []byte {
	t.Helper()
	frame, err := encodeSSE(event, data)
	if err != nil {
		t.Fatal(err)
	}
	return frame
}

func anthropicBetaBody(t *testing.T, extra map[string]any) []byte {
	t.Helper()
	extra["content"] = []any{map[string]any{"type": "text", "text": "hi"}}
	body, err := json.Marshal(anthropicBetaMessage(extra))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// Returns only the events and diagnostics of this one push. Diagnostics
// accumulate over a stream, so a per-push slice is the only honest count.
func anthropicBetaPush(
	t *testing.T, decoder llmprotocol.StreamDecoder, event string, data map[string]any,
) ([]llmprotocol.Event, llmprotocol.Diagnostics) {
	t.Helper()
	events, diagnostics, err := decoder.Push(anthropicBetaFrame(t, event, data))
	if err != nil {
		t.Fatalf("%s carrying beta fields was rejected: %v", event, err)
	}
	return events, diagnostics
}

func anthropicBetaDroppedCount(diagnostics llmprotocol.Diagnostics, field string) int {
	count := 0
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == field && diagnostic.Action == llmprotocol.DiagnosticDropped {
			count++
		}
	}
	return count
}

func anthropicBetaTokens(t *testing.T, count llmprotocol.TokenCount) int64 {
	t.Helper()
	if count.Value == nil {
		t.Fatal("token count was not recorded")
	}
	return *count.Value
}

// These four positions are the complete set of Anthropic beta response fields
// Claude Code elicits with default settings (#3417). Each is asserted alone so a
// regression names the exact position that broke.
func TestAnthropicBetaResponseFieldsDecode(t *testing.T) {
	t.Run("thinking delta estimated_tokens", func(t *testing.T) {
		decoder := anthropicBetaDecoder(t)
		anthropicBetaPush(t, decoder, "message_start", map[string]any{
			"type": "message_start", "message": anthropicBetaMessage(nil),
		})
		anthropicBetaPush(t, decoder, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "thinking", "thinking": ""},
		})
		events, _ := anthropicBetaPush(t, decoder, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "thinking_delta", "thinking": "inspect", "estimated_tokens": 42},
		})
		reasoning := 0
		for _, event := range events {
			if event.Type == llmprotocol.EventReasoningDelta && event.Delta == "inspect" {
				reasoning++
			}
		}
		if reasoning != 1 {
			t.Fatalf("thinking delta did not survive estimated_tokens: %+v", events)
		}
	})

	t.Run("message_start context_management", func(t *testing.T) {
		decoder := anthropicBetaDecoder(t)
		events, diagnostics := anthropicBetaPush(t, decoder, "message_start", map[string]any{
			"type": "message_start",
			"message": anthropicBetaMessage(map[string]any{
				"context_management": anthropicBetaContextEdits(),
			}),
		})
		// ResponseID, not Model: the decoder substitutes the public model on the
		// way out, so the ID is what proves the field-bearing message decoded.
		started := 0
		for _, event := range events {
			if event.Type == llmprotocol.EventResponseStarted && event.ResponseID == "msg_1" {
				started++
			}
		}
		if started != 1 {
			t.Fatalf("message_start did not open the response: %+v", events)
		}
		// Deliberately silent here: Anthropic documents the echo on message_delta
		// and the buffered body, so message_start accepts it defensively without
		// reporting a drop it cannot confirm the API made.
		if len(diagnostics) != 0 {
			t.Fatalf("message_start reported a drop it should stay silent about: %+v", diagnostics)
		}
	})

	// The only position that proves the field exists on anthropicEventWire. A
	// message_start test passes without it, because message_start decodes its
	// payload through anthropicResponseWire instead.
	t.Run("message_delta context_management", func(t *testing.T) {
		decoder := anthropicBetaDecoder(t)
		anthropicBetaPush(t, decoder, "message_start", map[string]any{
			"type": "message_start", "message": anthropicBetaMessage(nil),
		})
		events, diagnostics := anthropicBetaPush(t, decoder, "message_delta", map[string]any{
			"type":               "message_delta",
			"context_management": anthropicBetaContextEdits(),
			"delta":              map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
			"usage":              map[string]any{"input_tokens": 3, "output_tokens": 2},
		})
		if got := anthropicBetaDroppedCount(diagnostics, "stream.context_management"); got != 1 {
			t.Fatalf("message_delta echo produced %d dropped diagnostics, want 1: %+v", got, diagnostics)
		}
		var usage *llmprotocol.Usage
		for _, event := range events {
			if event.Type == llmprotocol.EventUsageUpdated {
				usage = event.Usage
			}
		}
		if usage == nil {
			t.Fatalf("message_delta usage was lost alongside the echo: %+v", events)
		}
		if got := anthropicBetaTokens(t, usage.OutputTotal); got != 2 {
			t.Fatalf("output tokens = %d, want 2", got)
		}
		if got := anthropicBetaTokens(t, usage.InputTotal); got != 3 {
			t.Fatalf("input tokens = %d, want 3", got)
		}
	})

	t.Run("buffered body context_management", func(t *testing.T) {
		body := anthropicBetaBody(t, map[string]any{"context_management": anthropicBetaContextEdits()})
		response, _, diagnostics, err := NewBuiltinEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
		if err != nil {
			t.Fatalf("buffered body carrying the echo was rejected: %v", err)
		}
		if got := anthropicBetaDroppedCount(diagnostics, "context_management"); got != 1 {
			t.Fatalf("buffered echo produced %d dropped diagnostics, want 1: %+v", got, diagnostics)
		}
		if len(response.Output) != 1 || len(response.Output[0].Content) != 1 ||
			response.Output[0].Content[0].Text != "hi" {
			t.Fatalf("buffered content was lost alongside the echo: %+v", response.Output)
		}
	})
}

// Guard for D1: naming beta fields must not turn strict decode into a silent
// JSON sink. If this ever passes, the fix went too far.
func TestAnthropicUnknownResponseFieldStillRejected(t *testing.T) {
	assert := func(t *testing.T, err error) {
		t.Helper()
		var protocolError *llmprotocol.ProtocolError
		if !errors.As(err, &protocolError) || protocolError.Code != "invalid_upstream_json" {
			t.Fatalf("unknown field was accepted: %v", err)
		}
		if protocolError.Category != llmprotocol.ErrorUpstreamUnavailable {
			t.Fatalf("category = %s, want upstream_unavailable", protocolError.Category)
		}
	}

	t.Run("message_start", func(t *testing.T) {
		decoder := anthropicBetaDecoder(t)
		_, _, err := decoder.Push(anthropicBetaFrame(t, "message_start", map[string]any{
			"type":    "message_start",
			"message": anthropicBetaMessage(map[string]any{"totally_made_up_field": 1}),
		}))
		assert(t, err)
	})

	// Separate from message_start because the two positions decode through
	// different wire structs; only this one covers anthropicEventWire.
	t.Run("message_delta", func(t *testing.T) {
		decoder := anthropicBetaDecoder(t)
		anthropicBetaPush(t, decoder, "message_start", map[string]any{
			"type": "message_start", "message": anthropicBetaMessage(nil),
		})
		_, _, err := decoder.Push(anthropicBetaFrame(t, "message_delta", map[string]any{
			"type": "message_delta", "totally_made_up_field": 1,
			"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
			"usage": map[string]any{"input_tokens": 3, "output_tokens": 2},
		}))
		assert(t, err)
	})

	t.Run("buffered", func(t *testing.T) {
		body := anthropicBetaBody(t, map[string]any{"totally_made_up_field": 1})
		_, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
		assert(t, err)
	})
}

// D2: this arrives on every thinking delta. A diagnostic per delta would
// exhaust the 64-entry budget and write a metric per frame. 200 deltas is
// well past that budget, so a leak fails loudly.
func TestAnthropicEstimatedTokensIsSilentAndNotReEmitted(t *testing.T) {
	const deltas = 200

	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}

	var emitted [][]byte
	push := func(event string, data map[string]any) []llmprotocol.Event {
		t.Helper()
		frames, events, diagnostics, err := stream.Push(anthropicBetaFrame(t, event, data))
		if err != nil {
			t.Fatalf("%s was rejected: %v", event, err)
		}
		// Counted per push, not cumulatively: a running total would still look
		// clean if a single push emitted all two hundred.
		if len(diagnostics) != 0 {
			t.Fatalf("%s emitted %d diagnostics, want none: %+v", event, len(diagnostics), diagnostics)
		}
		emitted = append(emitted, frames...)
		return events
	}

	push("message_start", map[string]any{"type": "message_start", "message": anthropicBetaMessage(nil)})
	push("content_block_start", map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "thinking", "thinking": ""},
	})

	reasoning := 0
	for i := 0; i < deltas; i++ {
		events := push("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "thinking_delta", "thinking": "x", "estimated_tokens": i},
		})
		for _, event := range events {
			if event.Type == llmprotocol.EventReasoningDelta {
				reasoning++
			}
		}
	}
	if reasoning != deltas {
		t.Fatalf("decoded %d reasoning deltas, want %d", reasoning, deltas)
	}
	if bytes.Contains(bytes.Join(emitted, nil), []byte("estimated_tokens")) {
		t.Fatal("estimated_tokens was re-emitted on the encoded stream")
	}
}
