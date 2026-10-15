package protocolcodec

import (
	"bytes"
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicPreservesCacheWriteTiersBuffered(t *testing.T) {
	engine := NewBuiltinEngine()
	wire := []byte(`{
		"id": "msg_tiers_1",
		"type": "message",
		"role": "assistant",
		"model": "claude-3-haiku",
		"content": [{"type": "text", "text": "hello"}],
		"stop_reason": "end_turn",
		"stop_sequence": null,
		"usage": {
			"input_tokens": 100,
			"output_tokens": 20,
			"cache_creation_input_tokens": 30,
			"cache_read_input_tokens": 50,
			"cache_creation": {
				"ephemeral_5m_input_tokens": 20,
				"ephemeral_1h_input_tokens": 10
			}
		}
	}`)

	resp, _, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, wire)
	if err != nil {
		t.Fatalf("DecodeResponse error: %v", err)
	}
	for _, d := range diagnostics {
		if d.Field == "usage.cache_creation" {
			t.Fatalf("expected no usage.cache_creation diagnostic for matching tiers, got: %v", d)
		}
	}
	if resp.Usage.InputCacheWrite.Value == nil || *resp.Usage.InputCacheWrite.Value != 30 {
		t.Fatalf("expected aggregate cache write 30, got: %v", resp.Usage.InputCacheWrite)
	}
	if resp.Usage.InputCacheWrite5m.Value == nil || *resp.Usage.InputCacheWrite5m.Value != 20 {
		t.Fatalf("expected 5m cache write 20, got: %v", resp.Usage.InputCacheWrite5m)
	}
	if resp.Usage.InputCacheWrite1h.Value == nil || *resp.Usage.InputCacheWrite1h.Value != 10 {
		t.Fatalf("expected 1h cache write 10, got: %v", resp.Usage.InputCacheWrite1h)
	}

	result, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, resp, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("EncodeResponse error: %v", err)
	}
	if !bytes.Contains(result.Body, []byte(`"ephemeral_5m_input_tokens":20`)) {
		t.Fatalf("re-encoded wire lost 5m tier: %s", result.Body)
	}
	if !bytes.Contains(result.Body, []byte(`"ephemeral_1h_input_tokens":10`)) {
		t.Fatalf("re-encoded wire lost 1h tier: %s", result.Body)
	}
}

func TestAnthropicMismatchedCacheWriteTiersEmitsDiagnosticAndFallsBack(t *testing.T) {
	engine := NewBuiltinEngine()
	// Tiers sum to 10 (5 + 5), but aggregate is 30
	wire := []byte(`{
		"id": "msg_tiers_mismatch",
		"type": "message",
		"role": "assistant",
		"model": "claude-3-haiku",
		"content": [{"type": "text", "text": "hello"}],
		"stop_reason": "end_turn",
		"stop_sequence": null,
		"usage": {
			"input_tokens": 100,
			"output_tokens": 20,
			"cache_creation_input_tokens": 30,
			"cache_read_input_tokens": 50,
			"cache_creation": {
				"ephemeral_5m_input_tokens": 5,
				"ephemeral_1h_input_tokens": 5
			}
		}
	}`)

	resp, _, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, wire)
	if err != nil {
		t.Fatalf("DecodeResponse error: %v", err)
	}
	assertDiagnosticFields(t, diagnostics, "usage.cache_creation")

	// Tiers should be unknown because of mismatch
	if resp.Usage.InputCacheWrite5m.Value != nil || resp.Usage.InputCacheWrite1h.Value != nil {
		t.Fatalf("expected tiers to be unknown on mismatch, got 5m=%v 1h=%v", resp.Usage.InputCacheWrite5m, resp.Usage.InputCacheWrite1h)
	}
	// Aggregate write tokens must still be preserved
	if resp.Usage.InputCacheWrite.Value == nil || *resp.Usage.InputCacheWrite.Value != 30 {
		t.Fatalf("expected aggregate cache write 30, got: %v", resp.Usage.InputCacheWrite)
	}

	// Encoding back should fall back to 5m for aggregate write tokens and emit a diagnostic
	result, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, resp, llmprotocol.Envelope{Format: llmprotocol.AnthropicMessagesV1})
	if err != nil {
		t.Fatalf("EncodeResponse error: %v", err)
	}
	assertDiagnosticFields(t, result.Diagnostics, "usage.cache_creation")
	if !bytes.Contains(result.Body, []byte(`"ephemeral_5m_input_tokens":30`)) {
		t.Fatalf("fallback wire expected ephemeral_5m_input_tokens=30: %s", result.Body)
	}
}

func TestAnthropicPreservesCacheWriteTiersStreaming(t *testing.T) {
	stream := []byte("event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_stream_1","type":"message","role":"assistant","model":"claude-3","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":50,"cache_read_input_tokens":20,"cache_creation_input_tokens":30,"cache_creation":{"ephemeral_5m_input_tokens":18,"ephemeral_1h_input_tokens":12}}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello world"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n")

	resp, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.AnthropicMessagesV1, stream, llmprotocol.StreamContext{
		Context:     context.Background(),
		PublicModel: "claude-3",
	})
	if err != nil {
		t.Fatalf("DecodeResponseStream error: %v", err)
	}
	if resp.Usage.InputCacheWrite.Value == nil || *resp.Usage.InputCacheWrite.Value != 30 {
		t.Fatalf("expected aggregate cache write 30, got: %v", resp.Usage.InputCacheWrite)
	}
	if resp.Usage.InputCacheWrite5m.Value == nil || *resp.Usage.InputCacheWrite5m.Value != 18 {
		t.Fatalf("expected 5m cache write 18, got: %v", resp.Usage.InputCacheWrite5m)
	}
	if resp.Usage.InputCacheWrite1h.Value == nil || *resp.Usage.InputCacheWrite1h.Value != 12 {
		t.Fatalf("expected 1h cache write 12, got: %v", resp.Usage.InputCacheWrite1h)
	}
}

func TestAnthropicDerivesAggregateFromCompleteTiersWhenAbsent(t *testing.T) {
	engine := NewBuiltinEngine()
	wire := []byte(`{
		"id": "msg_tiers_derived",
		"type": "message",
		"role": "assistant",
		"model": "claude-3-haiku",
		"content": [{"type": "text", "text": "hello"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 10,
			"output_tokens": 5,
			"cache_read_input_tokens": 10,
			"cache_creation": {
				"ephemeral_5m_input_tokens": 30,
				"ephemeral_1h_input_tokens": 50
			}
		}
	}`)

	resp, _, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, wire)
	if err != nil {
		t.Fatalf("DecodeResponse error: %v", err)
	}
	for _, d := range diagnostics {
		if d.Field == "usage.cache_creation" {
			t.Fatalf("unexpected diagnostic for complete tiers: %v", d)
		}
	}
	if resp.Usage.InputCacheWrite.Value == nil || *resp.Usage.InputCacheWrite.Value != 80 {
		t.Fatalf("expected derived aggregate cache write 80, got: %v", resp.Usage.InputCacheWrite)
	}
	if resp.Usage.InputCacheWrite5m.Value == nil || *resp.Usage.InputCacheWrite5m.Value != 30 {
		t.Fatalf("expected 5m cache write 30, got: %v", resp.Usage.InputCacheWrite5m)
	}
	if resp.Usage.InputCacheWrite1h.Value == nil || *resp.Usage.InputCacheWrite1h.Value != 50 {
		t.Fatalf("expected 1h cache write 50, got: %v", resp.Usage.InputCacheWrite1h)
	}
	if resp.Usage.InputTotal.Value == nil || *resp.Usage.InputTotal.Value != 100 {
		t.Fatalf("expected input total 100 (10 uncached + 10 read + 80 write), got: %v", resp.Usage.InputTotal)
	}
}

func TestAnthropicPreservesPartialTierEvidence(t *testing.T) {
	engine := NewBuiltinEngine()
	// aggregate=30, only 5m=20 reported
	wire := []byte(`{
		"id": "msg_tiers_partial",
		"type": "message",
		"role": "assistant",
		"model": "claude-3-haiku",
		"content": [{"type": "text", "text": "hello"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 50,
			"output_tokens": 10,
			"cache_creation_input_tokens": 30,
			"cache_creation": {
				"ephemeral_5m_input_tokens": 20
			}
		}
	}`)

	resp, _, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, wire)
	if err != nil {
		t.Fatalf("DecodeResponse error: %v", err)
	}
	for _, d := range diagnostics {
		if d.Field == "usage.cache_creation" {
			t.Fatalf("unexpected diagnostic for valid partial tier: %v", d)
		}
	}
	if resp.Usage.InputCacheWrite.Value == nil || *resp.Usage.InputCacheWrite.Value != 30 {
		t.Fatalf("expected aggregate cache write 30, got: %v", resp.Usage.InputCacheWrite)
	}
	if resp.Usage.InputCacheWrite5m.Value == nil || *resp.Usage.InputCacheWrite5m.Value != 20 {
		t.Fatalf("expected 5m cache write 20, got: %v", resp.Usage.InputCacheWrite5m)
	}
	if resp.Usage.InputCacheWrite1h.Value != nil {
		t.Fatalf("expected 1h cache write to remain unknown/nil, got: %v", resp.Usage.InputCacheWrite1h)
	}
}

func TestTargetEncodingEmitsOmissionForChatAndResponses(t *testing.T) {
	engine := NewBuiltinEngine()
	resp := llmprotocol.Response{
		Generation: 1,
		ID:         "resp_1",
		Model:      "test-model",
		Usage: llmprotocol.Usage{
			State:             llmprotocol.UsageAvailable,
			InputTotal:        authoritative(100),
			InputCacheWrite:   authoritative(30),
			InputCacheWrite5m: authoritative(20),
			InputCacheWrite1h: authoritative(10),
			OutputTotal:       authoritative(20),
		},
		Output: []llmprotocol.OutputItem{
			{
				ID:   "out_1",
				Role: llmprotocol.RoleAssistant,
				Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentText, Text: "hi"},
				},
			},
		},
	}

	chatResult, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, resp, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("EncodeResponse to Chat error: %v", err)
	}
	assertDiagnosticFields(t, chatResult.Diagnostics, "usage.cache_creation")

	respResult, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, resp, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("EncodeResponse to Responses error: %v", err)
	}
	assertDiagnosticFields(t, respResult.Diagnostics, "usage.cache_creation")
}

func TestAnthropicPartialTier1hRoundTripPreservesKnownTier(t *testing.T) {
	engine := NewBuiltinEngine()
	// aggregate=30, only 1h=20 reported (5m is unknown/nil)
	wire := []byte(`{
		"id": "msg_tiers_partial_1h",
		"type": "message",
		"role": "assistant",
		"model": "claude-3-haiku",
		"content": [{"type": "text", "text": "hello"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 50,
			"output_tokens": 10,
			"cache_creation_input_tokens": 30,
			"cache_creation": {
				"ephemeral_1h_input_tokens": 20
			}
		}
	}`)

	resp, _, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, wire)
	if err != nil {
		t.Fatalf("DecodeResponse error: %v", err)
	}
	for _, d := range diagnostics {
		if d.Field == "usage.cache_creation" {
			t.Fatalf("unexpected diagnostic for valid partial 1h tier: %v", d)
		}
	}
	if resp.Usage.InputCacheWrite.Value == nil || *resp.Usage.InputCacheWrite.Value != 30 {
		t.Fatalf("expected aggregate cache write 30, got: %v", resp.Usage.InputCacheWrite)
	}
	if resp.Usage.InputCacheWrite1h.Value == nil || *resp.Usage.InputCacheWrite1h.Value != 20 {
		t.Fatalf("expected 1h cache write 20, got: %v", resp.Usage.InputCacheWrite1h)
	}
	if resp.Usage.InputCacheWrite5m.Value != nil {
		t.Fatalf("expected 5m cache write to remain unknown/nil, got: %v", resp.Usage.InputCacheWrite5m)
	}

	result, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, resp, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("EncodeResponse error: %v", err)
	}
	if !bytes.Contains(result.Body, []byte(`"ephemeral_1h_input_tokens":20`)) {
		t.Fatalf("re-encoded wire lost 1h tier: %s", result.Body)
	}
	if bytes.Contains(result.Body, []byte(`ephemeral_5m_input_tokens`)) {
		t.Fatalf("re-encoded wire should omit unknown 5m tier: %s", result.Body)
	}

	// Re-decoding must preserve 1h tier and not treat it as an inconsistent split
	roundTripResp, _, rtDiagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, result.Body)
	if err != nil {
		t.Fatalf("round-trip DecodeResponse error: %v", err)
	}
	for _, d := range rtDiagnostics {
		if d.Field == "usage.cache_creation" {
			t.Fatalf("unexpected diagnostic on round trip: %v", d)
		}
	}
	if roundTripResp.Usage.InputCacheWrite1h.Value == nil || *roundTripResp.Usage.InputCacheWrite1h.Value != 20 {
		t.Fatalf("expected round trip 1h cache write 20, got: %v", roundTripResp.Usage.InputCacheWrite1h)
	}
	if roundTripResp.Usage.InputCacheWrite5m.Value != nil {
		t.Fatalf("expected round trip 5m cache write to remain nil, got: %v", roundTripResp.Usage.InputCacheWrite5m)
	}
}

func TestAnthropicEncodeResponseStreamRoundTripPreservesTiers(t *testing.T) {
	engine := NewBuiltinEngine()
	resp := llmprotocol.Response{
		Generation: 1,
		ID:         "resp_stream_tiers",
		Model:      "claude-3-haiku",
		StopReason: llmprotocol.StopEndTurn,
		Usage: llmprotocol.Usage{
			State:             llmprotocol.UsageAvailable,
			InputTotal:        authoritative(100),
			InputCacheRead:    authoritative(20),
			InputCacheWrite:   authoritative(30),
			InputCacheWrite5m: authoritative(20),
			InputCacheWrite1h: authoritative(10),
			InputUncached:     authoritative(50),
			OutputTotal:       authoritative(15),
			OutputOther:       authoritative(15),
			Total:             authoritative(115),
		},
		Output: []llmprotocol.OutputItem{
			{
				ID:   "out_1",
				Role: llmprotocol.RoleAssistant,
				Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentText, Text: "hello world"},
				},
			},
		},
	}

	streamBytes, _, err := engine.EncodeResponseStream(llmprotocol.AnthropicMessagesV1, resp, llmprotocol.StreamContext{
		PublicModel: resp.Model,
	})
	if err != nil {
		t.Fatalf("EncodeResponseStream error: %v", err)
	}

	if !bytes.Contains(streamBytes, []byte(`"ephemeral_5m_input_tokens":20`)) {
		t.Fatalf("stream output missing 5m tier: %s", streamBytes)
	}
	if !bytes.Contains(streamBytes, []byte(`"ephemeral_1h_input_tokens":10`)) {
		t.Fatalf("stream output missing 1h tier: %s", streamBytes)
	}

	decoded, _, err := engine.DecodeResponseStream(llmprotocol.AnthropicMessagesV1, streamBytes, llmprotocol.StreamContext{
		Context:     context.Background(),
		PublicModel: resp.Model,
	})
	if err != nil {
		t.Fatalf("DecodeResponseStream error: %v", err)
	}

	if decoded.Usage.InputCacheWrite.Value == nil || *decoded.Usage.InputCacheWrite.Value != 30 {
		t.Fatalf("expected aggregate cache write 30, got: %v", decoded.Usage.InputCacheWrite)
	}
	if decoded.Usage.InputCacheWrite5m.Value == nil || *decoded.Usage.InputCacheWrite5m.Value != 20 {
		t.Fatalf("expected 5m cache write 20, got: %v", decoded.Usage.InputCacheWrite5m)
	}
	if decoded.Usage.InputCacheWrite1h.Value == nil || *decoded.Usage.InputCacheWrite1h.Value != 10 {
		t.Fatalf("expected 1h cache write 10, got: %v", decoded.Usage.InputCacheWrite1h)
	}
}
