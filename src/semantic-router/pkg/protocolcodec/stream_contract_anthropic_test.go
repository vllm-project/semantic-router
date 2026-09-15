package protocolcodec

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var officialAnthropicStreamEvents = fields(
	"content_block_delta", "content_block_start", "content_block_stop", "error",
	"message_delta", "message_start", "message_stop", "ping",
)

var officialSupportedAnthropicStreamContentBlocks = fields("text", "thinking", "tool_use")

var officialUnsupportedAnthropicStreamContentBlocks = fields(
	"bash_code_execution_tool_result", "code_execution_tool_result", "container_upload",
	"redacted_thinking", "server_tool_use", "text_editor_code_execution_tool_result",
	"tool_search_tool_result", "web_fetch_tool_result", "web_search_tool_result",
)

var officialSupportedAnthropicStreamDeltas = fields(
	"input_json_delta", "signature_delta", "text_delta", "thinking_delta",
)

var officialUnsupportedAnthropicStreamDeltas = fields("citations_delta")

func TestOfficialAnthropicStreamUnionFieldsAreExplicit(t *testing.T) {
	assertClosedDiscriminatorInventory(t, "Anthropic stream event", 8, officialAnthropicStreamEvents, nil)
	assertClosedDiscriminatorInventory(
		t, "Anthropic stream content block", 12,
		officialSupportedAnthropicStreamContentBlocks,
		officialUnsupportedAnthropicStreamContentBlocks,
	)
	assertClosedDiscriminatorInventory(
		t, "Anthropic stream content delta", 5,
		officialSupportedAnthropicStreamDeltas,
		officialUnsupportedAnthropicStreamDeltas,
	)
	assertOfficialAnthropicStreamMetadata(t)
	for _, test := range officialUnsupportedAnthropicStreamCases() {
		t.Run(test.name, func(t *testing.T) { assertUnsupportedAnthropicStreamCase(t, test) })
	}
}

func assertOfficialAnthropicStreamMetadata(t *testing.T) {
	t.Helper()
	decoder := AnthropicMessagesCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
		llmprotocol.DefaultPolicy(),
	)
	start, err := encodeSSE("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": "msg_1", "type": "message", "role": "assistant", "model": "provider-model",
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": 3, "output_tokens": 0},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := decoder.Push(start); err != nil {
		t.Fatalf("message_start: %v", err)
	}
	delta, err := encodeSSE("message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"container": map[string]any{"id": "container_1"},
			"stop_details": map[string]any{
				"type": "refusal", "category": "general_harms", "explanation": "unsafe",
			},
			"stop_reason": "refusal", "stop_sequence": nil,
		},
		"usage": map[string]any{"input_tokens": 3, "output_tokens": 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, diagnostics, err := decoder.Push(delta)
	if err != nil {
		t.Fatalf("official message_delta fields were rejected: %v", err)
	}
	fields := map[string]bool{}
	for _, diagnostic := range diagnostics {
		fields[diagnostic.Field] = true
	}
	if !fields["stream.delta.container"] || !fields["stream.delta.stop_details"] {
		t.Fatalf("Anthropic stream metadata omissions were not explicit: %+v", diagnostics)
	}
}

type unsupportedAnthropicStreamCase struct {
	name  string
	event map[string]any
	code  string
}

func officialUnsupportedAnthropicStreamCases() []unsupportedAnthropicStreamCase {
	return []unsupportedAnthropicStreamCase{
		{
			name: "citation delta",
			event: map[string]any{
				"type": "content_block_delta", "index": 0,
				"delta": map[string]any{"type": "citations_delta", "citation": map[string]any{"type": "char_location"}},
			},
			code: "unknown_stream_delta",
		},
		{
			name: "redacted thinking block",
			event: map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "redacted_thinking", "data": "opaque"},
			},
			code: "unsupported_content",
		},
		{
			name: "server tool use block",
			event: map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "server_tool_use", "id": "tool_1", "name": "web_search", "input": map[string]any{}},
			},
			code: "unsupported_content",
		},
		{
			name: "web search result block",
			event: map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "web_search_tool_result", "tool_use_id": "tool_1", "content": []any{}},
			},
			code: "unsupported_content",
		},
		{
			name: "web fetch result block",
			event: map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "web_fetch_tool_result", "tool_use_id": "tool_1", "content": map[string]any{}},
			},
			code: "unsupported_content",
		},
		{
			name: "code execution result block",
			event: map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "code_execution_tool_result", "tool_use_id": "tool_1", "content": map[string]any{}},
			},
			code: "unsupported_content",
		},
		{
			name: "bash execution result block",
			event: map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "bash_code_execution_tool_result", "tool_use_id": "tool_1", "content": []any{}},
			},
			code: "unsupported_content",
		},
		{
			name: "text editor result block",
			event: map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "text_editor_code_execution_tool_result", "tool_use_id": "tool_1", "content": map[string]any{}},
			},
			code: "unsupported_content",
		},
		{
			name: "tool search result block",
			event: map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "tool_search_tool_result", "tool_use_id": "tool_1", "content": map[string]any{}},
			},
			code: "unsupported_content",
		},
		{
			name: "container upload block",
			event: map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "container_upload", "file_id": "file_1"},
			},
			code: "unsupported_content",
		},
	}
}

func assertUnsupportedAnthropicStreamCase(t *testing.T, test unsupportedAnthropicStreamCase) {
	t.Helper()
	eventType, _ := test.event["type"].(string)
	frame, err := encodeSSE(eventType, test.event)
	if err != nil {
		t.Fatal(err)
	}
	decoder := AnthropicMessagesCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
	)
	_, _, err = decoder.Push(frame)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) ||
		protocolError.Category != llmprotocol.ErrorUnsupportedFeature || protocolError.Code != test.code {
		t.Fatalf("returned %T %v, want unsupported_feature/%s", err, err, test.code)
	}
}

func TestAnthropicSignatureDeltaIsPreservedOrExplicitlyRejected(t *testing.T) {
	payload := []byte(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"response_1\",\"model\":\"model\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"signed\"}}\n\n",
	)
	engine := NewBuiltinEngine()
	same, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, err := same.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, event := range events {
		if event.Content != nil && event.Content.Signature == "signed" {
			found = true
		}
	}
	if !found || !bytes.Contains(bytes.Join(frames, nil), []byte(`"signature":"signed"`)) {
		t.Fatalf("signature delta was not preserved: events=%+v wire=%s", events, bytes.Join(frames, nil))
	}
	cross, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := cross.Push(payload); err == nil {
		t.Fatal("signed reasoning was silently dropped across formats")
	}
}

func TestAnthropicSameFormatStreamPreservesOrderedThinkingTextAndToolBlocks(t *testing.T) {
	payload := []byte(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"response_1\",\"model\":\"model\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"first thought\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"signature-1\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"checking\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"second thought\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"signature-2\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call-1\",\"name\":\"lookup\",\"input\":{}}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":3,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"weather\\\"}\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":3}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":7}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	finalFrames, finalEvents, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	frames = append(frames, finalFrames...)
	events = append(events, finalEvents...)
	wire := string(bytes.Join(frames, nil))
	wantOrder := []string{
		`"index":0`, `"thinking":"first thought"`, `"signature":"signature-1"`,
		`"index":1`, `"text":"checking"`,
		`"index":2`, `"thinking":"second thought"`, `"signature":"signature-2"`,
		`"index":3`, `"name":"lookup"`, `"partial_json":"{\"q\":\"weather\"}"`,
		`"type":"message_stop"`,
	}
	position := 0
	for _, marker := range wantOrder {
		relative := strings.Index(wire[position:], marker)
		if relative < 0 {
			t.Fatalf("missing ordered marker %s in %s", marker, wire)
		}
		position += relative + len(marker)
	}
	if len(events) != 17 || events[len(events)-1].Type != llmprotocol.EventResponseCompleted ||
		events[len(events)-1].StopReason != llmprotocol.StopToolCall {
		t.Fatalf("Anthropic stream lifecycle changed: %+v", events)
	}
}

func newTestStreamState() *streamState {
	return &streamState{
		context: llmprotocol.StreamContext{Context: context.Background(), ResponseID: "response_1", PublicModel: "model"},
		policy:  llmprotocol.DefaultPolicy(),
	}
}

func TestStreamOutputAndContentLimitsAreIndependentAndExact(t *testing.T) {
	state := newTestStreamState()
	state.policy.Limits.OutputItems = 1
	state.policy.Limits.ContentBlocks = 2
	startTestStream(t, state)
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0,
		Content: &llmprotocol.Content{Kind: llmprotocol.ContentText},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, ContentIndex: 0,
		Delta: "a", Content: &llmprotocol.Content{Kind: llmprotocol.ContentText},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventReasoningDelta, ItemIndex: 0, ContentIndex: 1,
		Delta: "b", Content: &llmprotocol.Content{Kind: llmprotocol.ContentReasoning},
	}); err != nil {
		t.Fatal(err)
	}
	_, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 1,
		Content: &llmprotocol.Content{Kind: llmprotocol.ContentText},
	})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "output_items_limit")

	state = newTestStreamState()
	state.policy.Limits.ContentBlocks = 1
	startTestStream(t, state)
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, ContentIndex: 0,
		Delta: "a", Content: &llmprotocol.Content{Kind: llmprotocol.ContentText},
	}); err != nil {
		t.Fatal(err)
	}
	_, err = state.next(llmprotocol.Event{
		Type: llmprotocol.EventReasoningDelta, ItemIndex: 0, ContentIndex: 1,
		Delta: "b", Content: &llmprotocol.Content{Kind: llmprotocol.ContentReasoning},
	})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "content_blocks_limit")
}

func TestStreamContentIndexCannotChangeSemanticKind(t *testing.T) {
	state := newTestStreamState()
	startTestStream(t, state)
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, ContentIndex: 0,
		Delta: "answer", Content: &llmprotocol.Content{Kind: llmprotocol.ContentText},
	}); err != nil {
		t.Fatal(err)
	}
	_, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventReasoningDelta, ItemIndex: 0, ContentIndex: 0,
		Delta: "thought", Content: &llmprotocol.Content{Kind: llmprotocol.ContentReasoning},
	})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_content_kind_mismatch")
}

func TestStreamContentIndexCannotChangeReasoningScope(t *testing.T) {
	state := streamState{policy: llmprotocol.DefaultPolicy()}
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseStarted}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "reasoning_1",
		Content: &llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Reasoning: llmprotocol.ReasoningScopeSummary,
		},
	}); err != nil {
		t.Fatal(err)
	}
	_, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventReasoningDelta, ItemIndex: 0, ItemID: "reasoning_1", Delta: "detail",
		Content: &llmprotocol.Content{
			Kind: llmprotocol.ContentReasoning, Reasoning: llmprotocol.ReasoningScopeText,
		},
	})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_reasoning_scope_mismatch")
}

func TestResponseAccumulatorPreservesRepeatedKindsByContentIndex(t *testing.T) {
	accumulator := newResponseAccumulator()
	events := []llmprotocol.Event{
		{Type: llmprotocol.EventResponseStarted, ResponseID: "resp_1", Model: "model"},
		{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "msg_1", Role: llmprotocol.RoleAssistant},
		{Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, ContentIndex: 0, Delta: "first", Content: &llmprotocol.Content{Kind: llmprotocol.ContentText}},
		{Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, ContentIndex: 1, Delta: "second", Content: &llmprotocol.Content{Kind: llmprotocol.ContentText}},
		{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0, ItemID: "msg_1"},
		{Type: llmprotocol.EventResponseCompleted, StopReason: llmprotocol.StopEndTurn},
	}
	if err := accumulator.apply(events); err != nil {
		t.Fatal(err)
	}
	response, err := accumulator.response()
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Output) != 1 || len(response.Output[0].Content) != 2 ||
		response.Output[0].Content[0].Text != "first" || response.Output[0].Content[1].Text != "second" {
		t.Fatalf("content part ordering was not preserved: %+v", response.Output)
	}
}

func startTestStream(t *testing.T, state *streamState) {
	t.Helper()
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseStarted}); err != nil {
		t.Fatal(err)
	}
}

func availableStreamUsage(input, output int64) llmprotocol.Usage {
	total := input + output
	return llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: authoritative(input), InputTotal: authoritative(input),
		OutputOther: authoritative(output), OutputTotal: authoritative(output), Total: authoritative(total),
	}
}
