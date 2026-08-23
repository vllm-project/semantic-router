package protocolcodec

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestStreamStateRejectsDecreasingUsage(t *testing.T) {
	state := newTestStreamState()
	startTestStream(t, state)
	first := availableStreamUsage(4, 1)
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &first}); err != nil {
		t.Fatal(err)
	}
	decreased := availableStreamUsage(3, 1)
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &decreased}); err == nil {
		t.Fatal("decreasing streaming usage was accepted")
	}
}

func TestStreamStateRejectsUsageEvidenceDowngrade(t *testing.T) {
	state := newTestStreamState()
	startTestStream(t, state)
	first := availableStreamUsage(4, 1)
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &first}); err != nil {
		t.Fatal(err)
	}
	input, output, total := int64(4), int64(1), int64(5)
	downgraded := llmprotocol.Usage{
		State:         llmprotocol.UsageAvailable,
		InputUncached: llmprotocol.TokenCount{Value: &input, Provenance: llmprotocol.UsageEstimated},
		InputTotal:    llmprotocol.TokenCount{Value: &input, Provenance: llmprotocol.UsageEstimated},
		OutputOther:   llmprotocol.TokenCount{Value: &output, Provenance: llmprotocol.UsageAuthoritative},
		OutputTotal:   llmprotocol.TokenCount{Value: &output, Provenance: llmprotocol.UsageAuthoritative},
		Total:         llmprotocol.TokenCount{Value: &total, Provenance: llmprotocol.UsageDerived},
	}
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &downgraded}); err == nil {
		t.Fatal("authoritative usage was replaced by estimated evidence")
	}
}

func TestStreamStateRejectsAvailableToUnknownUsage(t *testing.T) {
	state := newTestStreamState()
	startTestStream(t, state)
	first := availableStreamUsage(4, 1)
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &first}); err != nil {
		t.Fatal(err)
	}
	unknown := llmprotocol.Usage{State: llmprotocol.UsageUnavailable}
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &unknown}); err == nil {
		t.Fatal("available usage became unknown")
	}
}

func TestStreamStateRequiresCompleteToolLifecycle(t *testing.T) {
	state := newTestStreamState()
	startTestStream(t, state)
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "tool_item",
		Role: llmprotocol.RoleAssistant, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup"},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventToolCallDelta, ItemIndex: 0,
		ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: `{"query":`},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0}); err == nil {
		t.Fatal("incomplete streamed tool arguments were accepted")
	}
}

func TestStreamStateRejectsMalformedAndNonObjectToolArguments(t *testing.T) {
	for _, arguments := range []string{`{`, `[]`, `true`, `null`, `{"query":1,"query":2}`} {
		t.Run(arguments, func(t *testing.T) {
			state := newTestStreamState()
			startTestStream(t, state)
			if _, err := state.next(llmprotocol.Event{
				Type:      llmprotocol.EventOutputItemStarted,
				ItemIndex: 0,
				ItemID:    "tool_item",
				Role:      llmprotocol.RoleAssistant,
				ToolCall:  &llmprotocol.ToolCall{ID: "call_1", Name: "lookup"},
			}); err != nil {
				t.Fatal(err)
			}
			if _, err := state.next(llmprotocol.Event{
				Type:      llmprotocol.EventOutputItemCompleted,
				ItemIndex: 0,
				ToolCall:  &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: arguments},
			}); err == nil {
				t.Fatalf("streamed tool arguments %q were accepted", arguments)
			}
		})
	}
}

func TestStreamStateAcceptsExplicitUnknownTerminalUsage(t *testing.T) {
	state := newTestStreamState()
	startTestStream(t, state)
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "item_1", Role: llmprotocol.RoleAssistant}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, Delta: "hello"}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0}); err != nil {
		t.Fatal(err)
	}
	terminal, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: llmprotocol.StopEndTurn})
	if err != nil {
		t.Fatal(err)
	}
	if terminal.Usage == nil || terminal.Usage.State != llmprotocol.UsageUnavailable {
		t.Fatalf("terminal usage = %+v", terminal.Usage)
	}
}

func TestStreamStateRejectsMalformedTerminalShapes(t *testing.T) {
	state := newTestStreamState()
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseFailed, StopReason: llmprotocol.StopError}); err == nil {
		t.Fatal("failed terminal event without an error was accepted")
	}

	state = newTestStreamState()
	startTestStream(t, state)
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted}); err == nil {
		t.Fatal("successful terminal event without output was accepted")
	}
}

func TestStreamFinalizeClassifiesCancellation(t *testing.T) {
	state := newTestStreamState()
	events, err := state.finalize(context.Canceled)
	if err != nil || len(events) != 1 || events[0].Type != llmprotocol.EventResponseFailed ||
		events[0].Error == nil || !errors.Is(events[0].Error, context.Canceled) || events[0].Error.Code != "stream_canceled" {
		t.Fatalf("Finalize(canceled) = %+v, %v", events, err)
	}
}

func TestSSEFramerBoundsIncompleteAndMultipleFrames(t *testing.T) {
	framer := newSSEFramer(32)
	frames, err := framer.Push([]byte("data: one\n\ndata: two\n\n"))
	if err != nil || len(frames) != 2 {
		t.Fatalf("multiple frames = %q, %v", frames, err)
	}
	framer = newSSEFramer(8)
	if _, err := framer.Push([]byte("data: unfinished")); err == nil {
		t.Fatal("oversized unfinished SSE frame was accepted")
	}
}

func TestSSEFramerAcceptsCROnlyAndSplitCRLF(t *testing.T) {
	framer := newSSEFramer(64)
	frames, err := framer.Push([]byte("data: one\r\rdata: two\r\r"))
	if err != nil || len(frames) != 2 {
		t.Fatalf("CR-only frames = %q, %v", frames, err)
	}
	for _, frame := range frames {
		parsed, parseErr := parseSSEFrame(frame, 64)
		if parseErr != nil || string(parsed.Data) == "" {
			t.Fatalf("CR-only frame was not parsed: %q, %v", frame, parseErr)
		}
	}
	framer = newSSEFramer(64)
	if frames, err = framer.Push([]byte("data: split\r")); err != nil || len(frames) != 0 {
		t.Fatalf("split prefix = %q, %v", frames, err)
	}
	frames, err = framer.Push([]byte("\n\r\n"))
	if err != nil || len(frames) != 1 {
		t.Fatalf("split CRLF = %q, %v", frames, err)
	}
	parsed, parseErr := parseSSEFrame(frames[0], 64)
	if parseErr != nil || string(parsed.Data) != "split" {
		t.Fatalf("split CRLF payload = %q, %q, %v", frames[0], parsed.Data, parseErr)
	}
}

func TestStreamFinalizeValidatesTrailingBytesAfterSemanticTerminal(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, format := range builtinFormats {
		t.Run(string(format), func(t *testing.T) {
			stream, err := engine.NewStream(format, format, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
			if err != nil {
				t.Fatal(err)
			}
			payload := append(append([]byte(nil), streamFixture(format)...), []byte("data: {\"trailing\":")...)
			if _, events, _, pushErr := stream.Push(payload); pushErr != nil {
				t.Fatalf("terminal push failed before trailing fragment was finalized: %v", pushErr)
			} else if len(events) == 0 || events[len(events)-1].Type != llmprotocol.EventResponseCompleted {
				t.Fatalf("terminal event missing: %+v", events)
			}
			if _, _, _, finalizeErr := stream.Finalize(nil); finalizeErr == nil {
				t.Fatal("trailing partial frame was silently ignored")
			}
			if frames, events, diagnostics, finalizeErr := stream.Finalize(nil); finalizeErr != nil || len(frames) != 0 || len(events) != 0 || len(diagnostics) != 0 {
				t.Fatalf("second finalize was not idempotent: frames=%q events=%+v diagnostics=%+v err=%v", frames, events, diagnostics, finalizeErr)
			}
		})
	}
}

func TestStreamRejectsPushAfterTerminalAndFinalizesOnce(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, format := range builtinFormats {
		t.Run(string(format), func(t *testing.T) {
			stream, err := engine.NewStream(format, format, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
			if err != nil {
				t.Fatal(err)
			}
			if _, events, _, pushErr := stream.Push(streamFixture(format)); pushErr != nil {
				t.Fatal(pushErr)
			} else if len(events) == 0 || events[len(events)-1].Type != llmprotocol.EventResponseCompleted {
				t.Fatalf("terminal event missing: %+v", events)
			}
			if _, _, _, pushErr := stream.Push([]byte("\n")); pushErr == nil {
				t.Fatal("push after semantic terminal was accepted")
			}
			if _, _, _, finalizeErr := stream.Finalize(nil); finalizeErr != nil {
				t.Fatalf("clean terminal finalize failed: %v", finalizeErr)
			}
			if _, _, _, finalizeErr := stream.Finalize(context.Canceled); finalizeErr != nil {
				t.Fatalf("idempotent finalize synthesized a second terminal: %v", finalizeErr)
			}
		})
	}
}

func TestResponsesRefusalDeltaRemainsRefusalAcrossStreamingFormats(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte(
		"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"response_1\",\"model\":\"model\",\"status\":\"in_progress\"}}\n\n" +
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\"}}\n\n" +
			"event: response.refusal.delta\ndata: {\"type\":\"response.refusal.delta\",\"output_index\":0,\"item_id\":\"output_1\",\"delta\":\"cannot comply\"}\n\n" +
			"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\",\"status\":\"completed\"}}\n\n" +
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"response_1\",\"model\":\"model\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
	)
	frames, events, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	foundRefusal := false
	for _, event := range events {
		if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
			foundRefusal = true
		}
	}
	if !foundRefusal {
		t.Fatalf("neutral refusal event missing: %+v", events)
	}
	if !bytes.Contains(bytes.Join(frames, nil), []byte(`"refusal":"cannot comply"`)) {
		t.Fatalf("Chat target lost refusal semantics: %s", bytes.Join(frames, nil))
	}
}

func TestResponsesToolCompletionUsesFunctionCallItem(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte(
		"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"response_1\",\"model\":\"model\",\"status\":\"in_progress\"}}\n\n" +
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{}\"}}\n\n" +
			"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"item_id\":\"call_1\",\"delta\":\"{\\\"q\\\":\\\"x\\\"}\"}\n\n" +
			"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"x\\\"}\"}}\n\n" +
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"response_1\",\"model\":\"model\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
	)
	frames, _, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	encoded := bytes.Join(frames, nil)
	if !bytes.Contains(encoded, []byte(`"type":"function_call"`)) || !bytes.Contains(encoded, []byte(`"arguments":"{\"q\":\"x\"}"`)) {
		t.Fatalf("tool completion was not encoded as a function call: %s", encoded)
	}
}

func TestResponsesTextEncoderEmitsCompleteContentLifecycleOnce(t *testing.T) {
	encoder := OpenAIResponsesCodec{}.NewEncoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model", ResponseID: "response_1"},
		llmprotocol.DefaultPolicy(),
	)
	events := []llmprotocol.Event{
		{Type: llmprotocol.EventResponseStarted, ResponseID: "response_1", Model: "model"},
		{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "item_1", Role: llmprotocol.RoleAssistant},
		{Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, ItemID: "item_1", Delta: "hello", Content: &llmprotocol.Content{Kind: llmprotocol.ContentText}},
		{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0, ItemID: "item_1", StopReason: llmprotocol.StopEndTurn},
	}
	usage := availableStreamUsage(2, 1)
	events = append(events, llmprotocol.Event{
		Type: llmprotocol.EventResponseCompleted, ResponseID: "response_1", Model: "model",
		StopReason: llmprotocol.StopEndTurn, Usage: &usage,
	})
	var encoded bytes.Buffer
	for _, event := range events {
		frames, _, err := encoder.Push(event)
		if err != nil {
			t.Fatal(err)
		}
		for _, frame := range frames {
			encoded.Write(frame)
		}
	}
	wire := encoded.String()
	wantOrder := []string{
		`"type":"response.output_item.added"`,
		`"type":"response.content_part.added"`,
		`"type":"response.output_text.delta"`,
		`"type":"response.output_text.done"`,
		`"type":"response.content_part.done"`,
		`"type":"response.output_item.done"`,
		`"type":"response.completed"`,
	}
	position := 0
	for _, marker := range wantOrder {
		relative := bytes.Index([]byte(wire[position:]), []byte(marker))
		if relative < 0 {
			t.Fatalf("missing lifecycle marker %s in %s", marker, wire)
		}
		position += relative + len(marker)
		if bytes.Count([]byte(wire), []byte(marker)) != 1 {
			t.Fatalf("lifecycle marker %s was not emitted exactly once: %s", marker, wire)
		}
	}
}

func TestResponsesFailedReadsNestedResponseError(t *testing.T) {
	decoder := OpenAIResponsesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"}, llmprotocol.DefaultPolicy())
	payload := []byte("event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"id\":\"response_1\",\"status\":\"failed\",\"error\":{\"code\":\"provider_overloaded\",\"message\":\"try later\"}}}\n\n")
	events, _, err := decoder.Push(payload)
	if err != nil || len(events) != 1 || events[0].Error == nil || events[0].Error.Code != "provider_overloaded" {
		t.Fatalf("nested response error = %+v, %v", events, err)
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

func newTestStreamState() *streamState {
	return &streamState{
		context: llmprotocol.StreamContext{Context: context.Background(), ResponseID: "response_1", PublicModel: "model"},
		policy:  llmprotocol.DefaultPolicy(),
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
