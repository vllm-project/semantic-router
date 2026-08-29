package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestStreamFinalizeClassifiesCancellation(t *testing.T) {
	state := newTestStreamState()
	events, err := state.finalize(context.Canceled)
	if err != nil || len(events) != 1 || events[0].Type != llmprotocol.EventResponseFailed ||
		events[0].Error == nil || !errors.Is(events[0].Error, context.Canceled) || events[0].Error.Code != "stream_canceled" {
		t.Fatalf("Finalize(canceled) = %+v, %v", events, err)
	}
}

func TestStreamFinalizeClassifiesDeadline(t *testing.T) {
	state := newTestStreamState()
	events, err := state.finalize(context.DeadlineExceeded)
	if err != nil || len(events) != 1 || events[0].Type != llmprotocol.EventResponseFailed ||
		events[0].Error == nil || !errors.Is(events[0].Error, context.DeadlineExceeded) || events[0].Error.Code != "stream_timeout" {
		t.Fatalf("Finalize(deadline) = %+v, %v", events, err)
	}
}

func TestStreamFinalizationReasonsTranslateAcrossEveryProtocolPair(t *testing.T) {
	reasons := []streamReasonCase{
		{name: "canceled", err: context.Canceled, code: "stream_canceled"},
		{name: "deadline", err: context.DeadlineExceeded, code: "stream_timeout"},
		{name: "eof", err: nil, code: "stream_incomplete"},
	}
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		for _, reason := range reasons {
			t.Run(reason.name, func(t *testing.T) { assertStreamFinalizationReason(t, engine, source, target, reason) })
		}
	})
}

type streamReasonCase struct {
	name string
	err  error
	code string
}

func assertStreamFinalizationReason(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat, reason streamReasonCase) {
	t.Helper()
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, err := stream.Finalize(reason.err)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Type != llmprotocol.EventResponseFailed || events[0].Error == nil || events[0].Error.Code != reason.code {
		t.Fatalf("terminal events = %+v, want one %s failure", events, reason.code)
	}
	wire := bytes.Join(frames, nil)
	if bytes.Count(wire, []byte(events[0].Error.Message)) != 1 {
		t.Fatalf("terminal wire did not contain exactly one failure message for %s: %s", reason.code, wire)
	}
	assertStreamHasNoSuccessTerminal(t, wire)
	if finalFrames, finalEvents, _, finalErr := stream.Finalize(nil); finalErr != nil || len(finalFrames) != 0 || len(finalEvents) != 0 {
		t.Fatalf("second finalize was not idempotent: frames=%q events=%+v err=%v", finalFrames, finalEvents, finalErr)
	}
}

func assertStreamHasNoSuccessTerminal(t *testing.T, wire []byte) {
	t.Helper()
	if bytes.Contains(wire, []byte("response.completed")) || bytes.Contains(wire, []byte("message_stop")) || bytes.Contains(wire, []byte("[DONE]")) {
		t.Fatalf("failure stream also emitted a success terminal: %s", wire)
	}
}

func TestMalformedTrailingFrameReturnsParseErrorAndPublicFailure(t *testing.T) {
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertMalformedTrailingFrame(t, engine, source, target)
	})
}

func assertMalformedTrailingFrame(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat) {
	t.Helper()
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	if frames, events, _, pushErr := stream.Push([]byte("data: {\"type\":")); pushErr != nil || len(frames) != 0 || len(events) != 0 {
		t.Fatalf("partial frame was handled before EOF: frames=%q events=%+v err=%v", frames, events, pushErr)
	}
	frames, events, _, finalizeErr := stream.Finalize(context.DeadlineExceeded)
	if finalizeErr == nil {
		t.Fatal("malformed trailing frame did not return its parse error")
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(finalizeErr, &protocolError) || protocolError.Code != "invalid_upstream_json" {
		t.Fatalf("malformed trailing frame error = %T %v", finalizeErr, finalizeErr)
	}
	if len(events) != 0 {
		t.Fatalf("malformed frame produced semantic events: %+v", events)
	}
	wire := bytes.Join(frames, nil)
	if bytes.Count(wire, []byte(protocolError.Message)) != 1 {
		t.Fatalf("public parse-failure terminal is missing or duplicated: %s", wire)
	}
	assertStreamHasNoSuccessTerminal(t, wire)
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

func TestSSEParserHandlesCommentsMetadataAndMultilineData(t *testing.T) {
	framer := newSSEFramer(256)
	frames, err := framer.Push([]byte(
		": keepalive\n" +
			"id: provider-event-1\n" +
			"retry: 250\n" +
			"event: response.created\n" +
			"data: {\"type\":\n" +
			"data: \"response.created\"}\n\n",
	))
	if err != nil || len(frames) != 1 {
		t.Fatalf("SSE metadata frame = %q, %v", frames, err)
	}
	parsed, err := parseSSEFrame(frames[0], 256)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Event != "response.created" || string(parsed.Data) != "{\"type\":\n\"response.created\"}" {
		t.Fatalf("SSE metadata or multiline data changed: event=%q data=%q", parsed.Event, parsed.Data)
	}
	var body map[string]string
	if err := json.Unmarshal(parsed.Data, &body); err != nil || body["type"] != "response.created" {
		t.Fatalf("multiline SSE data is not valid JSON: %v data=%q", err, parsed.Data)
	}
}

func TestStreamsAcceptAnEOFDelimitedFinalSSEEventAcrossProtocolMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertEOFDelimitedFinalEvent(t, engine, source, target)
	})
}

func assertEOFDelimitedFinalEvent(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat) {
	t.Helper()
	payload := bytes.TrimSuffix(streamFixture(source), []byte("\n\n"))
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	finalFrames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("EOF-delimited final event was rejected: %v", err)
	}
	assertSuccessTerminalForFormat(t, target, append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...))
}

func assertSuccessTerminalForFormat(t *testing.T, target llmprotocol.WireFormat, wire []byte) {
	t.Helper()
	terminal := map[llmprotocol.WireFormat][]byte{
		llmprotocol.OpenAIChatV1:        []byte("data: [DONE]"),
		llmprotocol.OpenAIResponsesV1:   []byte("event: response.completed"),
		llmprotocol.AnthropicMessagesV1: []byte("event: message_stop"),
	}[target]
	if !bytes.Contains(wire, terminal) {
		t.Fatalf("%s success terminal is missing: %s", target, wire)
	}
}

func TestSSEParserAcceptsLeadingUTF8BOM(t *testing.T) {
	frame := append([]byte{0xef, 0xbb, 0xbf}, []byte("event: ping\ndata: {\"type\":\"ping\"}\n\n")...)
	parsed, err := parseSSEFrame(frame, 256)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Event != "ping" || string(parsed.Data) != `{"type":"ping"}` {
		t.Fatalf("BOM-prefixed SSE frame = event %q data %q", parsed.Event, parsed.Data)
	}
}

func TestSSEFramerAcceptsExactLimitAndRejectsOneByteOver(t *testing.T) {
	frame := []byte("data: 1234\n\n")
	framer := newSSEFramer(len(frame))
	frames, err := framer.Push(frame)
	if err != nil || len(frames) != 1 {
		t.Fatalf("exact-limit frame = %q, %v", frames, err)
	}
	framer = newSSEFramer(len(frame) - 1)
	if _, err := framer.Push(frame); err == nil {
		t.Fatal("frame one byte over the limit was accepted")
	}
}

func TestStreamMatrixAcceptsUnicodeSplitAtEveryByte(t *testing.T) {
	payload := []byte(
		"data: {\"id\":\"response_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"你好🌍\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"response_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: {\"id\":\"response_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}}\n\n" +
			"data: [DONE]\n\n",
	)
	engine := NewBuiltinEngine()
	for _, target := range builtinFormats {
		t.Run(string(target), func(t *testing.T) {
			assertUnicodeByteSplitStream(t, engine, target, payload)
		})
	}
}

func assertUnicodeByteSplitStream(t *testing.T, engine *Engine, target llmprotocol.WireFormat, payload []byte) {
	t.Helper()
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
		Options: llmprotocol.StreamOptions{IncludeUsage: boolPointer(true)},
	})
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	for _, value := range payload {
		frames, _, _, err := stream.Push([]byte{value})
		if err != nil {
			t.Fatal(err)
		}
		for _, frame := range frames {
			output.Write(frame)
		}
	}
	frames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, frame := range frames {
		output.Write(frame)
	}
	if !bytes.Contains(output.Bytes(), []byte("你好🌍")) {
		t.Fatalf("Unicode text changed after byte-split translation: %s", output.Bytes())
	}
}

func TestStreamFinalizeValidatesTrailingBytesAfterSemanticTerminal(t *testing.T) {
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertTrailingBytesAfterTerminal(t, engine, source, target)
	})
}

func assertTrailingBytesAfterTerminal(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat) {
	t.Helper()
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
	if err != nil {
		t.Fatal(err)
	}
	payload := append(append([]byte(nil), streamFixture(source)...), []byte("data: {\"trailing\":")...)
	frames, events, _, pushErr := stream.Push(payload)
	if pushErr != nil {
		t.Fatalf("terminal push failed before trailing fragment was finalized: %v", pushErr)
	}
	assertNoCompletedEvent(t, events, "success escaped before trailing bytes were validated")
	assertNoSuccessfulStreamTerminal(t, target, bytes.Join(frames, nil))
	finalFrames, _, _, finalizeErr := stream.Finalize(nil)
	if finalizeErr == nil {
		t.Fatal("trailing partial frame was silently ignored")
	}
	wire := append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...)
	assertNoSuccessfulStreamTerminal(t, target, wire)
	if !bytes.Contains(wire, []byte("stream")) {
		t.Fatalf("trailing partial frame did not publish a failure terminal: %s", wire)
	}
	if frames, events, diagnostics, finalizeErr := stream.Finalize(nil); finalizeErr != nil || len(frames) != 0 || len(events) != 0 || len(diagnostics) != 0 {
		t.Fatalf("second finalize was not idempotent: frames=%q events=%+v diagnostics=%+v err=%v", frames, events, diagnostics, finalizeErr)
	}
}

func TestStreamRejectsPushAfterTerminalAndFinalizesOnce(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, format := range builtinFormats {
		t.Run(string(format), func(t *testing.T) {
			assertPushAfterTerminalRejected(t, engine, format)
		})
	}
}

func assertPushAfterTerminalRejected(t *testing.T, engine *Engine, format llmprotocol.WireFormat) {
	t.Helper()
	stream, err := engine.NewStream(format, format, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, pushErr := stream.Push(streamFixture(format))
	if pushErr != nil {
		t.Fatal(pushErr)
	}
	assertNoCompletedEvent(t, events, "success was visible before transport finalization")
	assertNoSuccessfulStreamTerminal(t, format, bytes.Join(frames, nil))
	finalFrames, finalEvents, _, finalizeErr := stream.Finalize(nil)
	if finalizeErr != nil {
		t.Fatalf("clean terminal finalize failed: %v", finalizeErr)
	}
	if len(finalEvents) != 1 || finalEvents[0].Type != llmprotocol.EventResponseCompleted {
		t.Fatalf("finalized terminal events = %+v", finalEvents)
	}
	if len(append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...)) == 0 {
		t.Fatal("clean terminal stream is empty")
	}
	if _, _, _, pushErr := stream.Push([]byte("\n")); pushErr == nil {
		t.Fatal("push after finalized terminal was accepted")
	}
	if _, _, _, finalizeErr := stream.Finalize(context.Canceled); finalizeErr != nil {
		t.Fatalf("idempotent finalize synthesized a second terminal: %v", finalizeErr)
	}
}

func TestResponsesRefusalDeltaRemainsRefusalAcrossStreamingFormats(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte(
		"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"response_1\",\"model\":\"model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n" +
			"event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":2,\"output_index\":0,\"content_index\":0,\"item_id\":\"output_1\",\"part\":{\"type\":\"refusal\",\"refusal\":\"\"}}\n\n" +
			"event: response.refusal.delta\ndata: {\"type\":\"response.refusal.delta\",\"sequence_number\":3,\"output_index\":0,\"content_index\":0,\"item_id\":\"output_1\",\"delta\":\"cannot comply\"}\n\n" +
			"event: response.refusal.done\ndata: {\"type\":\"response.refusal.done\",\"sequence_number\":4,\"output_index\":0,\"content_index\":0,\"item_id\":\"output_1\",\"refusal\":\"cannot comply\"}\n\n" +
			"event: response.content_part.done\ndata: {\"type\":\"response.content_part.done\",\"sequence_number\":5,\"output_index\":0,\"content_index\":0,\"item_id\":\"output_1\",\"part\":{\"type\":\"refusal\",\"refusal\":\"cannot comply\"}}\n\n" +
			"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":6,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"cannot comply\"}]}}\n\n" +
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":7,\"response\":{\"id\":\"response_1\",\"model\":\"model\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"cannot comply\"}]}],\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
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
		"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"response_1\",\"model\":\"model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n" +
			"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"item_1\",\"delta\":\"{\\\"q\\\":\\\"x\\\"}\"}\n\n" +
			"event: response.function_call_arguments.done\ndata: {\"type\":\"response.function_call_arguments.done\",\"sequence_number\":3,\"output_index\":0,\"item_id\":\"item_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"x\\\"}\"}\n\n" +
			"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":4,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"x\\\"}\",\"status\":\"completed\"}}\n\n" +
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":5,\"response\":{\"id\":\"response_1\",\"model\":\"model\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"x\\\"}\",\"status\":\"completed\"}],\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
	)
	frames, _, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	encoded := bytes.Join(frames, nil)
	if !bytes.Contains(encoded, []byte(`"type":"function_call"`)) || !bytes.Contains(encoded, []byte(`"arguments":"{\"q\":\"x\"}"`)) {
		t.Fatalf("tool completion was not encoded as a function call: %s", encoded)
	}
	if strings.Count(string(encoded), `"type":"response.function_call_arguments.done"`) != 1 ||
		!bytes.Contains(encoded, []byte(`"name":"lookup"`)) {
		t.Fatalf("tool argument lifecycle is incomplete: %s", encoded)
	}
	for _, frame := range bytes.Split(encoded, []byte("\n\n")) {
		if bytes.Contains(frame, []byte(`"type":"response.function_call_arguments.delta"`)) &&
			bytes.Contains(frame, []byte(`"name":`)) {
			t.Fatalf("tool delta contains a done-only name field: %s", frame)
		}
	}
}

func TestResponsesTextEncoderEmitsCompleteContentLifecycleOnce(t *testing.T) {
	encoder := OpenAIResponsesCodec{}.NewEncoder(
		llmprotocol.StreamContext{
			Context: context.Background(), PublicModel: "model", ResponseID: "response_1",
			PreviousResponseID: "response_previous",
		},
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
	if count := strings.Count(wire, `"previous_response_id":"response_previous"`); count != 3 {
		t.Fatalf("previous_response_id appeared %d times, want created, in_progress, and completed: %s", count, wire)
	}
	wantOrder := []string{
		`"type":"response.created"`,
		`"type":"response.in_progress"`,
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

func TestResponsesEncoderRejectsRepeatedSemanticStartEvents(t *testing.T) {
	encoder := OpenAIResponsesCodec{}.NewEncoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model", ResponseID: "response_1"},
		llmprotocol.DefaultPolicy(),
	)
	var encoded bytes.Buffer
	for attempt := 0; attempt < 3; attempt++ {
		frames, _, err := encoder.Push(llmprotocol.Event{
			Type: llmprotocol.EventResponseStarted, ResponseID: "response_1", Model: "model",
		})
		if attempt == 0 && err != nil {
			t.Fatal(err)
		}
		if attempt > 0 {
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "duplicate_stream_start")
			continue
		}
		for _, frame := range frames {
			encoded.Write(frame)
		}
	}
	for _, eventType := range []string{"response.created", "response.in_progress"} {
		if count := strings.Count(encoded.String(), `"type":"`+eventType+`"`); count != 1 {
			t.Fatalf("%s emitted %d times, want once: %s", eventType, count, encoded.String())
		}
	}
}

func TestPublicChatStreamUsageIsExplicitAndEmittedOnce(t *testing.T) {
	engine := NewBuiltinEngine()
	response := llmprotocol.Response{
		Generation: 1,
		ID:         "response_1",
		Model:      "public-model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_1", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}},
		}},
		StopReason: llmprotocol.StopEndTurn,
		Usage:      availableStreamUsage(2, 1),
	}
	for _, test := range []struct {
		name      string
		requested *bool
		wantUsage int
	}{
		{name: "omitted", wantUsage: 0},
		{name: "false", requested: boolPointer(false), wantUsage: 0},
		{name: "true", requested: boolPointer(true), wantUsage: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			body, _, err := engine.EncodeResponseStream(llmprotocol.OpenAIChatV1, response, llmprotocol.StreamContext{
				Context: context.Background(), Options: llmprotocol.StreamOptions{IncludeUsage: test.requested},
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.Count(string(body), `"usage":`); got != test.wantUsage {
				t.Fatalf("usage chunks = %d, want %d\n%s", got, test.wantUsage, body)
			}
		})
	}
}

func TestTranslatedStreamUsageIsNeverDuplicated(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public-model",
				Options: llmprotocol.StreamOptions{IncludeUsage: boolPointer(true)},
			})
			if err != nil {
				t.Fatal(err)
			}
			frames, _, _, err := stream.Push(streamFixture(llmprotocol.OpenAIChatV1))
			if err != nil {
				t.Fatal(err)
			}
			final, _, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			wire := string(bytes.Join(append(frames, final...), nil))
			marker := `"usage":`
			if target == llmprotocol.AnthropicMessagesV1 {
				marker = `"type":"message_delta"`
			}
			if got := strings.Count(wire, marker); got != 1 {
				t.Fatalf("terminal usage marker %q appeared %d times, want once\n%s", marker, got, wire)
			}
		})
	}
}

func TestExplicitStreamObfuscationIsRenderedOnOpenAIStreams(t *testing.T) {
	engine := NewBuiltinEngine()
	response := llmprotocol.Response{
		Generation: 1,
		ID:         "response_1",
		Model:      "public-model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_1", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}},
		}},
		StopReason: llmprotocol.StopEndTurn,
		Usage:      availableStreamUsage(2, 1),
	}
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(format), func(t *testing.T) {
			body, _, err := engine.EncodeResponseStream(format, response, llmprotocol.StreamContext{
				Context: context.Background(), Options: llmprotocol.StreamOptions{IncludeObfuscation: boolPointer(true)},
			})
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(body, []byte(`"obfuscation":"`)) {
				t.Fatalf("explicit obfuscation was not rendered: %s", body)
			}
		})
	}
}

func TestResponsesFailedReadsNestedResponseError(t *testing.T) {
	decoder := OpenAIResponsesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"}, llmprotocol.DefaultPolicy())
	payload := []byte("event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":0,\"response\":{\"id\":\"response_1\",\"status\":\"failed\",\"error\":{\"code\":\"provider_overloaded\",\"message\":\"try later\"}}}\n\n")
	events, _, err := decoder.Push(payload)
	if err != nil || len(events) != 1 || events[0].Error == nil || events[0].Error.Code != "provider_overloaded" {
		t.Fatalf("nested response error = %+v, %v", events, err)
	}
}
