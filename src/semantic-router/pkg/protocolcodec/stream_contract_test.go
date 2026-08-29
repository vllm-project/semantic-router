package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	reasons := []struct {
		name string
		err  error
		code string
	}{
		{name: "canceled", err: context.Canceled, code: "stream_canceled"},
		{name: "deadline", err: context.DeadlineExceeded, code: "stream_timeout"},
		{name: "eof", err: nil, code: "stream_incomplete"},
	}
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			for _, reason := range reasons {
				t.Run(string(source)+"/"+string(target)+"/"+reason.name, func(t *testing.T) {
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
					if len(events) != 1 || events[0].Type != llmprotocol.EventResponseFailed ||
						events[0].Error == nil || events[0].Error.Code != reason.code {
						t.Fatalf("terminal events = %+v, want one %s failure", events, reason.code)
					}
					wire := bytes.Join(frames, nil)
					if bytes.Count(wire, []byte(events[0].Error.Message)) != 1 {
						t.Fatalf("terminal wire did not contain exactly one failure message for %s: %s", reason.code, wire)
					}
					if bytes.Contains(wire, []byte("response.completed")) ||
						bytes.Contains(wire, []byte("message_stop")) || bytes.Contains(wire, []byte("[DONE]")) {
						t.Fatalf("failure stream also emitted a success terminal: %s", wire)
					}
					if finalFrames, finalEvents, _, finalErr := stream.Finalize(nil); finalErr != nil || len(finalFrames) != 0 || len(finalEvents) != 0 {
						t.Fatalf("second finalize was not idempotent: frames=%q events=%+v err=%v", finalFrames, finalEvents, finalErr)
					}
				})
			}
		}
	}
}

func TestMalformedTrailingFrameReturnsParseErrorAndPublicFailure(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"/"+string(target), func(t *testing.T) {
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
				if bytes.Contains(wire, []byte("response.completed")) ||
					bytes.Contains(wire, []byte("message_stop")) || bytes.Contains(wire, []byte("[DONE]")) {
					t.Fatalf("malformed stream also emitted a success terminal: %s", wire)
				}
			})
		}
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
	for _, source := range builtinFormats {
		payload := bytes.TrimSuffix(streamFixture(source), []byte("\n\n"))
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
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
				wire := append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...)
				switch target {
				case llmprotocol.OpenAIChatV1:
					if !bytes.Contains(wire, []byte("data: [DONE]")) {
						t.Fatalf("Chat success terminal is missing: %s", wire)
					}
				case llmprotocol.OpenAIResponsesV1:
					if !bytes.Contains(wire, []byte("event: response.completed")) {
						t.Fatalf("Responses success terminal is missing: %s", wire)
					}
				case llmprotocol.AnthropicMessagesV1:
					if !bytes.Contains(wire, []byte("event: message_stop")) {
						t.Fatalf("Messages success terminal is missing: %s", wire)
					}
				}
			})
		}
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
		})
	}
}

func TestStreamFinalizeValidatesTrailingBytesAfterSemanticTerminal(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
				stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
				if err != nil {
					t.Fatal(err)
				}
				payload := append(append([]byte(nil), streamFixture(source)...), []byte("data: {\"trailing\":")...)
				frames, events, _, pushErr := stream.Push(payload)
				if pushErr != nil {
					t.Fatalf("terminal push failed before trailing fragment was finalized: %v", pushErr)
				}
				for _, event := range events {
					if event.Type == llmprotocol.EventResponseCompleted {
						t.Fatalf("success escaped before trailing bytes were validated: %+v", events)
					}
				}
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
			})
		}
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
			frames, events, _, pushErr := stream.Push(streamFixture(format))
			if pushErr != nil {
				t.Fatal(pushErr)
			}
			for _, event := range events {
				if event.Type == llmprotocol.EventResponseCompleted {
					t.Fatalf("success was visible before transport finalization: %+v", events)
				}
			}
			assertNoSuccessfulStreamTerminal(t, format, bytes.Join(frames, nil))
			finalFrames, finalEvents, _, finalizeErr := stream.Finalize(nil)
			if finalizeErr != nil {
				t.Fatalf("clean terminal finalize failed: %v", finalizeErr)
			}
			if len(finalEvents) != 1 || finalEvents[0].Type != llmprotocol.EventResponseCompleted {
				t.Fatalf("finalized terminal events = %+v", finalEvents)
			}
			wire := append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...)
			if len(wire) == 0 {
				t.Fatal("clean terminal stream is empty")
			}
			if _, _, _, pushErr := stream.Push([]byte("\n")); pushErr == nil {
				t.Fatal("push after finalized terminal was accepted")
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

var officialSupportedResponsesStreamEvents = fields(
	"error",
	"response.completed", "response.content_part.added", "response.content_part.done", "response.created",
	"response.failed", "response.function_call_arguments.delta", "response.function_call_arguments.done",
	"response.in_progress", "response.incomplete", "response.output_item.added", "response.output_item.done",
	"response.output_text.annotation.added", "response.output_text.delta", "response.output_text.done",
	"response.queued", "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
	"response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
	"response.reasoning_text.delta", "response.reasoning_text.done",
	"response.refusal.delta", "response.refusal.done",
)

var officialUnsupportedResponsesStreamEvents = fields(
	"response.audio.delta", "response.audio.done", "response.audio.transcript.delta", "response.audio.transcript.done",
	"response.code_interpreter_call_code.delta", "response.code_interpreter_call_code.done",
	"response.code_interpreter_call.completed", "response.code_interpreter_call.in_progress", "response.code_interpreter_call.interpreting",
	"response.custom_tool_call_input.delta", "response.custom_tool_call_input.done",
	"response.file_search_call.completed", "response.file_search_call.in_progress", "response.file_search_call.searching",
	"response.image_generation_call.completed", "response.image_generation_call.generating",
	"response.image_generation_call.in_progress", "response.image_generation_call.partial_image",
	"response.mcp_call_arguments.delta", "response.mcp_call_arguments.done",
	"response.mcp_call.completed", "response.mcp_call.failed", "response.mcp_call.in_progress",
	"response.mcp_list_tools.completed", "response.mcp_list_tools.failed", "response.mcp_list_tools.in_progress",
	"response.shell_call_command.added", "response.shell_call_command.delta", "response.shell_call_command.done",
	"response.shell_call_output_content.delta", "response.shell_call_output_content.done",
	"response.web_search_call.completed", "response.web_search_call.in_progress", "response.web_search_call.searching",
)

func TestOfficialResponsesStreamDiscriminatorInventoryIsClosed(t *testing.T) {
	const officialEventCount = 58
	seen := make(map[string]struct{}, officialEventCount)
	for _, eventType := range officialSupportedResponsesStreamEvents {
		if !isSupportedResponsesEvent(eventType) {
			t.Fatalf("official supported event %q is not registered", eventType)
		}
		seen[eventType] = struct{}{}
	}
	for _, eventType := range officialUnsupportedResponsesStreamEvents {
		if isSupportedResponsesEvent(eventType) {
			t.Fatalf("official unsupported event %q is unexpectedly registered", eventType)
		}
		seen[eventType] = struct{}{}
	}
	if len(seen) != officialEventCount {
		t.Fatalf("Responses stream discriminator inventory has %d unique events, want %d", len(seen), officialEventCount)
	}
}

func TestOfficialUnsupportedResponsesStreamEventsAreTyped(t *testing.T) {
	for _, eventType := range officialUnsupportedResponsesStreamEvents {
		t.Run(eventType, func(t *testing.T) {
			decoder := OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background(), Source: llmprotocol.OpenAIResponsesV1, Target: llmprotocol.OpenAIChatV1},
				llmprotocol.DefaultPolicy(),
			)
			frame, err := encodeSSE(eventType, map[string]any{
				"type":                 eventType,
				"sequence_number":      0,
				"provider_event_field": "must not be decoded as a supported event",
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, err = decoder.Push(frame)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature || protocolError.Code != "unknown_stream_event" {
				t.Fatalf("event %q returned %T %v, want unsupported_feature/unknown_stream_event", eventType, err, err)
			}
		})
	}
}

func TestProviderSSEEventDiscriminatorIsRequiredAndMustMatch(t *testing.T) {
	tests := []struct {
		name    string
		decoder llmprotocol.StreamDecoder
		frame   string
		code    string
	}{
		{
			name: "responses mismatch",
			decoder: OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: response.created\ndata: {\"type\":\"response.completed\",\"sequence_number\":0}\n\n",
			code:  "upstream_event_type_mismatch",
		},
		{
			name: "anthropic mismatch",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: message_start\ndata: {\"type\":\"message_stop\"}\n\n",
			code:  "upstream_event_type_mismatch",
		},
		{
			name: "responses missing",
			decoder: OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "data: {\"sequence_number\":0}\n\n",
			code:  "missing_upstream_event_type",
		},
		{
			name: "anthropic missing",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "data: {\"index\":0}\n\n",
			code:  "missing_upstream_event_type",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := test.decoder.Push([]byte(test.frame))
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUpstreamUnavailable || protocolError.Code != test.code {
				t.Fatalf("returned %T %v, want upstream_unavailable/%s", err, err, test.code)
			}
		})
	}
}

func TestChatStreamResourceIdentityAndEnumsAreClosed(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
		code   string
	}{
		{
			name: "object discriminator",
			chunks: []string{
				"data: {\"id\":\"chat_1\",\"object\":\"chat.completion\",\"model\":\"m\",\"choices\":[]}\n\n",
			},
			code: "invalid_chat_stream_object",
		},
		{
			name: "finish reason",
			chunks: []string{
				"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"done\"},\"finish_reason\":\"future_reason\"}]}\n\n",
			},
			code: "invalid_chat_finish_reason",
		},
		{
			name: "response ID changes",
			chunks: []string{
				"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"first\"},\"finish_reason\":null}]}\n\n",
				"data: {\"id\":\"chat_2\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"},\"finish_reason\":null}]}\n\n",
			},
			code: "stream_response_id_mismatch",
		},
		{
			name: "model changes",
			chunks: []string{
				"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"m1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"first\"},\"finish_reason\":null}]}\n\n",
				"data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"m2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"},\"finish_reason\":null}]}\n\n",
			},
			code: "stream_model_mismatch",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decoder := OpenAIChatCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			)
			var err error
			for _, chunk := range test.chunks {
				_, _, err = decoder.Push([]byte(chunk))
				if err != nil {
					break
				}
			}
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

func TestResponsesStreamResourceIdentityAndStatusAreStable(t *testing.T) {
	tests := []struct {
		name       string
		completion string
		code       string
	}{
		{
			name:       "response ID changes",
			completion: `{"id":"resp_2","object":"response","model":"m","status":"completed","output":[]}`,
			code:       "stream_response_id_mismatch",
		},
		{
			name:       "model changes",
			completion: `{"id":"resp_1","object":"response","model":"other","status":"completed","output":[]}`,
			code:       "stream_model_mismatch",
		},
		{
			name:       "event status mismatch",
			completion: `{"id":"resp_1","object":"response","model":"m","status":"in_progress","output":[]}`,
			code:       "stream_response_status_mismatch",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decoder := OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			)
			start := []byte("event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"m\",\"status\":\"in_progress\",\"output\":[]}}\n\n")
			if _, _, err := decoder.Push(start); err != nil {
				t.Fatal(err)
			}
			terminal := []byte("event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":" + test.completion + "}\n\n")
			_, _, err := decoder.Push(terminal)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

func TestResponsesStreamItemIdentityIsStable(t *testing.T) {
	start := []byte(
		"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"m\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
	)
	tests := []struct {
		name  string
		start bool
		frame string
		code  string
	}{
		{
			name:  "delta item id mismatch",
			start: true,
			frame: "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":2,\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_other\",\"delta\":\"x\"}\n\n",
			code:  "stream_item_id_mismatch",
		},
		{
			name:  "completed item id mismatch",
			start: true,
			frame: "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":2,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_other\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[]}}\n\n",
			code:  "stream_item_id_mismatch",
		},
		{
			name:  "orphan structural event",
			frame: "event: response.content_part.done\ndata: {\"type\":\"response.content_part.done\",\"sequence_number\":0,\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"part\":{\"type\":\"output_text\",\"text\":\"x\",\"annotations\":[]}}\n\n",
			code:  "invalid_item_lifecycle",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decoder := OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			)
			if test.start {
				if _, _, err := decoder.Push(start); err != nil {
					t.Fatal(err)
				}
			}
			_, _, err := decoder.Push([]byte(test.frame))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

func TestProviderStreamRequiredUnionFieldsAreClosed(t *testing.T) {
	tests := []struct {
		name    string
		decoder llmprotocol.StreamDecoder
		frame   string
		code    string
	}{
		{
			name: "responses lifecycle resource",
			decoder: OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0}\n\n",
			code:  "stream_response_required",
		},
		{
			name: "responses output item",
			decoder: OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":0,\"output_index\":0}\n\n",
			code:  "stream_item_required",
		},
		{
			name: "responses delta target",
			decoder: OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"output_index\":0,\"delta\":\"orphan\"}\n\n",
			code:  "stream_delta_target_required",
		},
		{
			name: "responses negative output index",
			decoder: OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":0,\"output_index\":-1,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
			code:  "invalid_stream_item_index",
		},
		{
			name: "responses missing content index",
			decoder: OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"output_index\":0,\"item_id\":\"msg_1\",\"delta\":\"x\"}\n\n",
			code:  "stream_delta_target_required",
		},
		{
			name: "responses missing required delta member",
			decoder: OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\"}\n\n",
			code:  "stream_required_field",
		},
		{
			name: "anthropic start message",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
			code:  "stream_message_required",
		},
		{
			name: "anthropic content block",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0}\n\n",
			code:  "stream_content_block_required",
		},
		{
			name: "anthropic missing content index",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
			code:  "invalid_stream_item_index",
		},
		{
			name: "anthropic negative content index",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":-1,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n",
			code:  "invalid_stream_item_index",
		},
		{
			name: "anthropic message delta",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: message_delta\ndata: {\"type\":\"message_delta\"}\n\n",
			code:  "stream_message_delta_required",
		},
		{
			name: "anthropic message delta usage",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}}\n\n",
			code:  "stream_message_usage_required",
		},
		{
			name: "anthropic content delta member",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\"}}\n\n",
			code:  "stream_required_field",
		},
		{
			name: "anthropic error payload",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"\",\"message\":\"failed\"}}\n\n",
			code:  "invalid_anthropic_stream_error",
		},
		{
			name: "anthropic stop sequence with wrong reason",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":\"END\"},\"usage\":{\"output_tokens\":1}}\n\n",
			code:  "anthropic_stop_sequence_reason",
		},
		{
			name: "anthropic empty stop sequence with no reason",
			decoder: AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			),
			frame: "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":null,\"stop_sequence\":\"\"},\"usage\":{\"output_tokens\":1}}\n\n",
			code:  "anthropic_stop_sequence_reason",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := test.decoder.Push([]byte(test.frame))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

func TestResponsesStreamSequenceNumbersAreRequiredAndContiguous(t *testing.T) {
	for _, secondSequence := range []uint64{0, 2, ^uint64(0)} {
		t.Run(fmt.Sprint(secondSequence), func(t *testing.T) {
			decoder := OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			)
			first, err := encodeSSE("response.queued", map[string]any{
				"type": "response.queued", "sequence_number": 0,
				"response": map[string]any{"id": "resp_1", "object": "response", "created_at": 1, "model": "model", "status": "queued", "output": []any{}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := decoder.Push(first); err != nil {
				t.Fatal(err)
			}
			second, err := encodeSSE("response.in_progress", map[string]any{
				"type": "response.in_progress", "sequence_number": secondSequence,
				"response": map[string]any{"id": "resp_1", "object": "response", "created_at": 1, "model": "model", "status": "in_progress", "output": []any{}},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, err = decoder.Push(second)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_sequence_order")
		})
	}

	decoder := OpenAIResponsesCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
	)
	frame, err := encodeSSE("response.queued", map[string]any{
		"type":     "response.queued",
		"response": map[string]any{"id": "resp_1", "model": "model", "status": "queued"},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = decoder.Push(frame)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "missing_stream_sequence")
}

func TestResponsesStreamSequenceNumberMustBeANonNullUnsignedInteger(t *testing.T) {
	for name, sequence := range map[string]string{
		"null":       "null",
		"negative":   "-1",
		"fractional": "1.5",
		"string":     `"1"`,
		"overflow":   "18446744073709551616",
	} {
		t.Run(name, func(t *testing.T) {
			decoder := OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			)
			frame := []byte("event: response.queued\ndata: {\"type\":\"response.queued\",\"sequence_number\":" + sequence + ",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"model\",\"status\":\"queued\",\"output\":[]}}\n\n")
			_, _, err := decoder.Push(frame)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_sequence")
		})
	}
}

func TestResponsesUnknownEventsCannotBypassSequenceValidation(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.UnknownFields = llmprotocol.UnknownPreserveSameFormat
	decoder := OpenAIResponsesCodec{}.NewDecoder(
		llmprotocol.StreamContext{
			Context: context.Background(), Source: llmprotocol.OpenAIResponsesV1,
			Target: llmprotocol.OpenAIResponsesV1,
		},
		policy,
	)
	first := []byte("event: response.future\ndata: {\"type\":\"response.future\",\"sequence_number\":0}\n\n")
	events, _, err := decoder.Push(first)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Type != llmprotocol.EventProviderOpaque {
		t.Fatalf("unknown event was not preserved: %+v", events)
	}
	gap := []byte("event: response.future\ndata: {\"type\":\"response.future\",\"sequence_number\":2}\n\n")
	_, _, err = decoder.Push(gap)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_sequence_order")
}

func TestResponsesStreamRejectsChatCompletionDoneSentinel(t *testing.T) {
	decoder := OpenAIResponsesCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
	)
	_, _, err := decoder.Push([]byte("data: [DONE]\n\n"))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_responses_stream_sentinel")
}

func TestOfficialResponsesStreamMetadataAndStructuralEventsAreAccepted(t *testing.T) {
	decoder := OpenAIResponsesCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
		llmprotocol.DefaultPolicy(),
	)
	setup := []map[string]any{
		{
			"type": "response.created", "sequence_number": 0,
			"response": map[string]any{
				"id": "response_1", "object": "response", "created_at": 100,
				"model": "provider-model", "status": "in_progress", "output": []any{},
			},
		},
		{
			"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
			"item": map[string]any{
				"type": "message", "id": "item_1", "role": "assistant", "status": "in_progress", "content": []any{},
			},
		},
		{
			"type": "response.output_item.added", "sequence_number": 2, "output_index": 1,
			"item": map[string]any{
				"type": "function_call", "id": "item_2", "call_id": "call_2",
				"name": "lookup", "arguments": "", "status": "in_progress",
			},
		},
		{
			"type": "response.output_item.added", "sequence_number": 3, "output_index": 2,
			"item": map[string]any{
				"type": "reasoning", "id": "item_3", "status": "in_progress", "summary": []any{},
			},
		},
	}
	for _, event := range setup {
		eventType := event["type"].(string)
		frame, err := encodeSSE(eventType, event)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := decoder.Push(frame); err != nil {
			t.Fatalf("setup event %q was rejected: %v", eventType, err)
		}
	}
	markers := []map[string]any{
		{
			"type": "response.content_part.added", "sequence_number": 4,
			"item_id": "item_1", "output_index": 0, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}},
		},
		{
			"type": "response.output_text.delta", "sequence_number": 5,
			"item_id": "item_1", "output_index": 0, "content_index": 0, "delta": "done",
		},
		{
			"type": "response.output_text.done", "sequence_number": 6,
			"item_id": "item_1", "output_index": 0, "content_index": 0,
			"text": "done", "logprobs": []any{}, "obfuscation": "padding",
		},
		{
			"type": "response.content_part.done", "sequence_number": 7,
			"item_id": "item_1", "output_index": 0, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": "done", "annotations": []any{}},
		},
		{
			"type": "response.content_part.added", "sequence_number": 8,
			"item_id": "item_1", "output_index": 0, "content_index": 1,
			"part": map[string]any{"type": "refusal", "refusal": ""},
		},
		{
			"type": "response.refusal.delta", "sequence_number": 9,
			"item_id": "item_1", "output_index": 0, "content_index": 1, "delta": "cannot comply",
		},
		{
			"type": "response.refusal.done", "sequence_number": 10,
			"item_id": "item_1", "output_index": 0, "content_index": 1, "refusal": "cannot comply",
		},
		{
			"type": "response.content_part.done", "sequence_number": 11,
			"item_id": "item_1", "output_index": 0, "content_index": 1,
			"part": map[string]any{"type": "refusal", "refusal": "cannot comply"},
		},
		{
			"type": "response.function_call_arguments.delta", "sequence_number": 12,
			"item_id": "item_2", "output_index": 1, "delta": `{"q":"weather"}`,
		},
		{
			"type": "response.function_call_arguments.done", "sequence_number": 13,
			"item_id": "item_2", "output_index": 1, "name": "lookup", "arguments": `{"q":"weather"}`,
		},
		{
			"type": "response.reasoning_summary_part.added", "sequence_number": 14,
			"item_id": "item_3", "output_index": 2, "summary_index": 0,
			"part": map[string]any{"type": "summary_text", "text": ""},
		},
		{
			"type": "response.reasoning_summary_text.delta", "sequence_number": 15,
			"item_id": "item_3", "output_index": 2, "summary_index": 0, "delta": "summary",
		},
		{
			"type": "response.reasoning_summary_text.done", "sequence_number": 16,
			"item_id": "item_3", "output_index": 2, "summary_index": 0, "text": "summary",
		},
		{
			"type": "response.reasoning_summary_part.done", "sequence_number": 17,
			"item_id": "item_3", "output_index": 2, "summary_index": 0,
			"part": map[string]any{"type": "summary_text", "text": "summary"},
		},
		{
			"type": "response.reasoning_text.delta", "sequence_number": 18,
			"item_id": "item_3", "output_index": 2, "content_index": 0, "delta": "reasoning",
		},
		{
			"type": "response.reasoning_text.done", "sequence_number": 19,
			"item_id": "item_3", "output_index": 2, "content_index": 0, "text": "reasoning",
		},
	}
	var diagnostics llmprotocol.Diagnostics
	for _, marker := range markers {
		eventType, _ := marker["type"].(string)
		frame, err := encodeSSE(eventType, marker)
		if err != nil {
			t.Fatal(err)
		}
		_, frameDiagnostics, err := decoder.Push(frame)
		if err != nil {
			t.Fatalf("official event %q was rejected: %v", eventType, err)
		}
		diagnostics = append(diagnostics, frameDiagnostics...)
	}
	foundLogprobs := false
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "stream.logprobs" && diagnostic.Action == llmprotocol.DiagnosticDropped {
			foundLogprobs = true
		}
	}
	if !foundLogprobs {
		t.Fatalf("stream logprobs omission was not explicit: %+v", diagnostics)
	}
}

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

	unsupported := []struct {
		name  string
		event map[string]any
		code  string
	}{
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
	for _, test := range unsupported {
		t.Run(test.name, func(t *testing.T) {
			eventType, _ := test.event["type"].(string)
			frame, err := encodeSSE(eventType, test.event)
			if err != nil {
				t.Fatal(err)
			}
			fresh := AnthropicMessagesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy(),
			)
			_, _, err = fresh.Push(frame)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature || protocolError.Code != test.code {
				t.Fatalf("returned %T %v, want unsupported_feature/%s", err, err, test.code)
			}
		})
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
