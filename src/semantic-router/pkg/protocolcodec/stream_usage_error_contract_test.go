package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
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

func TestOpenAIChatStreamRetainsOptionalBreakdownAcrossTotalsOnlyUsage(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
		llmprotocol.DefaultPolicy(),
	)
	payload := []byte(
		"data: {\"id\":\"response_1\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"response_1\",\"model\":\"model\",\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":4,\"total_tokens\":24,\"prompt_tokens_details\":{\"cached_tokens\":7},\"completion_tokens_details\":{\"reasoning_tokens\":1}}}\n\n" +
			"data: {\"id\":\"response_1\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: {\"id\":\"response_1\",\"model\":\"model\",\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":4,\"total_tokens\":24}}\n\n" +
			"data: [DONE]\n\n",
	)
	events, _, err := decoder.Push(payload)
	if err != nil {
		t.Fatalf("Push() error = %v", err)
	}
	terminal := events[len(events)-1]
	if terminal.Type != llmprotocol.EventResponseCompleted || terminal.Usage == nil ||
		terminal.Usage.InputCacheRead.Value == nil || *terminal.Usage.InputCacheRead.Value != 7 ||
		terminal.Usage.OutputReasoning.Value == nil || *terminal.Usage.OutputReasoning.Value != 1 {
		t.Fatalf("terminal usage = %#v", terminal.Usage)
	}
}

func TestOfficialChatStreamObfuscationAndModerationAreAccepted(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
		llmprotocol.DefaultPolicy(),
	)
	payload := []byte(
		"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model\",\"obfuscation\":\"padding\",\"moderation\":{\"status\":\"passed\"},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null,\"logprobs\":null}]}\n\n",
	)
	events, diagnostics, err := decoder.Push(payload)
	if err != nil {
		t.Fatalf("official Chat stream fields were rejected: %v", err)
	}
	if len(events) < 2 {
		t.Fatalf("Chat stream events = %+v", events)
	}
	if len(diagnostics) != 1 || diagnostics[0].Field != "stream.moderation" || diagnostics[0].Action != llmprotocol.DiagnosticDropped {
		t.Fatalf("moderation omission was not explicit: %+v", diagnostics)
	}
}

func TestChatStreamAcceptsBothReasoningAliases(t *testing.T) {
	for _, field := range []string{"reasoning_content", "reasoning"} {
		t.Run(field, func(t *testing.T) {
			decoder := OpenAIChatCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
				llmprotocol.DefaultPolicy(),
			)
			payload := []byte(`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"model","choices":[{"index":0,"delta":{"role":"assistant","` + field + `":"inspect the evidence"},"finish_reason":null,"logprobs":null}]}` + "\n\n")
			events, _, err := decoder.Push(payload)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, event := range events {
				if event.Type == llmprotocol.EventReasoningDelta && event.Delta == "inspect the evidence" {
					found = true
				}
			}
			if !found {
				t.Fatalf("reasoning alias %q was not decoded: %+v", field, events)
			}
		})
	}
}

func TestAnthropicStreamCacheAccountingSurvivesEveryTargetFormat(t *testing.T) {
	payload := []byte(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":13,\"output_tokens\":0,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":8502}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":13,\"output_tokens\":4,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":8502}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	engine := NewBuiltinEngine()
	for _, target := range builtinFormats {
		t.Run(string(target), func(t *testing.T) { assertAnthropicCacheTranslation(t, engine, target, payload) })
	}
}

func assertAnthropicCacheTranslation(t *testing.T, engine *Engine, target llmprotocol.WireFormat, payload []byte) {
	t.Helper()
	stream, err := engine.NewStream(
		llmprotocol.AnthropicMessagesV1,
		target,
		llmprotocol.StreamContext{
			Context: context.Background(), PublicModel: "public-model",
			Options: llmprotocol.StreamOptions{IncludeUsage: boolPointer(true)},
		},
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
	assertAnthropicCacheAccounting(t, events)
	assertEncodedAnthropicCacheAccounting(t, engine, target, frames)
}

func assertEncodedAnthropicCacheAccounting(t *testing.T, engine *Engine, target llmprotocol.WireFormat, frames [][]byte) {
	t.Helper()
	var encoded bytes.Buffer
	writeFrames(&encoded, frames)
	verify := mustNewMatrixStream(t, engine, target, target)
	_, events, _, err := verify.Push(encoded.Bytes())
	if err != nil {
		t.Fatalf("target stream decode failed: %v\n%s", err, encoded.Bytes())
	}
	_, finalEvents, _, err := verify.Finalize(nil)
	if err != nil {
		t.Fatalf("target stream finalize failed: %v\n%s", err, encoded.Bytes())
	}
	assertAnthropicCacheAccounting(t, append(events, finalEvents...))
}

func TestAnthropicMessageDeltaEncodingUsesOfficialVariantShape(t *testing.T) {
	encoder := AnthropicMessagesCodec{}.NewEncoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model", ResponseID: "response_1"},
		llmprotocol.DefaultPolicy(),
	)
	usage := availableStreamUsage(2, 1)
	events := []llmprotocol.Event{
		{Type: llmprotocol.EventResponseStarted, ResponseID: "response_1", Model: "model"},
		{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "item_1", Role: llmprotocol.RoleAssistant},
		{Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, ItemID: "item_1", Delta: "ok"},
		{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0, ItemID: "item_1"},
		{Type: llmprotocol.EventResponseCompleted, ResponseID: "response_1", Model: "model", StopReason: llmprotocol.StopEndTurn, Usage: &usage},
	}
	var output []byte
	for _, event := range events {
		frames, _, err := encoder.Push(event)
		if err != nil {
			t.Fatal(err)
		}
		output = append(output, bytes.Join(frames, nil)...)
	}
	frames := bytes.Split(output, []byte("\n\n"))
	found := false
	for _, frame := range frames {
		if !bytes.Contains(frame, []byte(`"type":"message_delta"`)) {
			continue
		}
		found = true
		parsed, err := parseSSEFrame(append(frame, '\n', '\n'), llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
		if err != nil {
			t.Fatal(err)
		}
		var event struct {
			Delta map[string]json.RawMessage `json:"delta"`
		}
		if err := json.Unmarshal(parsed.Data, &event); err != nil {
			t.Fatal(err)
		}
		if _, leaked := event.Delta["type"]; leaked {
			t.Fatalf("message_delta leaked a content-delta discriminator: %s", parsed.Data)
		}
		if stop, present := event.Delta["stop_sequence"]; !present || string(stop) != "null" {
			t.Fatalf("message_delta stop_sequence = %s present=%v, want explicit null", stop, present)
		}
	}
	if !found {
		t.Fatalf("message_delta was not emitted: %s", output)
	}
}

func TestAnthropicStreamPreservesMatchedStopSequence(t *testing.T) {
	payload := []byte(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"stop_sequence\",\"stop_sequence\":\"END\"},\"usage\":{\"output_tokens\":1}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	engine := NewBuiltinEngine()
	response, _, err := engine.DecodeResponseStream(
		llmprotocol.AnthropicMessagesV1,
		payload,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.StopReason != llmprotocol.StopSequence || response.MatchedStopSequence != "END" {
		t.Fatalf("decoded terminal = %q matched=%q", response.StopReason, response.MatchedStopSequence)
	}
	stream, err := engine.NewStream(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	finalFrames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	frames = append(frames, finalFrames...)
	output := bytes.Join(frames, nil)
	if !bytes.Contains(output, []byte(`"stop_reason":"stop_sequence"`)) ||
		!bytes.Contains(output, []byte(`"stop_sequence":"END"`)) {
		t.Fatalf("translated stream lost matched stop sequence: %s", output)
	}
}

func assertAnthropicCacheAccounting(t *testing.T, events []llmprotocol.Event) {
	t.Helper()
	if len(events) == 0 {
		t.Fatal("stream produced no events")
	}
	terminal := events[len(events)-1]
	if terminal.Type != llmprotocol.EventResponseCompleted || terminal.Usage == nil ||
		tokenValue(terminal.Usage.InputUncached) != 13 ||
		tokenValue(terminal.Usage.InputCacheRead) != 8502 ||
		tokenValue(terminal.Usage.InputCacheWrite) != 0 ||
		tokenValue(terminal.Usage.InputTotal) != 8515 ||
		tokenValue(terminal.Usage.OutputTotal) != 4 ||
		tokenValue(terminal.Usage.Total) != 8519 {
		t.Fatalf("terminal cache usage = %+v", terminal.Usage)
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
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventResponseFailed, StopReason: llmprotocol.StopError,
		Error:   llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "failed", "failed", nil),
		Failure: llmprotocol.FailureScope("unknown"),
	}); err == nil {
		t.Fatal("failed terminal event with an invalid failure scope was accepted")
	}

	state = newTestStreamState()
	startTestStream(t, state)
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted}); err == nil {
		t.Fatal("successful terminal event without output was accepted")
	}
}

func TestStreamFailurePreservesValidUsageAndRejectsInvalidUsage(t *testing.T) {
	state := newTestStreamState()
	usage := availableStreamUsage(4, 2)
	terminal, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventResponseFailed, StopReason: llmprotocol.StopError,
		Error: llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "failed", "failed", nil),
		Usage: &usage,
	})
	if err != nil {
		t.Fatalf("failed response usage rejected: %v", err)
	}
	if terminal.Usage == nil || tokenValue(terminal.Usage.Total) != 6 {
		t.Fatalf("failed response usage = %+v", terminal.Usage)
	}

	invalid := usage
	invalid.Total.Value = llmprotocol.Int64(-1)
	state = newTestStreamState()
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventResponseFailed, StopReason: llmprotocol.StopError,
		Error: llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "failed", "failed", nil),
		Usage: &invalid,
	}); err == nil {
		t.Fatal("failed response with invalid usage was accepted")
	}
}

func TestAnthropicStreamErrorUsesCategoryWhenCodeIsEmpty(t *testing.T) {
	encoder := AnthropicMessagesCodec{}.NewEncoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
		llmprotocol.DefaultPolicy(),
	)
	frames, _, err := encoder.Push(llmprotocol.Event{
		Type:       llmprotocol.EventResponseFailed,
		StopReason: llmprotocol.StopError,
		Error: &llmprotocol.ProtocolError{
			Category: llmprotocol.ErrorAuthentication,
			Message:  "authentication failed",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) != 1 || !bytes.Contains(frames[0], []byte(`"type":"authentication_error"`)) {
		t.Fatalf("Anthropic error frame = %q", frames)
	}
}

func TestAnthropicStreamFinalizeUsesCanonicalErrorType(t *testing.T) {
	encoder := AnthropicMessagesCodec{}.NewEncoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
		llmprotocol.DefaultPolicy(),
	)
	frames, _, err := encoder.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	golden := "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"stream ended before completion\"}}\n\n"
	if len(frames) != 1 || string(frames[0]) != golden {
		t.Fatalf("Anthropic final error = %q, want %q", frames, golden)
	}
}
