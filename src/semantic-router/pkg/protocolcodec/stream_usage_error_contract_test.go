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
		t.Run(string(target), func(t *testing.T) {
			stream, err := engine.NewStream(
				llmprotocol.AnthropicMessagesV1,
				target,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
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

			var encoded bytes.Buffer
			writeFrames(&encoded, frames)
			verify := mustNewMatrixStream(t, engine, target, target)
			_, targetEvents, _, err := verify.Push(encoded.Bytes())
			if err != nil {
				t.Fatalf("target stream decode failed: %v\n%s", err, encoded.Bytes())
			}
			_, targetFinalEvents, _, err := verify.Finalize(nil)
			if err != nil {
				t.Fatalf("target stream finalize failed: %v\n%s", err, encoded.Bytes())
			}
			targetEvents = append(targetEvents, targetFinalEvents...)
			assertAnthropicCacheAccounting(t, targetEvents)
		})
	}
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

func TestStreamErrorTranslationMatrixPreservesNeutralError(t *testing.T) {
	fixtures := map[llmprotocol.WireFormat][]byte{
		llmprotocol.OpenAIChatV1: []byte(
			"data: {\"error\":{\"message\":\"API key is invalid.\",\"type\":\"authentication_error\",\"param\":\"model\",\"code\":\"authentication_error\"}}\n\n",
		),
		llmprotocol.OpenAIResponsesV1: []byte(
			"event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":0,\"response\":{\"id\":\"response_1\",\"object\":\"response\",\"model\":\"provider-model\",\"status\":\"failed\",\"error\":{\"code\":\"authentication_error\",\"message\":\"API key is invalid.\"}}}\n\n",
		),
		llmprotocol.AnthropicMessagesV1: []byte(
			"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"API key is invalid.\"}}\n\n",
		),
	}
	failureScopes := map[llmprotocol.WireFormat]llmprotocol.FailureScope{
		llmprotocol.OpenAIChatV1:        llmprotocol.FailureTransport,
		llmprotocol.OpenAIResponsesV1:   llmprotocol.FailureResponse,
		llmprotocol.AnthropicMessagesV1: llmprotocol.FailureTransport,
	}
	parameters := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: "model",
	}
	formats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}
	engine := NewBuiltinEngine()
	for _, source := range formats {
		for _, target := range formats {
			t.Run(string(source)+"/"+string(target), func(t *testing.T) {
				stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model",
				})
				if err != nil {
					t.Fatal(err)
				}
				frames, events, _, err := stream.Push(fixtures[source])
				if err != nil {
					t.Fatal(err)
				}
				assertAuthenticationFailureEvent(t, events, failureScopes[source], parameters[source])
				assertPublicStreamErrorWire(
					t, target, frames, failureScopes[source], "authentication_error",
					"API key is invalid.", parameters[source],
				)
			})
		}
	}
}

func TestResponsesTopLevelErrorEventUsesTransportScope(t *testing.T) {
	fixture := []byte(
		"event: error\ndata: {\"type\":\"error\",\"code\":\"authentication_error\",\"message\":\"API key is invalid.\",\"param\":\"model\",\"sequence_number\":0}\n\n",
	)
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			stream, err := engine.NewStream(
				llmprotocol.OpenAIResponsesV1,
				target,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
			)
			if err != nil {
				t.Fatal(err)
			}
			frames, events, _, err := stream.Push(fixture)
			if err != nil {
				t.Fatal(err)
			}
			assertAuthenticationFailureEvent(t, events, llmprotocol.FailureTransport, "model")
			assertPublicStreamErrorWire(
				t, target, frames, llmprotocol.FailureTransport, "authentication_error",
				"API key is invalid.", "model",
			)
		})
	}
}

func TestIncompleteUpstreamStreamTerminatesEveryTargetWithFailure(t *testing.T) {
	partial := map[llmprotocol.WireFormat][]byte{
		llmprotocol.OpenAIChatV1: []byte(
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
		),
		llmprotocol.OpenAIResponsesV1: []byte(
			"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"provider-model\",\"status\":\"in_progress\"}}\n\n" +
				"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\"}}\n\n" +
				"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"partial\"}\n\n",
		),
		llmprotocol.AnthropicMessagesV1: []byte(
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
		),
	}
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"/"+string(target), func(t *testing.T) {
				stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model",
				})
				if err != nil {
					t.Fatal(err)
				}
				if _, events, _, err := stream.Push(partial[source]); err != nil {
					t.Fatalf("partial stream rejected before transport ended: %v", err)
				} else if len(events) == 0 || events[len(events)-1].Type == llmprotocol.EventResponseFailed {
					t.Fatalf("partial stream terminal state = %+v", events)
				}
				frames, events, _, err := stream.Finalize(nil)
				if err != nil {
					t.Fatal(err)
				}
				if len(events) != 1 || events[0].Type != llmprotocol.EventResponseFailed ||
					events[0].Error == nil || events[0].Error.Code != "stream_incomplete" {
					t.Fatalf("incomplete stream terminal = %+v", events)
				}
				wireCode := "stream_incomplete"
				if target == llmprotocol.AnthropicMessagesV1 {
					wireCode = "api_error"
				}
				assertPublicStreamErrorWire(
					t, target, frames, llmprotocol.FailureTransport,
					wireCode, "upstream stream ended before completion", "",
				)
			})
		}
	}
}

func TestChatUsageOnEveryChunkNeverTerminatesTranslationEarly(t *testing.T) {
	payload := []byte(
		"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"first \"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n" +
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n" +
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n" +
			"data: [DONE]\n\n",
	)
	engine := NewBuiltinEngine()
	for _, target := range builtinFormats {
		t.Run(string(target), func(t *testing.T) {
			stream, err := engine.NewStream(
				llmprotocol.OpenAIChatV1,
				target,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
			)
			if err != nil {
				t.Fatal(err)
			}
			_, events, _, err := stream.Push(payload)
			if err != nil {
				t.Fatal(err)
			}
			_, finalEvents, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, finalEvents...)
			text, terminal := "", 0
			var finalUsage *llmprotocol.Usage
			for _, event := range events {
				if event.Type == llmprotocol.EventOutputTextDelta {
					text += event.Delta
				}
				if event.Type == llmprotocol.EventResponseCompleted {
					terminal++
					finalUsage = event.Usage
				}
			}
			if text != "first second" || terminal != 1 || finalUsage == nil ||
				finalUsage.Total.Value == nil || *finalUsage.Total.Value != 5 {
				t.Fatalf("translated stream text=%q terminals=%d usage=%+v events=%+v", text, terminal, finalUsage, events)
			}
		})
	}
}

func TestResponsesQueuedEventStartsOneSemanticStream(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte("event: response.queued\ndata: {\"type\":\"response.queued\",\"sequence_number\":0,\"response\":{\"id\":\"resp_queued\",\"object\":\"response\",\"status\":\"queued\",\"model\":\"provider-model\",\"output\":[]}}\n\n")
	_, events, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Type != llmprotocol.EventResponseStarted || events[0].ResponseID != "resp_queued" {
		t.Fatalf("queued events = %+v", events)
	}
}

func TestResponsesStreamOutputItemUnionIsDiscriminatedBeforeVariantDecode(t *testing.T) {
	tests := []struct {
		name     string
		item     string
		category llmprotocol.ErrorCategory
		code     string
	}{
		{
			name:     "unsupported official item keeps typed capability error",
			item:     `{"type":"web_search_call","id":"search_1","variant_specific_field":true}`,
			category: llmprotocol.ErrorUnsupportedFeature,
			code:     "unsupported_output_item",
		},
		{
			name:     "supported item rejects another variant field",
			item:     `{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[],"arguments":"{}"}`,
			category: llmprotocol.ErrorUpstreamUnavailable,
			code:     "invalid_response_item",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decoder := OpenAIResponsesCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
				llmprotocol.DefaultPolicy(),
			)
			payload := []byte("event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":0,\"output_index\":0,\"item\":" + test.item + "}\n\n")
			_, _, err := decoder.Push(payload)
			assertProtocolError(t, err, test.category, test.code)
		})
	}
}

func assertAuthenticationFailureEvent(
	t *testing.T,
	events []llmprotocol.Event,
	failure llmprotocol.FailureScope,
	parameter string,
) {
	t.Helper()
	if len(events) != 1 || events[0].Type != llmprotocol.EventResponseFailed || events[0].Error == nil ||
		events[0].Failure != failure ||
		events[0].Error.Category != llmprotocol.ErrorAuthentication ||
		events[0].Error.Code != "authentication_error" ||
		events[0].Error.Message != "API key is invalid." || events[0].Error.Parameter != parameter {
		t.Fatalf("failure events = %#v", events)
	}
}

func assertPublicStreamErrorWire(
	t *testing.T,
	format llmprotocol.WireFormat,
	frames [][]byte,
	failure llmprotocol.FailureScope,
	code,
	message,
	parameter string,
) {
	t.Helper()
	if len(frames) != 1 {
		t.Fatalf("public error frame count = %d: %q", len(frames), frames)
	}
	quotedCode, quotedMessage := string(mustJSON(code)), string(mustJSON(message))
	quotedParameter := "null"
	if parameter != "" {
		quotedParameter = string(mustJSON(parameter))
	}
	var golden string
	switch format {
	case llmprotocol.OpenAIChatV1:
		golden = "data: {\"error\":{\"type\":\"authentication_error\",\"code\":" + quotedCode +
			",\"message\":" + quotedMessage + ",\"param\":" + quotedParameter + "}}\n\n"
	case llmprotocol.OpenAIResponsesV1:
		if failure == llmprotocol.FailureResponse {
			golden = "event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":0," +
				"\"response\":{\"id\":\"response_1\",\"object\":\"response\",\"model\":\"public-model\"," +
				"\"status\":\"failed\",\"error\":{\"code\":" + quotedCode + ",\"message\":" + quotedMessage + "}}}\n\n"
		} else {
			golden = "event: error\ndata: {\"type\":\"error\",\"code\":" + quotedCode +
				",\"message\":" + quotedMessage + ",\"param\":" + quotedParameter + ",\"sequence_number\":0}\n\n"
		}
	case llmprotocol.AnthropicMessagesV1:
		golden = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":" + quotedCode +
			",\"message\":" + quotedMessage + "}}\n\n"
	}
	if string(frames[0]) != golden {
		t.Fatalf("public error frame = %q, want %q", frames[0], golden)
	}
	parsed, err := parseSSEFrame(frames[0], llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
	if err != nil {
		t.Fatalf("public error frame is invalid: %v: %q", err, frames[0])
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(parsed.Data, &object); err != nil {
		t.Fatalf("public error payload is invalid JSON: %v: %s", err, parsed.Data)
	}
	assertPublicStreamErrorPayload(t, format, failure, parsed, object, code, message, parameter)
}

func assertPublicStreamErrorPayload(
	t *testing.T,
	format llmprotocol.WireFormat,
	failure llmprotocol.FailureScope,
	parsed sseFrame,
	object map[string]json.RawMessage,
	code,
	message,
	parameter string,
) {
	t.Helper()
	switch format {
	case llmprotocol.OpenAIChatV1:
		assertChatStreamErrorPayload(t, parsed, object, code, message, parameter)
	case llmprotocol.OpenAIResponsesV1:
		assertResponsesStreamErrorPayload(t, failure, parsed, object, code, message, parameter)
	case llmprotocol.AnthropicMessagesV1:
		assertAnthropicStreamErrorPayload(t, parsed, object, code, message)
	default:
		t.Fatalf("unexpected target format %q", format)
	}
}

func assertChatStreamErrorPayload(t *testing.T, parsed sseFrame, object map[string]json.RawMessage, code, message, parameter string) {
	t.Helper()
	if parsed.Event != "" || len(object) != 1 || object["error"] == nil {
		t.Fatalf("Chat stream error is not canonical: event=%q data=%s", parsed.Event, parsed.Data)
	}
	assertOpenAIErrorDetail(t, object["error"], code, message, parameter)
}

func assertResponsesStreamErrorPayload(
	t *testing.T,
	failure llmprotocol.FailureScope,
	parsed sseFrame,
	object map[string]json.RawMessage,
	code,
	message,
	parameter string,
) {
	t.Helper()
	if failure == llmprotocol.FailureResponse {
		assertResponsesFailedResource(t, parsed, object, code, message)
		return
	}
	if parsed.Event != "error" || len(object) != 5 || string(object["type"]) != `"error"` ||
		object["response"] != nil || object["error"] != nil {
		t.Fatalf("Responses transport error is not canonical: event=%q data=%s", parsed.Event, parsed.Data)
	}
	assertResponsesTopLevelError(t, object, code, message, parameter)
}

func assertResponsesFailedResource(
	t *testing.T,
	parsed sseFrame,
	object map[string]json.RawMessage,
	code,
	message string,
) {
	t.Helper()
	assertResponsesFailedEnvelope(t, parsed, object)
	assertResponsesFailedDetail(t, parsed, object["response"], code, message)
}

func assertResponsesFailedEnvelope(t *testing.T, parsed sseFrame, object map[string]json.RawMessage) {
	t.Helper()
	if parsed.Event != "response.failed" || len(object) != 3 ||
		string(object["type"]) != `"response.failed"` || object["response"] == nil || object["error"] != nil {
		t.Fatalf("Responses failed event is not canonical: event=%q data=%s", parsed.Event, parsed.Data)
	}
}

func assertResponsesFailedDetail(t *testing.T, parsed sseFrame, raw json.RawMessage, code, message string) {
	t.Helper()
	var responseObject map[string]json.RawMessage
	if err := json.Unmarshal(raw, &responseObject); err != nil || len(responseObject) != 5 {
		t.Fatalf("Responses failed resource fields are not canonical: %v data=%s", err, parsed.Data)
	}
	var response struct {
		ID     string `json:"id"`
		Object string `json:"object"`
		Model  string `json:"model"`
		Status string `json:"status"`
		Error  struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	err := json.Unmarshal(raw, &response)
	if err != nil || response.ID != "response_1" || response.Object != "response" ||
		response.Model != "public-model" || response.Status != "failed" ||
		response.Error.Code != code || response.Error.Message != message {
		t.Fatalf("Responses failed resource is not canonical: %+v/%v data=%s", response, err, parsed.Data)
	}
}

func assertAnthropicStreamErrorPayload(t *testing.T, parsed sseFrame, object map[string]json.RawMessage, code, message string) {
	t.Helper()
	if parsed.Event != "error" || len(object) != 2 || string(object["type"]) != `"error"` || object["error"] == nil {
		t.Fatalf("Anthropic stream error is not canonical: event=%q data=%s", parsed.Event, parsed.Data)
	}
	var detail struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(object["error"], &detail); err != nil || detail.Type != code || detail.Message != message {
		t.Fatalf("Anthropic stream error detail is not canonical: %+v/%v data=%s", detail, err, parsed.Data)
	}
}

func assertOpenAIErrorDetail(t *testing.T, raw json.RawMessage, code, message, parameter string) {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil || len(object) != 4 {
		t.Fatalf("OpenAI error detail fields are not canonical: %v raw=%s", err, raw)
	}
	var detail struct {
		Type    string  `json:"type"`
		Code    *string `json:"code"`
		Message string  `json:"message"`
		Param   *string `json:"param"`
	}
	err := json.Unmarshal(raw, &detail)
	if err != nil || detail.Type != "authentication_error" || detail.Code == nil ||
		*detail.Code != code || detail.Message != message || !optionalStringMatches(detail.Param, parameter) {
		t.Fatalf("OpenAI error detail is not canonical: %+v raw=%s", detail, raw)
	}
}

func optionalStringMatches(actual *string, expected string) bool {
	if expected == "" {
		return actual == nil
	}
	return actual != nil && *actual == expected
}

func assertResponsesTopLevelError(
	t *testing.T,
	object map[string]json.RawMessage,
	code,
	message,
	parameter string,
) {
	t.Helper()
	var codeValue, messageValue string
	var parameterValue *string
	if err := json.Unmarshal(object["code"], &codeValue); err != nil {
		t.Fatalf("Responses error code is invalid: %v", err)
	}
	if err := json.Unmarshal(object["message"], &messageValue); err != nil {
		t.Fatalf("Responses error message is invalid: %v", err)
	}
	if err := json.Unmarshal(object["param"], &parameterValue); err != nil {
		t.Fatalf("Responses error parameter is invalid: %v", err)
	}
	if codeValue != code || messageValue != message ||
		(parameter == "" && parameterValue != nil) ||
		(parameter != "" && (parameterValue == nil || *parameterValue != parameter)) {
		t.Fatalf("Responses top-level error is not canonical: %s", mustJSON(object))
	}
}

func mustJSON(value any) []byte {
	body, _ := json.Marshal(value)
	return body
}
