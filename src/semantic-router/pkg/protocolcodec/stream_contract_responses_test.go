package protocolcodec

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var officialSupportedResponsesStreamEvents = fields(
	"error",
	"response.completed", "response.content_part.added", "response.content_part.done", "response.created",
	"response.failed", "response.function_call_arguments.delta", "response.function_call_arguments.done",
	"response.in_progress", "response.incomplete", "response.output_item.added", "response.output_item.done",
	"response.image_generation_call.completed", "response.image_generation_call.generating",
	"response.image_generation_call.in_progress", "response.image_generation_call.partial_image",
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
	for _, test := range providerRequiredFieldCases() {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := newProviderStreamDecoder(test.format).Push([]byte(test.frame))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

type providerRequiredFieldCase struct {
	name   string
	format llmprotocol.WireFormat
	frame  string
	code   string
}

func providerRequiredFieldCases() []providerRequiredFieldCase {
	responses, anthropic := llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1
	return []providerRequiredFieldCase{
		{"responses lifecycle resource", responses, "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0}\n\n", "stream_response_required"},
		{"responses output item", responses, "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":0,\"output_index\":0}\n\n", "stream_item_required"},
		{"responses delta target", responses, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"output_index\":0,\"delta\":\"orphan\"}\n\n", "stream_delta_target_required"},
		{"responses negative output index", responses, "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":0,\"output_index\":-1,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n", "invalid_stream_item_index"},
		{"responses missing content index", responses, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"output_index\":0,\"item_id\":\"msg_1\",\"delta\":\"x\"}\n\n", "stream_delta_target_required"},
		{"responses missing required delta member", responses, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\"}\n\n", "stream_required_field"},
		{"anthropic start message", anthropic, "event: message_start\ndata: {\"type\":\"message_start\"}\n\n", "stream_message_required"},
		{"anthropic content block", anthropic, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0}\n\n", "stream_content_block_required"},
		{"anthropic missing content index", anthropic, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n", "invalid_stream_item_index"},
		{"anthropic negative content index", anthropic, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":-1,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n", "invalid_stream_item_index"},
		{"anthropic message delta", anthropic, "event: message_delta\ndata: {\"type\":\"message_delta\"}\n\n", "stream_message_delta_required"},
		{"anthropic message delta usage", anthropic, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}}\n\n", "stream_message_usage_required"},
		{"anthropic content delta member", anthropic, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\"}}\n\n", "stream_required_field"},
		{"anthropic error payload", anthropic, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"\",\"message\":\"failed\"}}\n\n", "invalid_anthropic_stream_error"},
		{"anthropic stop sequence with wrong reason", anthropic, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":\"END\"},\"usage\":{\"output_tokens\":1}}\n\n", "anthropic_stop_sequence_reason"},
		{"anthropic empty stop sequence with no reason", anthropic, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":null,\"stop_sequence\":\"\"},\"usage\":{\"output_tokens\":1}}\n\n", "anthropic_stop_sequence_reason"},
	}
}

func newProviderStreamDecoder(format llmprotocol.WireFormat) llmprotocol.StreamDecoder {
	streamContext := llmprotocol.StreamContext{Context: context.Background()}
	if format == llmprotocol.AnthropicMessagesV1 {
		return AnthropicMessagesCodec{}.NewDecoder(streamContext, llmprotocol.DefaultPolicy())
	}
	return OpenAIResponsesCodec{}.NewDecoder(streamContext, llmprotocol.DefaultPolicy())
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
	markers := officialResponsesStreamMarkers
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

var officialResponsesStreamMarkers = []map[string]any{
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
