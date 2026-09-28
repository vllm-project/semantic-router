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

func TestSSEAndSemanticEventLimitsAreInclusive(t *testing.T) {
	frame := []byte("data: x\n\n")
	framer := newSSEFramer(len(frame))
	frames, err := framer.Push(frame)
	if err != nil || len(frames) != 1 {
		t.Fatalf("SSE frame exactly at limit = %q, %v", frames, err)
	}
	framer = newSSEFramer(len(frame) - 1)
	_, err = framer.Push(frame)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit")

	policy := llmprotocol.DefaultPolicy()
	policy.Limits.Events = 2
	state := newStartedStreamState(t, policy)
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventProviderOpaque}); err != nil {
		t.Fatalf("event exactly at limit was rejected: %v", err)
	}
	_, err = state.next(llmprotocol.Event{Type: llmprotocol.EventProviderOpaque})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_event_limit")
}

func TestUTF8BOMIsAcceptedOnlyAtTheStartOfAnSSEStream(t *testing.T) {
	bom := []byte{0xef, 0xbb, 0xbf}
	for _, source := range builtinFormats {
		payload := streamFixture(source)
		boundary := bytes.Index(payload, []byte("\n\n"))
		if boundary < 0 {
			t.Fatalf("%s fixture has no first SSE frame", source)
		}
		boundary += 2
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
				stream, err := NewBuiltinEngine().NewStream(source, target, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
				})
				if err != nil {
					t.Fatal(err)
				}
				if _, _, _, err := stream.Push(append(append([]byte(nil), bom...), payload[:boundary]...)); err != nil {
					t.Fatalf("leading UTF-8 BOM was rejected: %v", err)
				}
				late := append(append([]byte(nil), bom...), payload[boundary:]...)
				_, _, _, err = stream.Push(late)
				assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "unexpected_stream_bom")
			})
		}
	}
}

func TestStreamWireIndexesCompactSparseNeutralIndexes(t *testing.T) {
	var indexes streamWireIndexes
	started := llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 7}
	if got := indexes.translate(started); got != 0 {
		t.Fatalf("first wire index = %d, want 0", got)
	}
	delta := llmprotocol.Event{Type: llmprotocol.EventToolCallDelta, ItemIndex: 7}
	if got := indexes.translate(delta); got != 0 {
		t.Fatalf("delta wire index = %d, want 0", got)
	}
	if got := indexes.translate(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 2}); got != 1 {
		t.Fatalf("second wire index = %d, want 1", got)
	}
}

func TestReasoningDeltasShareTheBoundedTextBudget(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.TextBytes = 4
	state := streamState{
		context: llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
		policy:  policy,
	}
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseStarted, ResponseID: "response_1"}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "reasoning_1"}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventReasoningDelta, ItemIndex: 0, Delta: "1234"}); err != nil {
		t.Fatalf("reasoning at the text limit was rejected: %v", err)
	}
	_, err := state.next(llmprotocol.Event{Type: llmprotocol.EventReasoningDelta, ItemIndex: 0, Delta: "5"})
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "text_limit")
}

func TestStreamToolArgumentsAreBoundedAtEveryLifecycleStage(t *testing.T) {
	newState := func(unfinished, final int) streamState {
		policy := llmprotocol.DefaultPolicy()
		policy.Limits.UnfinishedArguments = unfinished
		policy.Limits.ToolArgumentsBytes = final
		return newStartedStreamState(t, policy)
	}
	start := func(arguments string) llmprotocol.Event {
		return llmprotocol.Event{
			Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "item_1",
			ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: arguments},
		}
	}

	state := newState(6, 64)
	_, err := state.next(start(`{"x":1}`))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "tool_arguments_limit")

	state = newState(7, 7)
	if _, err := state.next(start(`{"x":1}`)); err != nil {
		t.Fatalf("tool arguments exactly at both limits were rejected: %v", err)
	}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0, ItemID: "item_1",
	}); err != nil {
		t.Fatalf("tool arguments exactly at both limits could not complete: %v", err)
	}

	state = newState(6, 64)
	if _, err := state.next(start(`{}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventToolCallDelta, ItemIndex: 0,
		ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: `{"x"`},
	}); err != nil {
		t.Fatal(err)
	}
	_, err = state.next(llmprotocol.Event{
		Type: llmprotocol.EventToolCallDelta, ItemIndex: 0,
		ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: `:1}`},
	})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "tool_arguments_limit")

	state = newState(64, 6)
	_, err = state.next(start(`{"x":1}`))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "tool_arguments_limit")

	state = newState(6, 64)
	if _, err := state.next(start("")); err != nil {
		t.Fatal(err)
	}
	_, err = state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0, ItemID: "item_1",
		ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: `{"x":1}`},
	})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "tool_arguments_limit")

	state = newState(64, 6)
	if _, err := state.next(start("")); err != nil {
		t.Fatal(err)
	}
	_, err = state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0, ItemID: "item_1",
		ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: `{"x":1}`},
	})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "tool_arguments_limit")
}

func TestStreamIdentitiesAreBoundedAndUnique(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.IdentifierBytes = 8
	policy.Limits.ModelBytes = 8

	state := streamState{policy: policy}
	err := state.observeProviderIdentity("response-too-long", "model")
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_response_id_limit")
	state = streamState{policy: policy}
	err = state.observeProviderIdentity("response", "model-too-long")
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_model_limit")

	state = newStartedStreamState(t, policy)
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "item_1",
		Content: &llmprotocol.Content{Kind: llmprotocol.ContentText},
	}); err != nil {
		t.Fatal(err)
	}
	_, err = state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 1, ItemID: "item_1",
		Content: &llmprotocol.Content{Kind: llmprotocol.ContentText},
	})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "duplicate_stream_item_id")

	state = newStartedStreamState(t, policy)
	if _, err := state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "item_1",
		ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: "{}"},
	}); err != nil {
		t.Fatal(err)
	}
	_, err = state.next(llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ItemIndex: 1, ItemID: "item_2",
		ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: "{}"},
	})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "duplicate_stream_tool_call_id")
}

func newStartedStreamState(t *testing.T, policy llmprotocol.Policy) streamState {
	t.Helper()
	state := streamState{
		context: llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public"},
		policy:  policy,
	}
	if _, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseStarted, ResponseID: "response"}); err != nil {
		t.Fatal(err)
	}
	return state
}

func TestChatToolOnlyStreamDoesNotCreateEmptyTextItem(t *testing.T) {
	payload := []byte(
		"data: {\"id\":\"chatcmpl_tool\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n" +
			"data: [DONE]\n\n",
	)
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(target), func(t *testing.T) {
			assertChatToolOnlyStream(t, engine, target, payload)
		})
	}
}

func assertChatToolOnlyStream(t *testing.T, engine *Engine, target llmprotocol.WireFormat, payload []byte) {
	t.Helper()
	stream, err := engine.NewStream(
		llmprotocol.OpenAIChatV1,
		target,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	started := 0
	for _, event := range events {
		if event.Type == llmprotocol.EventOutputItemStarted {
			started++
			if event.ToolCall == nil {
				t.Fatalf("tool-only stream created a non-tool output: %+v", event)
			}
		}
	}
	if started != 1 {
		t.Fatalf("started items = %d, want 1", started)
	}
	output := bytes.Join(frames, nil)
	if !bytes.Contains(output, []byte(`"index":0`)) && !bytes.Contains(output, []byte(`"output_index":0`)) {
		t.Fatalf("target stream did not start at wire index zero: %s", output)
	}
}

func TestInvalidUTF8IsRejectedWithoutReplacement(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name           string
		format         llmprotocol.WireFormat
		requestPrefix  string
		requestSuffix  string
		responsePrefix string
		responseSuffix string
		streamPrefix   string
		streamSuffix   string
	}{
		{
			name: "chat", format: llmprotocol.OpenAIChatV1,
			requestPrefix: `{"model":"m","messages":[{"role":"user","content":"`, requestSuffix: `"}]}`,
			responsePrefix: `{"id":"r","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"`,
			responseSuffix: `"},"finish_reason":"stop"}]}`,
			streamPrefix:   "data: {\"id\":\"r\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"",
			streamSuffix:   "\"},\"finish_reason\":null}]}\n\n",
		},
		{
			name: "responses", format: llmprotocol.OpenAIResponsesV1,
			requestPrefix: `{"model":"m","input":"`, requestSuffix: `"}`,
			responsePrefix: `{"id":"r","object":"response","created_at":1,"model":"m","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"`,
			responseSuffix: `","annotations":[]}]}]}`,
			streamPrefix:   "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"",
			streamSuffix:   "\",\"logprobs\":[]}\n\n",
		},
		{
			name: "anthropic", format: llmprotocol.AnthropicMessagesV1,
			requestPrefix: `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"`, requestSuffix: `"}]}`,
			responsePrefix: `{"id":"r","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"`,
			responseSuffix: `"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`,
			streamPrefix:   "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"",
			streamSuffix:   "\"}}\n\n",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := invalidUTF8Between(test.requestPrefix, test.requestSuffix)
			_, _, _, err := engine.DecodeRequest(test.format, request)
			assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_utf8")

			response := invalidUTF8Between(test.responsePrefix, test.responseSuffix)
			_, _, _, err = engine.DecodeResponse(test.format, response)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_utf8")

			stream, err := engine.NewStream(
				test.format,
				test.format,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
			)
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = stream.Push(invalidUTF8Between(test.streamPrefix, test.streamSuffix))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_utf8")
		})
	}
}

func TestInvalidUTF8OutsideSSEDataIsRejectedAcrossProtocolMatrix(t *testing.T) {
	frames := map[llmprotocol.WireFormat][]byte{
		llmprotocol.OpenAIChatV1: []byte(
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
		),
		llmprotocol.OpenAIResponsesV1: []byte(
			"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"provider-model\",\"status\":\"in_progress\"}}\n\n",
		),
		llmprotocol.AnthropicMessagesV1: []byte(
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
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
				payload := append([]byte(": invalid-utf8-"), byte(0xff))
				payload = append(payload, '\n')
				payload = append(payload, frames[source]...)
				_, _, _, err = stream.Push(payload)
				assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_utf8")
			})
		}
	}
}

func TestEmptySSEDataIsNotSilentlyTreatedAsKeepaliveAcrossProtocolMatrix(t *testing.T) {
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
				frames, events, _, err := stream.Push([]byte(": keepalive\n\n"))
				if err != nil || len(frames) != 0 || len(events) != 0 {
					t.Fatalf("comment keepalive changed: frames=%q events=%+v err=%v", frames, events, err)
				}
				eventLine := ""
				if source == llmprotocol.OpenAIResponsesV1 {
					eventLine = "event: response.created\n"
				} else if source == llmprotocol.AnthropicMessagesV1 {
					eventLine = "event: message_start\n"
				}
				_, _, _, err = stream.Push([]byte(eventLine + "data:\n\n"))
				assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "upstream_body_limit")
			})
		}
	}
}

func invalidUTF8Between(prefix, suffix string) []byte {
	body := append([]byte(prefix), 0xff)
	return append(body, suffix...)
}

func TestUnicodeSurrogatePairsAndEscapedLiteralsRemainValid(t *testing.T) {
	engine := NewBuiltinEngine()
	valid := []byte(`{"model":"m","messages":[{"role":"user","content":"\ud83d\ude80"}]}`)
	if _, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, valid); err != nil {
		t.Fatalf("valid surrogate pair rejected: %v", err)
	}
	escapedLiteral := []byte(`{"model":"m","messages":[{"role":"user","content":"\\ud800"}]}`)
	if _, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, escapedLiteral); err != nil {
		t.Fatalf("escaped literal rejected: %v", err)
	}
}

func TestUnpairedUnicodeSurrogatesAreRejectedAcrossProtocolMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name     string
		format   llmprotocol.WireFormat
		request  string
		response string
		stream   string
	}{
		{
			name: "chat", format: llmprotocol.OpenAIChatV1,
			request:  `{"model":"m","messages":[{"role":"user","content":"\ud800"}]}`,
			response: `{"id":"r","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"\ud800"},"finish_reason":"stop"}]}`,
			stream:   "data: {\"id\":\"r\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\\ud800\"},\"finish_reason\":null}]}\n\n",
		},
		{
			name: "responses", format: llmprotocol.OpenAIResponsesV1,
			request:  `{"model":"m","input":"\ud800"}`,
			response: `{"id":"r","object":"response","created_at":1,"model":"m","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"\ud800","annotations":[]}]}]}`,
			stream:   "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"\\ud800\",\"logprobs\":[]}\n\n",
		},
		{
			name: "anthropic", format: llmprotocol.AnthropicMessagesV1,
			request:  `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"\ud800"}]}`,
			response: `{"id":"r","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"\ud800"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`,
			stream:   "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"\\ud800\"}}\n\n",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for name, surrogate := range map[string]string{"high": `\ud800`, "low": `\udc00`} {
				t.Run(name, func(t *testing.T) {
					request := strings.ReplaceAll(test.request, `\ud800`, surrogate)
					_, _, _, err := engine.DecodeRequest(test.format, []byte(request))
					assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_unicode")
					response := strings.ReplaceAll(test.response, `\ud800`, surrogate)
					_, _, _, err = engine.DecodeResponse(test.format, []byte(response))
					assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_unicode")

					stream, err := engine.NewStream(
						test.format,
						test.format,
						llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
					)
					if err != nil {
						t.Fatal(err)
					}
					wire := strings.ReplaceAll(test.stream, `\ud800`, surrogate)
					_, _, _, err = stream.Push([]byte(wire))
					assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_unicode")
				})
			}
		})
	}
}

func TestCaseFoldedDuplicateFieldsAreRejectedAcrossProtocolMatrix(t *testing.T) {
	requests := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:      `{"model":"m","Model":"shadow","messages":[{"role":"user","content":"hello"}]}`,
		llmprotocol.OpenAIResponsesV1: `{"model":"m","Model":"shadow","input":"hello"}`,
		llmprotocol.AnthropicMessagesV1: `{"model":"m","Model":"shadow","max_tokens":16,` +
			`"messages":[{"role":"user","content":"hello"}]}`,
	}
	responses := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: `{"id":"chat_1","object":"chat.completion","model":"m","Model":"shadow",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}`,
		llmprotocol.OpenAIResponsesV1: `{"id":"resp_1","object":"response","model":"m","Model":"shadow",` +
			`"status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant",` +
			`"status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}]}`,
		llmprotocol.AnthropicMessagesV1: `{"id":"msg_1","type":"message","role":"assistant",` +
			`"model":"m","Model":"shadow","content":[{"type":"text","text":"done"}],` +
			`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`,
	}
	streams := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: "data: {\"id\":\"chat_1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\"," +
			"\"Model\":\"shadow\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
		llmprotocol.OpenAIResponsesV1: "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0," +
			"\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"m\",\"Model\":\"shadow\"," +
			"\"status\":\"in_progress\"}}\n\n",
		llmprotocol.AnthropicMessagesV1: "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{" +
			"\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"Model\":\"shadow\"," +
			"\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
	}

	engine := NewBuiltinEngine()
	for _, format := range builtinFormats {
		t.Run(string(format), func(t *testing.T) {
			_, _, _, err := engine.DecodeRequest(format, []byte(requests[format]))
			assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "duplicate_json_field")

			_, _, _, err = engine.DecodeResponse(format, []byte(responses[format]))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "upstream_duplicate_json_field")

			stream, err := engine.NewStream(
				format,
				format,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
			)
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = stream.Push([]byte(streams[format]))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "upstream_duplicate_json_field")
		})
	}
}

func TestMatchedStopSequenceCannotSilentlyCrossWireFormats(t *testing.T) {
	engine := NewBuiltinEngine()
	response := llmprotocol.Response{
		Generation: 1, ID: "response_1", Model: "public-model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_1", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "done"}},
		}},
		StopReason: llmprotocol.StopSequence, MatchedStopSequence: "END",
		Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			_, err := engine.EncodeResponse(target, response, llmprotocol.Envelope{})
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability")
			_, _, err = engine.EncodeResponseStream(
				target,
				response,
				llmprotocol.StreamContext{Context: context.Background()},
			)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability")
		})
	}
}

func TestStreamFinalizationPreservesCancellationCategory(t *testing.T) {
	errorEvent := streamFinalizationError(context.Canceled, "incomplete")
	if !errors.Is(errorEvent, context.Canceled) || errorEvent.Code != "stream_canceled" {
		t.Fatalf("cancellation = %+v", errorEvent)
	}
}

func TestResponsesFailureUsageSurvivesBufferedAndStreamingDecode(t *testing.T) {
	engine := NewBuiltinEngine()
	resource := `{"id":"resp_failed","object":"response","model":"provider-model","status":"failed","error":{"code":"provider_error","message":"failed after generation"},"output":[],"usage":{"input_tokens":4,"input_tokens_details":{"cached_tokens":1,"cache_write_tokens":0},"output_tokens":2,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":6}}`
	buffered, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, []byte(resource))
	if err != nil {
		t.Fatal(err)
	}
	assertFailedResponseUsage(t, buffered)

	stream := []byte(
		"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_failed\",\"object\":\"response\",\"model\":\"provider-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
			"event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":1,\"response\":" + resource + "}\n\n",
	)
	streamed, _, err := engine.DecodeResponseStream(
		llmprotocol.OpenAIResponsesV1,
		stream,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	assertFailedResponseUsage(t, streamed)
}

func TestResponsesTerminalResourceContainsItsCompletedOutput(t *testing.T) {
	engine := NewBuiltinEngine()
	for name, source := range map[string][]byte{
		"text":      streamFixture(llmprotocol.OpenAIChatV1),
		"tool_call": toolStreamFixture(llmprotocol.OpenAIChatV1),
	} {
		t.Run(name, func(t *testing.T) {
			assertResponsesTerminalOutput(t, engine, name, source)
		})
	}
}

func assertResponsesTerminalOutput(t *testing.T, engine *Engine, name string, source []byte) {
	t.Helper()
	stream, err := engine.NewStream(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model", ProviderModel: "source-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push(source)
	if err != nil {
		t.Fatal(err)
	}
	finalFrames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	output := responsesCompletedOutputFromFrames(t, append(frames, finalFrames...))
	if len(output) != 1 {
		t.Fatalf("terminal response output items = %d, want 1", len(output))
	}
	item, err := decodeResponsesItemWire(output[0], llmprotocol.DefaultPolicy(), true)
	if err != nil {
		t.Fatal(err)
	}
	assertResponsesTerminalItem(t, name, item)
}

func assertResponsesTerminalItem(t *testing.T, name string, item responsesItemWire) {
	t.Helper()
	if name == "tool_call" && (item.Type != "function_call" || item.CallID != "call_1" || item.Name != "lookup") {
		t.Fatalf("terminal tool item = %+v", item)
	}
	if name == "text" && (item.Type != "message" || item.Role != "assistant") {
		t.Fatalf("terminal message item = %+v", item)
	}
}

func responsesCompletedOutputFromFrames(t *testing.T, frames [][]byte) []json.RawMessage {
	t.Helper()
	framer := newSSEFramer(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
	for _, encoded := range frames {
		complete, err := framer.Push(encoded)
		if err != nil {
			t.Fatal(err)
		}
		for _, frame := range complete {
			parsed, err := parseSSEFrame(frame, llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
			if err != nil {
				t.Fatal(err)
			}
			var event responsesEventWire
			if err := json.Unmarshal(parsed.Data, &event); err != nil {
				t.Fatal(err)
			}
			if event.Type != "response.completed" {
				continue
			}
			var output []json.RawMessage
			if event.Response == nil || json.Unmarshal(event.Response.Output, &output) != nil {
				t.Fatalf("terminal Responses resource has invalid output: %+v", event.Response)
			}
			return output
		}
	}
	t.Fatal("Responses stream did not emit response.completed")
	return nil
}

func assertFailedResponseUsage(t *testing.T, response llmprotocol.Response) {
	t.Helper()
	if response.Error == nil || response.Usage.State != llmprotocol.UsageAvailable ||
		tokenValue(response.Usage.InputTotal) != 4 || tokenValue(response.Usage.OutputTotal) != 2 ||
		tokenValue(response.Usage.Total) != 6 {
		t.Fatalf("failed response usage = %+v error=%+v", response.Usage, response.Error)
	}
}
