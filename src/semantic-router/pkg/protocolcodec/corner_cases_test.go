package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestBufferedWireContractsRejectNonObjectDocuments(t *testing.T) {
	engine := NewBuiltinEngine()
	formats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}
	documents := []struct {
		name string
		body string
	}{
		{name: "null", body: "null"},
		{name: "array", body: `[]`},
		{name: "string", body: `"text"`},
		{name: "boolean", body: "true"},
		{name: "number", body: "42"},
		{name: "whitespace", body: " \t\n"},
	}
	for _, document := range documents {
		for _, format := range formats {
			t.Run("request/"+string(format)+"/"+document.name, func(t *testing.T) {
				_, _, _, err := engine.DecodeRequest(format, []byte(document.body))
				assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_json")
			})
			t.Run("response/"+string(format)+"/"+document.name, func(t *testing.T) {
				_, _, _, err := engine.DecodeResponse(format, []byte(document.body))
				assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
			})
			t.Run("transport error/"+string(format)+"/"+document.name, func(t *testing.T) {
				_, err := engine.TranslateTransportError(format, llmprotocol.OpenAIChatV1, []byte(document.body), nil)
				assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
			})
		}
	}
}

func TestRequestTranslationRejectsTargetsThatRequireConversationMessages(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"provider-model","input":[]}`)

	_, err := engine.TranslateRequest(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, body, nil)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "chat_messages_required")

	_, err = engine.TranslateRequest(llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1, body, nil)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "anthropic_messages_required")
}

func TestResponsesTranslationSynthesizesRequiredErrorCode(t *testing.T) {
	engine := NewBuiltinEngine()
	translated, err := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		[]byte(`{"error":{"type":"server_error","message":"upstream failed"}}`),
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, translated.Body); err != nil {
		t.Fatalf("translated Responses error is invalid: %v\n%s", err, translated.Body)
	}
	if !bytes.Contains(translated.Body, []byte(`"code":"server_error"`)) {
		t.Fatalf("translated Responses error does not contain its required canonical code: %s", translated.Body)
	}
}

func TestAnthropicTransportErrorCannotEnterModelResponsePath(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"type":"error","error":{"type":"api_error","message":"upstream failed"}}`)

	_, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "anthropic_transport_error_on_response_path")

	transportError, _, err := engine.DecodeTransportError(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatalf("Anthropic error envelope was not accepted on the transport-error path: %v", err)
	}
	if transportError.Error == nil || transportError.Error.Message != "upstream failed" {
		t.Fatalf("Anthropic transport error was not decoded: %#v", transportError)
	}
}

func TestResponsesContentRequiresItsDiscriminatorFields(t *testing.T) {
	engine := NewBuiltinEngine()
	missingRefusal := []byte(`{"id":"response_1","status":"completed","output":[{"type":"message","id":"message_1","role":"assistant","status":"completed","content":[{"type":"refusal"}]}]}`)
	_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, missingRefusal)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_response_content")

	emptyRefusal := []byte(`{"id":"response_1","status":"completed","output":[{"type":"message","id":"message_1","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":""}]}]}`)
	translated, err := engine.TranslateResponse(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, emptyRefusal, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, translated.Body); err != nil {
		t.Fatalf("empty refusal was not preserved as an explicit Chat refusal: %v\n%s", err, translated.Body)
	}
}

func TestStreamWireContractsRejectNonObjectEventData(t *testing.T) {
	fixtures := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:        "data: null\n\n",
		llmprotocol.OpenAIResponsesV1:   "event: response.created\ndata: null\n\n",
		llmprotocol.AnthropicMessagesV1: "event: message_start\ndata: null\n\n",
	}
	for format, body := range fixtures {
		t.Run(string(format), func(t *testing.T) {
			stream, err := NewBuiltinEngine().NewStream(format, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = stream.Push([]byte(body))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
		})
	}
}

func TestBufferedBodyLimitIsInclusiveAcrossEveryProtocol(t *testing.T) {
	for _, format := range builtinFormats {
		t.Run(string(format)+"/request", func(t *testing.T) {
			body := requestFixture(format)
			policy := llmprotocol.DefaultPolicy()
			policy.Limits.BodyBytes = len(body)
			engine, err := NewEngine(NewBuiltinRegistry(), policy)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, _, err := engine.DecodeRequest(format, body); err != nil {
				t.Fatalf("request exactly at body limit was rejected: %v", err)
			}
			policy.Limits.BodyBytes--
			engine, err = NewEngine(NewBuiltinRegistry(), policy)
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeRequest(format, body)
			assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "body_limit")
		})

		t.Run(string(format)+"/response", func(t *testing.T) {
			body := responseFixture(format)
			policy := llmprotocol.DefaultPolicy()
			policy.Limits.BodyBytes = len(body)
			engine, err := NewEngine(NewBuiltinRegistry(), policy)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, _, err := engine.DecodeResponse(format, body); err != nil {
				t.Fatalf("response exactly at body limit was rejected: %v", err)
			}
			policy.Limits.BodyBytes--
			engine, err = NewEngine(NewBuiltinRegistry(), policy)
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeResponse(format, body)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "upstream_body_limit")
		})
	}
}

func TestStreamingBodyLimitIsCumulativeAndInclusiveAcrossEveryProtocol(t *testing.T) {
	forEachBuiltinFormatPair(t, assertStreamingBodyLimit)
}

func assertStreamingBodyLimit(t *testing.T, source, target llmprotocol.WireFormat) {
	t.Helper()
	payload := streamFixture(source)
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.BodyBytes = len(payload)
	stream := newTestStream(t, policy, source, target)
	for _, chunk := range splitBytesAt(payload, len(payload)/2) {
		if _, _, _, err := stream.Push(chunk); err != nil {
			t.Fatalf("stream exactly at cumulative body limit was rejected: %v", err)
		}
	}
	policy.Limits.BodyBytes--
	stream = newTestStream(t, policy, source, target)
	_, _, _, err := stream.Push(payload)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "upstream_body_limit")
}

func newTestStream(t *testing.T, policy llmprotocol.Policy, source, target llmprotocol.WireFormat) *StreamEngine {
	t.Helper()
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

func TestJSONDepthLimitIsInclusiveAcrossEveryProtocolSurface(t *testing.T) {
	for _, format := range builtinFormats {
		assertBufferedJSONDepthSurfaces(t, format)
	}
	forEachBuiltinFormatPair(t, assertStreamJSONDepth)
}

func assertBufferedJSONDepthSurfaces(t *testing.T, format llmprotocol.WireFormat) {
	t.Helper()
	t.Run(string(format)+"/request", func(t *testing.T) { assertRequestJSONDepth(t, format) })
	t.Run(string(format)+"/response", func(t *testing.T) { assertResponseJSONDepth(t, format) })
	t.Run(string(format)+"/transport-error", func(t *testing.T) { assertTransportErrorJSONDepth(t, format) })
}

func assertRequestJSONDepth(t *testing.T, format llmprotocol.WireFormat) {
	body := requestFixture(format)
	depth := maximumJSONDepth(t, body)
	if _, _, _, err := engineWithJSONDepth(t, depth).DecodeRequest(format, body); err != nil {
		t.Fatalf("request exactly at JSON depth limit was rejected: %v", err)
	}
	_, _, _, err := engineWithJSONDepth(t, depth-1).DecodeRequest(format, body)
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_json")
}

func assertResponseJSONDepth(t *testing.T, format llmprotocol.WireFormat) {
	body := responseFixture(format)
	depth := maximumJSONDepth(t, body)
	if _, _, _, err := engineWithJSONDepth(t, depth).DecodeResponse(format, body); err != nil {
		t.Fatalf("response exactly at JSON depth limit was rejected: %v", err)
	}
	_, _, _, err := engineWithJSONDepth(t, depth-1).DecodeResponse(format, body)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}

func assertTransportErrorJSONDepth(t *testing.T, format llmprotocol.WireFormat) {
	body := transportErrorFixture(format)
	depth := maximumJSONDepth(t, body)
	if _, err := engineWithJSONDepth(t, depth).TranslateTransportError(format, format, body, nil); err != nil {
		t.Fatalf("transport error exactly at JSON depth limit was rejected: %v", err)
	}
	_, err := engineWithJSONDepth(t, depth-1).TranslateTransportError(format, format, body, nil)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}

func assertStreamJSONDepth(t *testing.T, source, target llmprotocol.WireFormat) {
	payload := streamFixture(source)
	depth := maximumSSEJSONDepth(t, payload)
	stream := newTestStream(t, policyWithJSONDepth(depth), source, target)
	if _, _, _, err := stream.Push(payload); err != nil {
		t.Fatalf("stream exactly at JSON depth limit was rejected: %v", err)
	}
	if _, _, _, err := stream.Finalize(nil); err != nil {
		t.Fatalf("stream exactly at JSON depth limit could not finalize: %v", err)
	}
	stream = newTestStream(t, policyWithJSONDepth(depth-1), source, target)
	_, _, _, err := stream.Push(payload)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}

func policyWithJSONDepth(depth int) llmprotocol.Policy {
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.JSONDepth = depth
	return policy
}

func engineWithJSONDepth(t *testing.T, depth int) *Engine {
	t.Helper()
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.JSONDepth = depth
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	return engine
}

func maximumJSONDepth(t *testing.T, body []byte) int {
	t.Helper()
	var document any
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&document); err != nil {
		t.Fatal(err)
	}
	return maximumDecodedJSONDepth(document, 0)
}

func maximumSSEJSONDepth(t *testing.T, body []byte) int {
	t.Helper()
	maximum := 0
	for _, line := range bytes.Split(body, []byte{'\n'}) {
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		data := bytes.TrimPrefix(line, []byte("data: "))
		if bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		if depth := maximumJSONDepth(t, data); depth > maximum {
			maximum = depth
		}
	}
	if maximum == 0 {
		t.Fatal("SSE fixture has no JSON data event")
	}
	return maximum
}

func maximumDecodedJSONDepth(value any, depth int) int {
	maximum := depth
	switch typed := value.(type) {
	case map[string]any:
		for _, child := range typed {
			if candidate := maximumDecodedJSONDepth(child, depth+1); candidate > maximum {
				maximum = candidate
			}
		}
	case []any:
		for _, child := range typed {
			if candidate := maximumDecodedJSONDepth(child, depth+1); candidate > maximum {
				maximum = candidate
			}
		}
	}
	return maximum
}

func TestValidFramesBeforeMalformedFrameAreNotDroppedAcrossEveryProtocolPair(t *testing.T) {
	partial := map[llmprotocol.WireFormat][]byte{
		llmprotocol.OpenAIChatV1: []byte(
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
		),
		llmprotocol.OpenAIResponsesV1: []byte(
			"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"provider-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
				"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n" +
				"event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}\n\n" +
				"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":3,\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"partial\"}\n\n",
		),
		llmprotocol.AnthropicMessagesV1: []byte(
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
		),
	}
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertValidPrefixBeforeMalformedFrame(t, engine, source, target, partial[source])
	})
}

func assertValidPrefixBeforeMalformedFrame(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat, prefix []byte) {
	t.Helper()
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	payload := append(append([]byte(nil), prefix...), []byte("data: {\n\n")...)
	frames, events, _, pushErr := stream.Push(payload)
	if pushErr == nil {
		t.Fatal("malformed trailing frame was accepted")
	}
	if len(frames) == 0 || !bytes.Contains(bytes.Join(frames, nil), []byte("partial")) {
		t.Fatalf("valid public prefix was dropped: %q", bytes.Join(frames, nil))
	}
	if !containsTextDelta(events, "partial") {
		t.Fatalf("valid semantic prefix was dropped: %+v", events)
	}
	assertPoisonedStreamErrorStable(t, stream, pushErr)
}

func containsTextDelta(events []llmprotocol.Event, want string) bool {
	for _, event := range events {
		if event.Type == llmprotocol.EventOutputTextDelta && event.Delta == want {
			return true
		}
	}
	return false
}

func assertPoisonedStreamErrorStable(t *testing.T, stream *StreamEngine, firstErr error) {
	t.Helper()
	var first *llmprotocol.ProtocolError
	if !errors.As(firstErr, &first) {
		t.Fatalf("first failure = %T %v", firstErr, firstErr)
	}
	_, _, _, repeatedErr := stream.Push([]byte("data: {}\n\n"))
	var repeated *llmprotocol.ProtocolError
	if !errors.As(repeatedErr, &repeated) || repeated.Code != first.Code || repeated.Category != first.Category {
		t.Fatalf("poisoned stream failure changed: first=%+v repeated=%T %v", first, repeatedErr, repeatedErr)
	}
}

func TestMalformedDataAfterSuccessTerminalCannotPublishSuccess(t *testing.T) {
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertMalformedDataAfterTerminal(t, engine, source, target)
	})
}

func assertMalformedDataAfterTerminal(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat) {
	t.Helper()
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	payload := append(append([]byte(nil), streamFixture(source)...), []byte("data: {\n\n")...)
	frames, events, _, pushErr := stream.Push(payload)
	if pushErr == nil {
		t.Fatal("data after terminal was accepted")
	}
	assertNoCompletedEvent(t, events, "invalid source published a successful semantic terminal")
	wire := bytes.Join(frames, nil)
	assertNoSuccessfulStreamTerminal(t, target, wire)
	terminalFrames, terminalEvents, _, finalizeErr := stream.Finalize(pushErr)
	if finalizeErr != nil {
		t.Fatal(finalizeErr)
	}
	wire = append(wire, bytes.Join(terminalFrames, nil)...)
	assertNoSuccessfulStreamTerminal(t, target, wire)
	assertNoCompletedEvent(t, terminalEvents, "invalid source finalized as success")
	if !bytes.Contains(wire, []byte("stream")) {
		t.Fatalf("public failure terminal is missing: %s", wire)
	}
}

func assertNoCompletedEvent(t *testing.T, events []llmprotocol.Event, message string) {
	t.Helper()
	for _, event := range events {
		if event.Type == llmprotocol.EventResponseCompleted {
			t.Fatalf("%s: %+v", message, events)
		}
	}
}

func TestTransportFailureAfterProviderSuccessSuppressesDeferredTerminal(t *testing.T) {
	reasons := []streamFinalizationCase{
		{name: "canceled", reason: context.Canceled, message: "stream was canceled"},
		{name: "deadline", reason: context.DeadlineExceeded, message: "stream deadline was exceeded"},
	}
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		for _, testCase := range reasons {
			t.Run(testCase.name, func(t *testing.T) {
				assertTransportFailureAfterSuccess(t, engine, source, target, testCase)
			})
		}
	})
}

type streamFinalizationCase struct {
	name    string
	reason  error
	message string
}

func assertTransportFailureAfterSuccess(
	t *testing.T,
	engine *Engine,
	source, target llmprotocol.WireFormat,
	testCase streamFinalizationCase,
) {
	t.Helper()
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, err := stream.Push(streamFixture(source))
	if err != nil {
		t.Fatal(err)
	}
	assertNoCompletedEvent(t, events, "provider success was not held until HTTP EOS")
	assertNoSuccessfulStreamTerminal(t, target, bytes.Join(frames, nil))
	finalFrames, finalEvents, _, finalizeErr := stream.Finalize(testCase.reason)
	if finalizeErr != nil {
		t.Fatal(finalizeErr)
	}
	assertTransportFailureEvent(t, finalEvents, testCase.message)
	wire := append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...)
	assertNoSuccessfulStreamTerminal(t, target, wire)
	if !bytes.Contains(wire, []byte(testCase.message)) {
		t.Fatalf("public %s failure terminal is missing: %s", testCase.name, wire)
	}
}

func assertTransportFailureEvent(t *testing.T, events []llmprotocol.Event, message string) {
	t.Helper()
	failed := false
	for _, event := range events {
		if event.Type == llmprotocol.EventResponseCompleted {
			t.Fatalf("transport failure released provider success: %+v", events)
		}
		if event.Type == llmprotocol.EventResponseFailed && event.Error != nil {
			failed = event.Error.Message == message && event.Failure == llmprotocol.FailureTransport
		}
	}
	if !failed {
		t.Fatalf("transport failure has no matching neutral terminal: %+v", events)
	}
}

func TestStreamFinalizationPreservesAnExistingTypedFailure(t *testing.T) {
	cause := llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable,
		"invalid_provider_event",
		"provider event was invalid",
		errors.New("wire detail"),
	)
	if got := streamFinalizationError(cause, "generic fallback"); got != cause {
		t.Fatalf("typed stream failure was replaced: got=%p want=%p", got, cause)
	}
}

func assertNoSuccessfulStreamTerminal(t *testing.T, format llmprotocol.WireFormat, wire []byte) {
	t.Helper()
	switch format {
	case llmprotocol.OpenAIChatV1:
		if bytes.Contains(wire, []byte("data: [DONE]")) {
			t.Fatalf("Chat stream published a success sentinel: %s", wire)
		}
	case llmprotocol.OpenAIResponsesV1:
		if bytes.Contains(wire, []byte("event: response.completed")) {
			t.Fatalf("Responses stream published a success terminal: %s", wire)
		}
	case llmprotocol.AnthropicMessagesV1:
		if bytes.Contains(wire, []byte("event: message_stop")) {
			t.Fatalf("Messages stream published a success terminal: %s", wire)
		}
	}
}

func TestEventStreamEncoderCannotFinalizeIncompleteStreamAsSuccess(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, format := range builtinFormats {
		t.Run(string(format), func(t *testing.T) {
			encoder, err := engine.NewEventStreamEncoder(format, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public-model", ResponseID: "response_1",
			})
			if err != nil {
				t.Fatal(err)
			}
			frames, _, err := encoder.Push(llmprotocol.Event{
				Type: llmprotocol.EventResponseStarted, ResponseID: "response_1", Model: "public-model",
			})
			if err != nil {
				t.Fatal(err)
			}
			terminal, _, err := encoder.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			wire := append(bytes.Join(frames, nil), bytes.Join(terminal, nil)...)
			assertNoSuccessfulStreamTerminal(t, format, wire)
			if !bytes.Contains(wire, []byte("stream")) {
				t.Fatalf("incomplete event stream has no failure terminal: %s", wire)
			}
		})
	}
}

func TestEventStreamEncoderIsPoisonedByItsFirstInvalidEvent(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, format := range builtinFormats {
		t.Run(string(format), func(t *testing.T) {
			encoder, err := engine.NewEventStreamEncoder(format, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public-model", ResponseID: "response_1",
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, firstErr := encoder.Push(llmprotocol.Event{
				Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, ContentIndex: 0, Delta: "orphan",
			})
			assertProtocolError(t, firstErr, llmprotocol.ErrorUpstreamUnavailable, "stream_start_missing")
			_, _, repeatedErr := encoder.Push(llmprotocol.Event{
				Type: llmprotocol.EventResponseStarted, ResponseID: "response_1", Model: "public-model",
			})
			var first, repeated *llmprotocol.ProtocolError
			if !errors.As(firstErr, &first) || !errors.As(repeatedErr, &repeated) ||
				first.Code != repeated.Code || first.Category != repeated.Category {
				t.Fatalf("event encoder failure changed: first=%T %v repeated=%T %v", firstErr, firstErr, repeatedErr, repeatedErr)
			}
			terminal, _, err := encoder.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			wire := bytes.Join(terminal, nil)
			assertNoSuccessfulStreamTerminal(t, format, wire)
			if !bytes.Contains(wire, []byte("stream")) {
				t.Fatalf("poisoned event stream has no failure terminal: %s", wire)
			}
		})
	}
}

func TestDeferredTerminalMutationFailurePublishesNoSuccessAcrossProtocolMatrix(t *testing.T) {
	mutationFailure := errors.New("terminal mutation rejected")
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertDeferredTerminalMutationFailure(t, engine, source, target, mutationFailure)
	})
}

func assertDeferredTerminalMutationFailure(
	t *testing.T,
	engine *Engine,
	source, target llmprotocol.WireFormat,
	mutationFailure error,
) {
	t.Helper()
	stream, err := engine.NewStreamWithMutation(
		source,
		target,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
		func(event *llmprotocol.Event) error {
			if event.Type == llmprotocol.EventResponseCompleted {
				return mutationFailure
			}
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, err := stream.Push(streamFixture(source))
	if err != nil {
		t.Fatal(err)
	}
	assertNoCompletedEvent(t, events, "completion was not deferred")
	finalFrames, finalEvents, _, finalizeErr := stream.Finalize(nil)
	if !errors.Is(finalizeErr, mutationFailure) {
		t.Fatalf("terminal mutation failure = %T %v", finalizeErr, finalizeErr)
	}
	assertNoCompletedEvent(t, finalEvents, "failed terminal mutation returned success")
	wire := append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...)
	assertNoSuccessfulStreamTerminal(t, target, wire)
	if !bytes.Contains(wire, []byte("stream")) {
		t.Fatalf("terminal mutation failure has no public error: %s", wire)
	}
}

func splitBytesAt(payload []byte, index int) [][]byte {
	if index <= 0 || index >= len(payload) {
		return [][]byte{payload}
	}
	return [][]byte{payload[:index], payload[index:]}
}
