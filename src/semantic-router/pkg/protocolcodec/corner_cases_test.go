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
	for _, source := range builtinFormats {
		payload := streamFixture(source)
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
				policy := llmprotocol.DefaultPolicy()
				policy.Limits.BodyBytes = len(payload)
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
				for _, chunk := range splitBytesAt(payload, len(payload)/2) {
					if _, _, _, err := stream.Push(chunk); err != nil {
						t.Fatalf("stream exactly at cumulative body limit was rejected: %v", err)
					}
				}

				policy.Limits.BodyBytes = len(payload) - 1
				engine, err = NewEngine(NewBuiltinRegistry(), policy)
				if err != nil {
					t.Fatal(err)
				}
				stream, err = engine.NewStream(source, target, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
				})
				if err != nil {
					t.Fatal(err)
				}
				_, _, _, err = stream.Push(payload)
				assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "upstream_body_limit")
			})
		}
	}
}

func TestJSONDepthLimitIsInclusiveAcrossEveryProtocolSurface(t *testing.T) {
	for _, format := range builtinFormats {
		t.Run(string(format)+"/request", func(t *testing.T) {
			body := requestFixture(format)
			depth := maximumJSONDepth(t, body)
			engine := engineWithJSONDepth(t, depth)
			if _, _, _, err := engine.DecodeRequest(format, body); err != nil {
				t.Fatalf("request exactly at JSON depth limit was rejected: %v", err)
			}
			engine = engineWithJSONDepth(t, depth-1)
			_, _, _, err := engine.DecodeRequest(format, body)
			assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_json")
		})

		t.Run(string(format)+"/response", func(t *testing.T) {
			body := responseFixture(format)
			depth := maximumJSONDepth(t, body)
			engine := engineWithJSONDepth(t, depth)
			if _, _, _, err := engine.DecodeResponse(format, body); err != nil {
				t.Fatalf("response exactly at JSON depth limit was rejected: %v", err)
			}
			engine = engineWithJSONDepth(t, depth-1)
			_, _, _, err := engine.DecodeResponse(format, body)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
		})

		t.Run(string(format)+"/transport-error", func(t *testing.T) {
			body := transportErrorFixture(format)
			depth := maximumJSONDepth(t, body)
			engine := engineWithJSONDepth(t, depth)
			if _, err := engine.TranslateTransportError(format, format, body, nil); err != nil {
				t.Fatalf("transport error exactly at JSON depth limit was rejected: %v", err)
			}
			engine = engineWithJSONDepth(t, depth-1)
			_, err := engine.TranslateTransportError(format, format, body, nil)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
		})

		payload := streamFixture(format)
		depth := maximumSSEJSONDepth(t, payload)
		for _, target := range builtinFormats {
			t.Run(string(format)+"/stream-to-"+string(target), func(t *testing.T) {
				engine := engineWithJSONDepth(t, depth)
				stream, err := engine.NewStream(format, target, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
				})
				if err != nil {
					t.Fatal(err)
				}
				if _, _, _, err := stream.Push(payload); err != nil {
					t.Fatalf("stream exactly at JSON depth limit was rejected: %v", err)
				}
				if _, _, _, err := stream.Finalize(nil); err != nil {
					t.Fatalf("stream exactly at JSON depth limit could not finalize: %v", err)
				}

				engine = engineWithJSONDepth(t, depth-1)
				stream, err = engine.NewStream(format, target, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
				})
				if err != nil {
					t.Fatal(err)
				}
				_, _, _, err = stream.Push(payload)
				assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
			})
		}
	}
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
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"/"+string(target), func(t *testing.T) {
				stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model",
				})
				if err != nil {
					t.Fatal(err)
				}
				payload := append(append([]byte(nil), partial[source]...), []byte("data: {\n\n")...)
				frames, events, _, err := stream.Push(payload)
				if err == nil {
					t.Fatal("malformed trailing frame was accepted")
				}
				if len(frames) == 0 || !bytes.Contains(bytes.Join(frames, nil), []byte("partial")) {
					t.Fatalf("valid public prefix was dropped: %q", bytes.Join(frames, nil))
				}
				foundDelta := false
				for _, event := range events {
					if event.Type == llmprotocol.EventOutputTextDelta && event.Delta == "partial" {
						foundDelta = true
					}
				}
				if !foundDelta {
					t.Fatalf("valid semantic prefix was dropped: %+v", events)
				}
				var first *llmprotocol.ProtocolError
				if !errors.As(err, &first) {
					t.Fatalf("first failure = %T %v", err, err)
				}
				_, _, _, repeatedErr := stream.Push([]byte("data: {}\n\n"))
				var repeated *llmprotocol.ProtocolError
				if !errors.As(repeatedErr, &repeated) || repeated.Code != first.Code || repeated.Category != first.Category {
					t.Fatalf("poisoned stream failure changed: first=%+v repeated=%T %v", first, repeatedErr, repeatedErr)
				}
			})
		}
	}
}

func TestMalformedDataAfterSuccessTerminalCannotPublishSuccess(t *testing.T) {
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
				payload := append(append([]byte(nil), streamFixture(source)...), []byte("data: {\n\n")...)
				frames, events, _, pushErr := stream.Push(payload)
				if pushErr == nil {
					t.Fatal("data after terminal was accepted")
				}
				for _, event := range events {
					if event.Type == llmprotocol.EventResponseCompleted {
						t.Fatalf("invalid source published a successful semantic terminal: %+v", events)
					}
				}
				wire := bytes.Join(frames, nil)
				assertNoSuccessfulStreamTerminal(t, target, wire)
				terminalFrames, terminalEvents, _, finalizeErr := stream.Finalize(pushErr)
				if finalizeErr != nil {
					t.Fatal(finalizeErr)
				}
				wire = append(wire, bytes.Join(terminalFrames, nil)...)
				assertNoSuccessfulStreamTerminal(t, target, wire)
				for _, event := range terminalEvents {
					if event.Type == llmprotocol.EventResponseCompleted {
						t.Fatalf("invalid source finalized as success: %+v", terminalEvents)
					}
				}
				if !bytes.Contains(wire, []byte("stream")) {
					t.Fatalf("public failure terminal is missing: %s", wire)
				}
			})
		}
	}
}

func TestTransportFailureAfterProviderSuccessSuppressesDeferredTerminal(t *testing.T) {
	reasons := []struct {
		name    string
		reason  error
		message string
	}{
		{name: "canceled", reason: context.Canceled, message: "stream was canceled"},
		{name: "deadline", reason: context.DeadlineExceeded, message: "stream deadline was exceeded"},
	}
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			for _, testCase := range reasons {
				t.Run(string(source)+"/"+string(target)+"/"+testCase.name, func(t *testing.T) {
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
					for _, event := range events {
						if event.Type == llmprotocol.EventResponseCompleted {
							t.Fatalf("provider success was not held until HTTP EOS: %+v", events)
						}
					}
					assertNoSuccessfulStreamTerminal(t, target, bytes.Join(frames, nil))

					finalFrames, finalEvents, _, finalizeErr := stream.Finalize(testCase.reason)
					if finalizeErr != nil {
						t.Fatal(finalizeErr)
					}
					failed := false
					for _, event := range finalEvents {
						if event.Type == llmprotocol.EventResponseCompleted {
							t.Fatalf("transport failure released provider success: %+v", finalEvents)
						}
						if event.Type == llmprotocol.EventResponseFailed && event.Error != nil {
							failed = event.Error.Message == testCase.message && event.Failure == llmprotocol.FailureTransport
						}
					}
					if !failed {
						t.Fatalf("transport failure has no matching neutral terminal: %+v", finalEvents)
					}
					wire := append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...)
					assertNoSuccessfulStreamTerminal(t, target, wire)
					if !bytes.Contains(wire, []byte(testCase.message)) {
						t.Fatalf("public %s failure terminal is missing: %s", testCase.name, wire)
					}
				})
			}
		}
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
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"/"+string(target), func(t *testing.T) {
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
				for _, event := range events {
					if event.Type == llmprotocol.EventResponseCompleted {
						t.Fatalf("completion was not deferred: %+v", events)
					}
				}
				finalFrames, finalEvents, _, finalizeErr := stream.Finalize(nil)
				if !errors.Is(finalizeErr, mutationFailure) {
					t.Fatalf("terminal mutation failure = %T %v", finalizeErr, finalizeErr)
				}
				for _, event := range finalEvents {
					if event.Type == llmprotocol.EventResponseCompleted {
						t.Fatalf("failed terminal mutation returned success: %+v", finalEvents)
					}
				}
				wire := append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...)
				assertNoSuccessfulStreamTerminal(t, target, wire)
				if !bytes.Contains(wire, []byte("stream")) {
					t.Fatalf("terminal mutation failure has no public error: %s", wire)
				}
			})
		}
	}
}

func splitBytesAt(payload []byte, index int) [][]byte {
	if index <= 0 || index >= len(payload) {
		return [][]byte{payload}
	}
	return [][]byte{payload[:index], payload[index:]}
}

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
		})
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
			stream, err := engine.NewStream(
				llmprotocol.OpenAIChatV1,
				llmprotocol.OpenAIResponsesV1,
				llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model",
					ProviderModel: "source-model",
				},
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
			frames = append(frames, finalFrames...)
			output := responsesCompletedOutputFromFrames(t, frames)
			if len(output) != 1 {
				t.Fatalf("terminal response output items = %d, want 1", len(output))
			}
			item, err := decodeResponsesItemWire(output[0], llmprotocol.DefaultPolicy(), true)
			if err != nil {
				t.Fatal(err)
			}
			if name == "tool_call" && (item.Type != "function_call" || item.CallID != "call_1" || item.Name != "lookup") {
				t.Fatalf("terminal tool item = %+v", item)
			}
			if name == "text" && (item.Type != "message" || item.Role != "assistant") {
				t.Fatalf("terminal message item = %+v", item)
			}
		})
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
