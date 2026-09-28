package protocolcodec

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

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
		assertIncompleteStreamFailure(t, engine, source, target, partial[source])
	})
}

func assertIncompleteStreamFailure(
	t *testing.T,
	engine *Engine,
	source, target llmprotocol.WireFormat,
	payload []byte,
) {
	t.Helper()
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, events, _, pushErr := stream.Push(payload); pushErr != nil {
		t.Fatalf("partial stream rejected before transport ended: %v", pushErr)
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
		t.Run(string(target), func(t *testing.T) { assertChatUsageStreamTranslation(t, engine, target, payload) })
	}
}

func assertChatUsageStreamTranslation(t *testing.T, engine *Engine, target llmprotocol.WireFormat, payload []byte) {
	t.Helper()
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
	assertChatUsageEvents(t, append(events, finalEvents...))
}

func assertChatUsageEvents(t *testing.T, events []llmprotocol.Event) {
	t.Helper()
	text, terminal := "", 0
	var usage *llmprotocol.Usage
	for _, event := range events {
		switch event.Type {
		case llmprotocol.EventOutputTextDelta:
			text += event.Delta
		case llmprotocol.EventResponseCompleted:
			terminal++
			usage = event.Usage
		}
	}
	if text != "first second" || terminal != 1 || usage == nil ||
		usage.Total.Value == nil || *usage.Total.Value != 5 {
		t.Fatalf("translated stream text=%q terminals=%d usage=%+v events=%+v", text, terminal, usage, events)
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
		errorType := "authentication_error"
		if code != "authentication_error" {
			errorType = "server_error"
		}
		assertChatStreamErrorPayload(t, parsed, object, errorType, code, message, parameter)
	case llmprotocol.OpenAIResponsesV1:
		assertResponsesStreamErrorPayload(t, failure, parsed, object, code, message, parameter)
	case llmprotocol.AnthropicMessagesV1:
		assertAnthropicStreamErrorPayload(t, parsed, object, code, message)
	default:
		t.Fatalf("unexpected target format %q", format)
	}
}

func assertChatStreamErrorPayload(
	t *testing.T,
	parsed sseFrame,
	object map[string]json.RawMessage,
	errorType,
	code,
	message,
	parameter string,
) {
	t.Helper()
	if parsed.Event != "" || len(object) != 1 || object["error"] == nil {
		t.Fatalf("Chat stream error is not canonical: event=%q data=%s", parsed.Event, parsed.Data)
	}
	assertOpenAIErrorDetail(t, object["error"], errorType, code, message, parameter)
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
	if err := json.Unmarshal(raw, &responseObject); err != nil {
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

func assertOpenAIErrorDetail(t *testing.T, raw json.RawMessage, errorType, code, message, parameter string) {
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
	if err != nil || detail.Type != errorType || detail.Code == nil ||
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
