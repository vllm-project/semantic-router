package protocolcodec

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicTransportErrorTranslationUsesCanonicalWireShapes(t *testing.T) {
	upstream := []byte(`{"type":"error","error":{"type":"authentication_error","message":"API key is invalid."},"request_id":"req_provider_1"}`)
	engine := NewBuiltinEngine()
	tests := []struct {
		name   string
		target llmprotocol.WireFormat
		golden string
	}{
		{
			name: "Anthropic", target: llmprotocol.AnthropicMessagesV1,
			golden: `{"type":"error","error":{"type":"authentication_error","message":"API key is invalid."},"request_id":"req_provider_1"}`,
		},
		{
			name: "OpenAI Chat", target: llmprotocol.OpenAIChatV1,
			golden: `{"error":{"type":"authentication_error","code":"authentication_error","message":"API key is invalid.","param":null}}`,
		},
		{
			name: "OpenAI Responses", target: llmprotocol.OpenAIResponsesV1,
			golden: `{"error":{"type":"authentication_error","code":"authentication_error","message":"API key is invalid.","param":null}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			translated, err := engine.TranslateTransportError(
				llmprotocol.AnthropicMessagesV1,
				test.target,
				upstream,
				nil,
			)
			if err != nil {
				t.Fatalf("TranslateTransportError() error = %v", err)
			}
			assertAnthropicAuthenticationError(t, translated.TransportError.Error)
			if translated.TransportError.ProviderRequestID != "req_provider_1" {
				t.Fatalf("provider request ID = %q", translated.TransportError.ProviderRequestID)
			}
			if string(translated.Body) != test.golden {
				t.Fatalf("transport error body = %s, want %s", translated.Body, test.golden)
			}
		})
	}
}

func TestAnthropicTransportErrorTranslationRejectsMalformedProviderJSON(t *testing.T) {
	translated, err := NewBuiltinEngine().TranslateTransportError(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIChatV1,
		[]byte(`{"type":"error"`),
		nil,
	)
	if err == nil {
		t.Fatal("malformed Anthropic transport error was accepted")
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUpstreamUnavailable {
		t.Fatalf("malformed provider error = %T %v", err, err)
	}
	if protocolError.Code != "invalid_upstream_json" {
		t.Fatalf("malformed provider error code = %q", protocolError.Code)
	}
	if translated.Body != nil || translated.TransportError.Error != nil {
		t.Fatalf("malformed provider error produced public output: %+v", translated)
	}
}

func TestTransportErrorTranslationRejectsInvalidProviderEnvelopes(t *testing.T) {
	tests := []struct {
		name   string
		source llmprotocol.WireFormat
		body   string
		code   string
	}{
		{name: "Chat missing error", source: llmprotocol.OpenAIChatV1, body: `{}`, code: "upstream_error_required"},
		{name: "Responses null error", source: llmprotocol.OpenAIResponsesV1, body: `{"error":null}`, code: "upstream_error_required"},
		{name: "Chat missing type", source: llmprotocol.OpenAIChatV1, body: `{"error":{"message":"failed"}}`, code: "upstream_error_type_required"},
		{name: "Responses whitespace type", source: llmprotocol.OpenAIResponsesV1, body: `{"error":{"type":"  ","message":"failed"}}`, code: "upstream_error_type_required"},
		{name: "Chat missing message", source: llmprotocol.OpenAIChatV1, body: `{"error":{"type":"server_error"}}`, code: "upstream_error_message_required"},
		{name: "Responses whitespace message", source: llmprotocol.OpenAIResponsesV1, body: `{"error":{"type":"server_error","message":"\t"}}`, code: "upstream_error_message_required"},
		{name: "Anthropic missing envelope type", source: llmprotocol.AnthropicMessagesV1, body: `{"error":{"type":"api_error","message":"failed"}}`, code: "invalid_upstream_error_envelope"},
		{name: "Anthropic wrong envelope type", source: llmprotocol.AnthropicMessagesV1, body: `{"type":"message","error":{"type":"api_error","message":"failed"}}`, code: "invalid_upstream_error_envelope"},
		{name: "Anthropic missing error", source: llmprotocol.AnthropicMessagesV1, body: `{"type":"error"}`, code: "upstream_error_required"},
		{name: "Anthropic null error", source: llmprotocol.AnthropicMessagesV1, body: `{"type":"error","error":null}`, code: "upstream_error_required"},
		{name: "Anthropic missing error type", source: llmprotocol.AnthropicMessagesV1, body: `{"type":"error","error":{"message":"failed"}}`, code: "upstream_error_type_required"},
		{name: "Anthropic whitespace error type", source: llmprotocol.AnthropicMessagesV1, body: `{"type":"error","error":{"type":" ","message":"failed"}}`, code: "upstream_error_type_required"},
		{name: "Anthropic missing message", source: llmprotocol.AnthropicMessagesV1, body: `{"type":"error","error":{"type":"api_error"}}`, code: "upstream_error_message_required"},
		{name: "Anthropic whitespace message", source: llmprotocol.AnthropicMessagesV1, body: `{"type":"error","error":{"type":"api_error","message":"\n"}}`, code: "upstream_error_message_required"},
	}
	for _, test := range tests {
		for _, target := range []llmprotocol.WireFormat{
			llmprotocol.OpenAIChatV1,
			llmprotocol.OpenAIResponsesV1,
			llmprotocol.AnthropicMessagesV1,
		} {
			t.Run(test.name+"/to/"+string(target), func(t *testing.T) {
				assertInvalidProviderEnvelope(t, test.source, target, []byte(test.body), test.code)
			})
		}
	}
}

func assertInvalidProviderEnvelope(
	t *testing.T,
	source, target llmprotocol.WireFormat,
	body []byte,
	wantCode string,
) {
	t.Helper()
	translated, err := NewBuiltinEngine().TranslateTransportError(source, target, body, nil)
	if err == nil {
		t.Fatal("invalid provider transport error was accepted")
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("provider error = %T %v, want *llmprotocol.ProtocolError", err, err)
	}
	if protocolError.Category != llmprotocol.ErrorUpstreamUnavailable || protocolError.Code != wantCode {
		t.Fatalf("provider error = %+v, want category=%q code=%q", protocolError, llmprotocol.ErrorUpstreamUnavailable, wantCode)
	}
	if translated.Body != nil || translated.TransportError.Error != nil {
		t.Fatalf("invalid provider error produced public output: %+v", translated)
	}
}

func TestTransportErrorBodyLimitIsExactForEverySourceFormat(t *testing.T) {
	fixtures := map[llmprotocol.WireFormat][]byte{
		llmprotocol.OpenAIChatV1:        []byte(`{"error":{"type":"server_error","message":"failed"}}`),
		llmprotocol.OpenAIResponsesV1:   []byte(`{"error":{"type":"server_error","message":"failed"}}`),
		llmprotocol.AnthropicMessagesV1: []byte(`{"type":"error","error":{"type":"api_error","message":"failed"}}`),
	}
	for source, body := range fixtures {
		t.Run(string(source), func(t *testing.T) {
			policy := llmprotocol.DefaultPolicy()
			policy.Limits.BodyBytes = len(body)
			engine, err := NewEngine(NewBuiltinRegistry(), policy)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := engine.TranslateTransportError(source, llmprotocol.OpenAIChatV1, body, nil); err != nil {
				t.Fatalf("body at the exact limit was rejected: %v", err)
			}

			policy.Limits.BodyBytes = len(body) - 1
			engine, err = NewEngine(NewBuiltinRegistry(), policy)
			if err != nil {
				t.Fatal(err)
			}
			_, err = engine.TranslateTransportError(source, llmprotocol.OpenAIChatV1, body, nil)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "upstream_body_limit")
		})
	}
}

func TestOpenAIProviderCodeDoesNotBecomeAnthropicErrorType(t *testing.T) {
	tests := []struct {
		name, errorType, code, anthropicType string
	}{
		{name: "unknown code", errorType: "authentication_error", code: "invalid_api_key", anthropicType: "authentication_error"},
		{name: "colliding conflict code", errorType: "authentication_error", code: "conflict_error", anthropicType: "authentication_error"},
		{name: "colliding timeout code", errorType: "server_error", code: "timeout_error", anthropicType: "api_error"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := []byte(`{"error":{"type":"` + test.errorType + `","code":"` + test.code + `","message":"request failed","param":null}}`)
			translated, err := NewBuiltinEngine().TranslateTransportError(
				llmprotocol.OpenAIChatV1,
				llmprotocol.AnthropicMessagesV1,
				body,
				nil,
			)
			if err != nil {
				t.Fatal(err)
			}
			if translated.TransportError.Error == nil ||
				translated.TransportError.Error.Code != test.code {
				t.Fatalf("neutral provider code was not preserved: %+v", translated.TransportError.Error)
			}
			golden := `{"type":"error","error":{"type":"` + test.anthropicType + `","message":"request failed"}}`
			if string(translated.Body) != golden {
				t.Fatalf("Anthropic transport error = %s, want %s", translated.Body, golden)
			}
		})
	}
}

func TestAnthropicConflictAndTimeoutTranslateThroughNeutralCategory(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		category   llmprotocol.ErrorCategory
		openAIType string
		code       string
		message    string
	}{
		{
			name: "conflict", body: `{"type":"error","error":{"type":"conflict_error","message":"request conflict"}}`,
			category: llmprotocol.ErrorConflict, openAIType: "invalid_request_error",
			code: "conflict_error", message: "request conflict",
		},
		{
			name: "timeout", body: `{"type":"error","error":{"type":"timeout_error","message":"request timed out"}}`,
			category: llmprotocol.ErrorUpstreamTimeout, openAIType: "server_error",
			code: "timeout_error", message: "request timed out",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			translated, err := NewBuiltinEngine().TranslateTransportError(
				llmprotocol.AnthropicMessagesV1,
				llmprotocol.OpenAIResponsesV1,
				[]byte(test.body),
				nil,
			)
			if err != nil {
				t.Fatal(err)
			}
			if translated.TransportError.Error == nil ||
				translated.TransportError.Error.Category != test.category ||
				translated.TransportError.Error.Code != test.code {
				t.Fatalf("neutral transport error = %+v", translated.TransportError.Error)
			}
			golden := `{"error":{"type":"` + test.openAIType + `","code":"` + test.code + `","message":"` + test.message + `","param":null}}`
			if string(translated.Body) != golden {
				t.Fatalf("OpenAI transport error = %s, want %s", translated.Body, golden)
			}
		})
	}
}

func TestEncodeErrorDelegatesToCanonicalTransportContract(t *testing.T) {
	protocolError := &llmprotocol.ProtocolError{
		Category: llmprotocol.ErrorAuthentication,
		Code:     "authentication_error", Message: "API key is invalid.", Parameter: "model",
	}
	tests := []struct {
		format llmprotocol.WireFormat
		golden string
	}{
		{
			format: llmprotocol.OpenAIChatV1,
			golden: `{"error":{"type":"authentication_error","code":"authentication_error","message":"API key is invalid.","param":"model"}}`,
		},
		{
			format: llmprotocol.OpenAIResponsesV1,
			golden: `{"error":{"type":"authentication_error","code":"authentication_error","message":"API key is invalid.","param":"model"}}`,
		},
		{
			format: llmprotocol.AnthropicMessagesV1,
			golden: `{"type":"error","error":{"type":"authentication_error","message":"API key is invalid."}}`,
		},
	}
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(string(test.format), func(t *testing.T) {
			body, err := engine.EncodeError(test.format, protocolError)
			if err != nil {
				t.Fatalf("EncodeError() error = %v", err)
			}
			if string(body) != test.golden {
				t.Fatalf("encoded error = %s, want %s", body, test.golden)
			}
		})
	}
}

func TestOpenAITransportErrorTypesAreCanonical(t *testing.T) {
	types := map[llmprotocol.ErrorCategory]string{
		llmprotocol.ErrorInvalidRequest:      "invalid_request_error",
		llmprotocol.ErrorAuthentication:      "authentication_error",
		llmprotocol.ErrorPermission:          "permission_error",
		llmprotocol.ErrorNotFound:            "invalid_request_error",
		llmprotocol.ErrorConflict:            "invalid_request_error",
		llmprotocol.ErrorUnsupportedFeature:  "invalid_request_error",
		llmprotocol.ErrorRateLimited:         "rate_limit_error",
		llmprotocol.ErrorUpstreamUnavailable: "server_error",
		llmprotocol.ErrorUpstreamTimeout:     "server_error",
		llmprotocol.ErrorInternal:            "server_error",
	}
	engine := NewBuiltinEngine()
	for category, expectedType := range types {
		for _, format := range []llmprotocol.WireFormat{
			llmprotocol.OpenAIChatV1,
			llmprotocol.OpenAIResponsesV1,
		} {
			t.Run(string(format)+"/"+string(category), func(t *testing.T) {
				body, err := engine.EncodeTransportError(format, llmprotocol.TransportError{
					Error: llmprotocol.NewError(category, "provider_code", "request failed", nil),
				})
				if err != nil {
					t.Fatal(err)
				}
				var wire struct {
					Error struct {
						Type string `json:"type"`
					} `json:"error"`
				}
				if err := json.Unmarshal(body, &wire); err != nil || wire.Error.Type != expectedType {
					t.Fatalf("OpenAI error type = %q/%v, want %q; body=%s", wire.Error.Type, err, expectedType, body)
				}
			})
		}
	}
}

func TestAnthropicTransportErrorTypesAreCanonical(t *testing.T) {
	types := map[llmprotocol.ErrorCategory]string{
		llmprotocol.ErrorInvalidRequest:      "invalid_request_error",
		llmprotocol.ErrorAuthentication:      "authentication_error",
		llmprotocol.ErrorPermission:          "permission_error",
		llmprotocol.ErrorNotFound:            "not_found_error",
		llmprotocol.ErrorConflict:            "conflict_error",
		llmprotocol.ErrorUnsupportedFeature:  "invalid_request_error",
		llmprotocol.ErrorRateLimited:         "rate_limit_error",
		llmprotocol.ErrorUpstreamUnavailable: "api_error",
		llmprotocol.ErrorUpstreamTimeout:     "timeout_error",
		llmprotocol.ErrorInternal:            "api_error",
	}
	engine := NewBuiltinEngine()
	for category, expectedType := range types {
		t.Run(string(category), func(t *testing.T) {
			body, err := engine.EncodeTransportError(
				llmprotocol.AnthropicMessagesV1,
				llmprotocol.TransportError{
					Error: llmprotocol.NewError(category, "provider_code", "request failed", nil),
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			golden := `{"type":"error","error":{"type":"` + expectedType + `","message":"request failed"}}`
			if string(body) != golden {
				t.Fatalf("Anthropic transport error = %s, want %s", body, golden)
			}
		})
	}
}

func TestModelFailureEncodingKeepsResponsesResourceSemantics(t *testing.T) {
	response := llmprotocol.Response{
		Generation: 1, ID: "response_1", Model: "public-model", StopReason: llmprotocol.StopError,
		Error: &llmprotocol.ProtocolError{
			Category: llmprotocol.ErrorAuthentication,
			Code:     "authentication_error", Message: "API key is invalid.",
		},
	}
	assertChatModelFailure(t, response)
	assertResponsesModelFailure(t, response)
	assertAnthropicModelFailure(t, response)
}

func assertChatModelFailure(t *testing.T, response llmprotocol.Response) {
	t.Helper()
	body, _, err := (OpenAIChatCodec{}).EncodeResponse(response, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("encode Chat model failure: %v", err)
	}
	want := `{"error":{"type":"authentication_error","code":"authentication_error","message":"API key is invalid.","param":null}}`
	if string(body) != want {
		t.Fatalf("Chat model failure = %s, want %s", body, want)
	}
}

func assertResponsesModelFailure(t *testing.T, response llmprotocol.Response) {
	t.Helper()
	body, _, err := (OpenAIResponsesCodec{}).EncodeResponse(response, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("encode Responses model failure: %v", err)
	}
	decoded, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatalf("encoded Responses model failure violates its own wire contract: %v\n%s", err, body)
	}
	if decoded.ID != response.ID || decoded.Model != response.Model || decoded.StopReason != llmprotocol.StopError ||
		decoded.Error == nil || decoded.Error.Category != llmprotocol.ErrorAuthentication ||
		decoded.Error.Code != response.Error.Code || decoded.Error.Message != response.Error.Message {
		t.Fatalf("Responses model failure changed semantics: %+v\n%s", decoded, body)
	}
}

func assertAnthropicModelFailure(t *testing.T, response llmprotocol.Response) {
	t.Helper()
	body, _, err := (AnthropicMessagesCodec{}).EncodeResponse(response, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("encode Anthropic model failure: %v", err)
	}
	want := `{"type":"error","error":{"type":"authentication_error","message":"API key is invalid."}}`
	if string(body) != want {
		t.Fatalf("Anthropic model failure = %s, want %s", body, want)
	}
}

func assertAnthropicAuthenticationError(t *testing.T, protocolError *llmprotocol.ProtocolError) {
	t.Helper()
	if protocolError == nil {
		t.Fatal("neutral response did not preserve the provider error")
	}
	if protocolError.Category != llmprotocol.ErrorAuthentication ||
		protocolError.Code != "authentication_error" ||
		protocolError.Message != "API key is invalid." {
		t.Fatalf("neutral error = %+v", protocolError)
	}
}
