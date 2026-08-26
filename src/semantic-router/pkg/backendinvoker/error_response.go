package backendinvoker

import (
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const maximumPublicProviderRequestIDBytes = 512

const maximumPublicProviderErrorMessageBytes = 2048

const maximumProviderErrorBodyBytes = 64 << 10

const maximumPublicProviderErrorCodeBytes = 256

const maximumPublicProviderErrorParameterBytes = 256

var providerCredentialPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\b(?:bearer|basic)\s+[^\s,;]+`),
	regexp.MustCompile(`(?i)\b(?:sk|rk|pk|api|key|token|secret)[-_][a-z0-9._-]{4,}`),
	regexp.MustCompile(`(?i)\b(?:api[_ -]?key|access[_ -]?token|authorization|credential|password|secret)\s*(?:=|:)\s*[^\s,;]+`),
	regexp.MustCompile(`(?i)\b(?:token|secret|password|api_key)=[^\s&]+`),
	regexp.MustCompile(`-----BEGIN [A-Z0-9 ]+PRIVATE KEY-----`),
	regexp.MustCompile(`[A-Za-z0-9+/_=-]{32,}`),
}

// translateUpstreamHTTPError keeps the HTTP status authoritative while using
// the neutral protocol pipeline to preserve a supported provider's public
// error code and message. Malformed, oversized, or unsupported provider bodies
// fail closed to the generic status-derived error.
func translateUpstreamHTTPError(
	engine *protocolcodec.Engine,
	providerFormat llmprotocol.WireFormat,
	clientFormat llmprotocol.WireFormat,
	status int,
	headers http.Header,
	body []byte,
	readErr error,
	sensitiveValues []string,
) (*llmprotocol.ProtocolError, []byte, string) {
	fallback := safeUpstreamHTTPError(status, headers)
	if readErr == nil {
		translated, err := engine.TranslateTransportError(
			providerFormat,
			clientFormat,
			body,
			func(transportError *llmprotocol.TransportError) error {
				if transportError.Error == nil {
					return fallback
				}
				transportError.ProviderRequestID = canonicalProviderRequestID(
					headers, transportError.ProviderRequestID, sensitiveValues,
				)
				transportError.Error = publicProviderProtocolError(transportError.Error, fallback, sensitiveValues)
				return nil
			},
		)
		if err == nil && translated.TransportError.Error != nil {
			return translated.TransportError.Error, translated.Body, translated.TransportError.ProviderRequestID
		}
	}
	providerRequestID := canonicalProviderRequestID(headers, "", sensitiveValues)
	encoded, err := engine.EncodeTransportError(clientFormat, llmprotocol.TransportError{
		Error: fallback, ProviderRequestID: providerRequestID,
	})
	if err != nil {
		// The caller validated both formats while building the invocation plan,
		// so this is only a defensive fallback for direct unit construction.
		return fallback, []byte(`{"error":{"message":"request failed"}}`), ""
	}
	return fallback, encoded, providerRequestID
}

func publicProviderProtocolError(
	providerError *llmprotocol.ProtocolError,
	fallback *llmprotocol.ProtocolError,
	sensitiveValues []string,
) *llmprotocol.ProtocolError {
	result := &llmprotocol.ProtocolError{
		Category:   fallback.Category,
		Code:       strings.TrimSpace(providerError.Code),
		Message:    strings.TrimSpace(providerError.Message),
		Parameter:  strings.TrimSpace(providerError.Parameter),
		RetryAfter: fallback.RetryAfter,
	}
	if !safePublicProviderErrorIdentity(result.Code, maximumPublicProviderErrorCodeBytes, sensitiveValues) {
		result.Code = fallback.Code
	}
	if result.Parameter != "" &&
		!safePublicProviderErrorIdentity(result.Parameter, maximumPublicProviderErrorParameterBytes, sensitiveValues) {
		result.Parameter = ""
	}
	if !safePublicProviderErrorMessage(result.Message, sensitiveValues) {
		result.Message = fallback.Message
	}
	return result
}

func safePublicProviderErrorIdentity(value string, limit int, sensitiveValues []string) bool {
	if value == "" || len(value) > limit || !utf8.ValidString(value) ||
		containsCredentialMaterial(value) || containsSensitiveValue(value, sensitiveValues) {
		return false
	}
	for _, character := range value {
		if unicode.IsControl(character) {
			return false
		}
	}
	return true
}

func safePublicProviderErrorMessage(message string, sensitiveValues []string) bool {
	if message == "" || len(message) > maximumPublicProviderErrorMessageBytes || !utf8.ValidString(message) ||
		containsCredentialMaterial(message) || containsSensitiveValue(message, sensitiveValues) {
		return false
	}
	for _, value := range message {
		if unicode.IsControl(value) {
			return false
		}
	}
	return true
}

func containsSensitiveValue(value string, sensitiveValues []string) bool {
	for _, sensitive := range sensitiveValues {
		sensitive = strings.TrimSpace(sensitive)
		if sensitive != "" && strings.Contains(value, sensitive) {
			return true
		}
	}
	return false
}

func publicStreamErrorMutation(sensitiveValues []string) protocolcodec.StreamEventMutation {
	return func(event *llmprotocol.Event) error {
		if event == nil || event.Type != llmprotocol.EventResponseFailed || event.Error == nil {
			return nil
		}
		event.Error = publicProviderProtocolError(event.Error, safeStreamProtocolError(event.Error.Category), sensitiveValues)
		return nil
	}
}

func safeStreamProtocolError(category llmprotocol.ErrorCategory) *llmprotocol.ProtocolError {
	code, message := "upstream_stream_error", "the selected model stream failed"
	switch category {
	case llmprotocol.ErrorInvalidRequest:
		code, message = "upstream_invalid_request", "the selected model rejected the request"
	case llmprotocol.ErrorAuthentication:
		code, message = "upstream_authentication", "the selected model could not authenticate the request"
	case llmprotocol.ErrorPermission:
		code, message = "upstream_permission", "the selected model denied the request"
	case llmprotocol.ErrorNotFound:
		code, message = "upstream_not_found", "the selected model or endpoint was not found"
	case llmprotocol.ErrorConflict:
		code, message = "upstream_conflict", "the selected model reported a request conflict"
	case llmprotocol.ErrorRateLimited:
		code, message = "upstream_rate_limited", "the selected model is rate limited"
	case llmprotocol.ErrorUpstreamTimeout:
		code, message = "upstream_timeout", "the selected model timed out"
	case llmprotocol.ErrorUpstreamUnavailable:
	default:
		category = llmprotocol.ErrorUpstreamUnavailable
	}
	return llmprotocol.NewError(category, code, message, nil)
}

func containsCredentialMaterial(value string) bool {
	for _, pattern := range providerCredentialPatterns {
		if pattern.MatchString(value) {
			return true
		}
	}
	return false
}

func publicUpstreamErrorHeaders(
	bodyLength int,
	protocolError *llmprotocol.ProtocolError,
	providerRequestID string,
	sensitiveValues []string,
) http.Header {
	result := make(http.Header)
	result.Set("Content-Length", strconv.Itoa(bodyLength))
	result.Set("Content-Type", "application/json")
	if protocolError != nil && protocolError.RetryAfter > 0 {
		result.Set("Retry-After", strconv.FormatInt(protocolError.RetryAfter, 10))
	}
	if requestID := publicProviderRequestID(providerRequestID, sensitiveValues); requestID != "" {
		result.Set("Request-Id", requestID)
		result.Set("X-Request-Id", requestID)
	}
	return result
}

func canonicalProviderRequestID(headers http.Header, bodyValue string, sensitiveValues []string) string {
	for _, candidate := range []string{headers.Get("Request-Id"), headers.Get("X-Request-Id"), bodyValue} {
		if value := publicProviderRequestID(candidate, sensitiveValues); value != "" {
			return value
		}
	}
	return ""
}

func publicProviderRequestID(value string, sensitiveValues []string) string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > maximumPublicProviderRequestIDBytes || strings.ContainsAny(value, "\r\n\x00") ||
		containsCredentialMaterial(value) || containsSensitiveValue(value, sensitiveValues) {
		return ""
	}
	return value
}
