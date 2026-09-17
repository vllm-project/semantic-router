package protocolcodec

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// OpenAI Chat Completions and Responses share the same non-2xx API error
// envelope. This is intentionally not a Responses resource whose status is
// "failed"; that object describes model-generation failure after acceptance.
type openAITransportErrorWire struct {
	Error *openAITransportErrorDetailWire `json:"error"`
}

type openAITransportErrorDetailWire struct {
	Type    string           `json:"type"`
	Code    *openAIErrorCode `json:"code"`
	Message string           `json:"message"`
	Param   *string          `json:"param"`
}

func decodeOpenAITransportError(
	body []byte,
	policy llmprotocol.Policy,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	var wire openAITransportErrorWire
	if err := decodeProviderWire(body, &wire, policy); err != nil {
		return llmprotocol.TransportError{}, nil, err
	}
	if wire.Error == nil {
		return llmprotocol.TransportError{}, nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"upstream_error_required",
			"upstream transport error body is missing error details",
			nil,
		)
	}
	if err := validateTransportErrorDetails(wire.Error.Type, wire.Error.Message); err != nil {
		return llmprotocol.TransportError{}, nil, err
	}
	code, parameter := "", ""
	if wire.Error.Code != nil {
		code = string(*wire.Error.Code)
	}
	if wire.Error.Param != nil {
		parameter = *wire.Error.Param
	}
	return llmprotocol.TransportError{Error: &llmprotocol.ProtocolError{
		Category:  decodeProviderErrorCategory(wire.Error.Type, code),
		Code:      code,
		Message:   wire.Error.Message,
		Parameter: parameter,
	}}, nil, nil
}

func encodeOpenAITransportError(transportError llmprotocol.TransportError) []byte {
	protocolError := transportError.Error
	if protocolError == nil {
		protocolError = llmprotocol.NewError(llmprotocol.ErrorInternal, "internal", "request failed", nil)
	}
	wire := openAITransportErrorEnvelope(protocolError)
	body, _ := marshalWire(wire)
	return body
}

func openAITransportErrorEnvelope(protocolError *llmprotocol.ProtocolError) openAITransportErrorWire {
	return openAITransportErrorWire{Error: &openAITransportErrorDetailWire{
		Type:    canonicalOpenAIErrorType(protocolError.Category),
		Code:    (*openAIErrorCode)(optionalString(protocolError.Code)),
		Message: protocolError.Message,
		Param:   optionalString(protocolError.Parameter),
	}}
}

func optionalString(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func canonicalOpenAIErrorType(category llmprotocol.ErrorCategory) string {
	switch category {
	case llmprotocol.ErrorInvalidRequest, llmprotocol.ErrorNotFound,
		llmprotocol.ErrorConflict, llmprotocol.ErrorUnsupportedFeature:
		return "invalid_request_error"
	case llmprotocol.ErrorAuthentication:
		return "authentication_error"
	case llmprotocol.ErrorPermission:
		return "permission_error"
	case llmprotocol.ErrorRateLimited:
		return "rate_limit_error"
	default:
		return "server_error"
	}
}

// openAIErrorCode accepts string codes and the integral HTTP error statuses
// emitted by vLLM. The neutral and public contracts always use string codes.
// Arbitrary numbers, booleans and structured values remain malformed errors.
type openAIErrorCode string

func (code *openAIErrorCode) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*code = openAIErrorCode(text)
		return nil
	}
	var status int
	if err := json.Unmarshal(data, &status); err != nil || status < 400 || status > 599 {
		return fmt.Errorf("error code must be a string or an integral HTTP error status")
	}
	*code = openAIErrorCode(strconv.Itoa(status))
	return nil
}
