package protocolcodec

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// openRouterTransportErrorWire is OpenRouter's failure envelope: the canonical
// OpenAI error object without a type, extended with a provider metadata object
// (the upstream provider's raw error and name) and a top-level user_id. The
// canonical wire rejects both extensions and requires a type, so a strict decode
// fails before the message and HTTP-status code can be read.
type openRouterTransportErrorWire struct {
	Error  *openRouterTransportErrorDetailWire `json:"error"`
	UserID string                              `json:"user_id,omitempty"`
}

type openRouterTransportErrorDetailWire struct {
	Type     string           `json:"type,omitempty"`
	Code     *openAIErrorCode `json:"code"`
	Message  string           `json:"message"`
	Param    *string          `json:"param,omitempty"`
	Metadata json.RawMessage  `json:"metadata,omitempty"`
}

// decodeOpenRouterTransportError decodes the OpenRouter failure envelope at the
// transport edge. It uses the same provider JSON validation as the canonical
// decoder, so limits, duplicate fields, and malformed Unicode are still
// rejected. The metadata object is validated as a JSON object but not forwarded,
// because it carries the upstream provider's raw error text. A body without a
// message falls back to the canonical OpenAI decoder, so an unexpected shape
// still reports the same typed error.
func decodeOpenRouterTransportError(
	body []byte,
	policy llmprotocol.Policy,
	format llmprotocol.WireFormat,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	var wire openRouterTransportErrorWire
	if _, _, err := decodeProviderWireVendorAware(body, &wire, policy); err != nil ||
		wire.Error == nil || strings.TrimSpace(wire.Error.Message) == "" ||
		!openRouterMetadataIsObject(wire.Error.Metadata) {
		return decodeOpenAITransportError(body, policy, format)
	}
	detail := wire.Error
	code, parameter := "", ""
	if detail.Code != nil {
		code = string(*detail.Code)
	}
	if detail.Param != nil {
		parameter = *detail.Param
	}
	category := decodeProviderErrorCategory(detail.Type)
	if strings.TrimSpace(detail.Type) == "" {
		category = openRouterErrorCategory(code)
	}
	return llmprotocol.TransportError{Error: &llmprotocol.ProtocolError{
		Category:  category,
		Code:      code,
		Message:   detail.Message,
		Parameter: parameter,
	}}, nil, nil
}

func openRouterMetadataIsObject(metadata json.RawMessage) bool {
	if len(metadata) == 0 {
		return true
	}
	var object map[string]json.RawMessage
	return json.Unmarshal(metadata, &object) == nil && object != nil
}

// openRouterErrorCategory maps the HTTP status OpenRouter carries in code to the
// neutral categories. Statuses without a clear neutral meaning keep the
// unavailability category and still carry their own code and message.
func openRouterErrorCategory(code string) llmprotocol.ErrorCategory {
	status, err := strconv.Atoi(strings.TrimSpace(code))
	if err != nil {
		return llmprotocol.ErrorUpstreamUnavailable
	}
	switch status {
	case 400:
		return llmprotocol.ErrorInvalidRequest
	case 401:
		return llmprotocol.ErrorAuthentication
	case 403:
		return llmprotocol.ErrorPermission
	case 404:
		return llmprotocol.ErrorNotFound
	case 408:
		return llmprotocol.ErrorUpstreamTimeout
	case 429:
		return llmprotocol.ErrorRateLimited
	}
	return llmprotocol.ErrorUpstreamUnavailable
}
