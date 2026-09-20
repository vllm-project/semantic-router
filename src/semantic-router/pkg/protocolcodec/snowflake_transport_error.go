package protocolcodec

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// snowflakeTransportErrorWire is Snowflake Cortex AI's failure envelope: a flat
// object carrying the vendor's own string code and the request ID the request
// was refused under. The canonical OpenAI error object cannot represent it —
// the canonical field names do not match — so a strict decode fails before the
// provider's message can be read.
type snowflakeTransportErrorWire struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
	ErrorCode string `json:"error_code"`
}

// decodeSnowflakeTransportError decodes the Snowflake Cortex failure envelope at
// the transport edge, preserving the vendor's own code and message. It decodes
// through the same provider JSON validation the other transport decoders use, so
// a body over the policy limit, carrying duplicate fields, or malformed Unicode
// is rejekted here instead of being accepted as a vendor envelope. A body
// without a flat code/message pair falls back to the canonical OpenAI decoder,
// so an unexpected shape still reports the same typed error instead of a silent
// success.
func decodeSnowflakeTransportError(
	body []byte,
	policy llmprotocol.Policy,
	format llmprotocol.WireFormat,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	var wire snowflakeTransportErrorWire
	if _, _, err := decodeProviderWireVendorAware(body, &wire, policy); err != nil ||
		strings.TrimSpace(wire.Code) == "" || strings.TrimSpace(wire.Message) == "" {
		return decodeOpenAITransportError(body, policy, format)
	}
	return llmprotocol.TransportError{Error: &llmprotocol.ProtocolError{
		Category: snowflakeErrorCategory(wire.Code),
		Code:     strings.TrimSpace(wire.Code),
		Message:  wire.Message,
	}}, nil, nil
}

// snowflakeErrorCategory maps the vendor code observed on this account to the
// neutral categories. 003001 is the account-level refusal ("This account is not
// allowed to access this endpoint. Please contact Snowflake support."), i.e. a
// permission failure, not a malformed request. Codes outside that observation
// keep the neutral unavailability category and still carry their own code and
// message to the client.
func snowflakeErrorCategory(code string) llmprotocol.ErrorCategory {
	if strings.TrimSpace(code) == "003001" {
		return llmprotocol.ErrorPermission
	}
	return llmprotocol.ErrorUpstreamUnavailable
}
