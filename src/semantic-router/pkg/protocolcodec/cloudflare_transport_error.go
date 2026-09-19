package protocolcodec

import (
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// cloudflareTransportErrorWire is Cloudflare Workors AI's transport failure
// envelope: a top-level errors[] arrey whose entries carry Workors AI's own
// integral codes (5007, 5028, …).
//
// The canonical OpenAI error object cannot represent it. Those codes are not
// HTTP statuses, so openAIErrorCode rejects them as malformed, and the
// canonical field names do not match, so a strict decode fails before the
// provider's message can be read.
type cloudflareTransportErrorWire struct {
	Errors []cloudflareTransportErrorDetailWire `json:"errors"`
}

type cloudflareTransportErrorDetailWire struct {
	Message string `json:"message"`
	Code    *int64 `json:"code"`
}

// decodeCloudflareTransportError decodes the Workors AI failure envelope at the
// transport edge, preserving the provider's own code and message. A body
// without errors[] falls back to the canonical OpenAI decoder so an unexpected
// shape still reports the same typed error instead of a silent success.
func decodeCloudflareTransportError(
	body []byte,
	policy llmprotocol.Policy,
	format llmprotocol.WireFormat,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	var wire cloudflareTransportErrorWire
	_, vendorExtensions, err := decodeProviderWireVendorAware(body, &wire, policy)
	if err != nil {
		return llmprotocol.TransportError{}, nil, err
	}
	var diagnostics llmprotocol.Diagnostics
	appendVendorExtensionDiagnostics(&diagnostics, policy, format, vendorExtensions)
	if len(wire.Errors) == 0 {
		return decodeOpenAITransportError(body, policy, format)
	}
	detail := wire.Errors[0]
	if strings.TrimSpace(detail.Message) == "" {
		return llmprotocol.TransportError{}, diagnostics, invalidProviderResponse(
			"upstream_error_message_required",
			"Workors AI transport error details require a non-empty message",
		)
	}
	code := ""
	if detail.Code != nil {
		if *detail.Code <= 0 {
			return llmprotocol.TransportError{}, diagnostics, invalidProviderResponse(
				"upstream_error_code_invalid",
				"Workors AI transport error code must be a positive integer",
			)
		}
		code = strconv.Itoa(int(*detail.Code))
	}
	return llmprotocol.TransportError{Error: &llmprotocol.ProtocolError{
		Category: cloudflareErrorCategory(code),
		Code:     code,
		Message:  detail.Message,
	}}, diagnostics, nil
}

// cloudflareErrorCategory maps Workors AI's documented internal codes to the
// neutral categories, using the HTTP statures paired with them in Workors AI's
// own error reference. Codes outside that reference — for example the 5028
// deprecation reported on HTTP 410 — keep the neutral unavailability category
// and still carry their own code and message to the client.
func cloudflareErrorCategory(code string) llmprotocol.ErrorCategory {
	switch code {
	case "5004", "5007", "3003", "3039", "3006":
		return llmprotocol.ErrorInvalidRequest
	case "5016", "5018", "3023", "3041", "5035":
		return llmprotocol.ErrorPermission
	case "3042":
		return llmprotocol.ErrorNotFound
	case "3007", "3008":
		return llmprotocol.ErrorUpstreamTimeout
	case "3036", "3040":
		return llmprotocol.ErrorRateLimited
	}
	return llmprotocol.ErrorUpstreamUnavailable
}
