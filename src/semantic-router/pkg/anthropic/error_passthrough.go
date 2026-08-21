// Error passthrough between the Anthropic and OpenAI wire formats.
//
// Upstream error bodies must survive response translation in both
// directions: an Anthropic error envelope ({"type":"error","error":{...}})
// is preserved as an OpenAI-shape error body for OpenAI-protocol clients
// (consumed by toOpenAIResponseBody in client.go), and an OpenAI-shape
// error body is re-emitted as the Anthropic envelope for Anthropic-protocol
// clients (consumed by EmitAnthropicResponse in outbound.go). Without these
// guards both directions flatten error bodies into empty success-shaped
// responses, destroying the upstream failure reason.
package anthropic

import (
	"encoding/json"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

// openAIErrorEnvelope is the OpenAI error wire shape ({"error":{...}}).
// anthropicErrorToOpenAIBody marshals it when preserving an Anthropic error
// envelope for OpenAI-protocol clients, and openAIErrorBody unmarshals it
// to detect error bodies before the Anthropic emit.
// Deliberately not using openai-go's shared.ErrorObject, because the SDK
// type's decoder is lenient and would decode a successful response with an
// `"error": false` property as an ErrorObject.
type openAIErrorEnvelope struct {
	Error *openAIErrorDetail `json:"error"`
}

type openAIErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// anthropicErrorToOpenAIBody converts an Anthropic error envelope
// ({"type":"error","error":{...}}) into the OpenAI error wire shape
// ({"error":{"message":...,"type":...}}) so the upstream failure reason
// survives translation for OpenAI-protocol clients. The envelope's
// request_id is captured onto the sidecar (not the body) for the Anthropic
// re-emit; OpenAI clients get it via the request-id response header.
// Returns ok=false for non-error bodies.
//
// This is the entry half of a two-stage relay: the OpenAI error body it
// produces travels through the router's OpenAI-shaped pipeline, and — when
// the client speaks the Anthropic protocol — openAIErrorBody recognizes it
// on the way out so EmitAnthropicResponse can re-wrap it as the envelope.
func anthropicErrorToOpenAIBody(anthropicResponse []byte, ext *ir.IRExtensions) ([]byte, bool) {
	var probe anthropic.ErrorResponse
	if err := json.Unmarshal(anthropicResponse, &probe); err != nil || probe.Type != constant.ValueOf[constant.Error]() {
		return nil, false
	}
	if ext != nil {
		ext.AnthropicErrorRequestID = probe.RequestID
	}
	body, err := json.Marshal(openAIErrorEnvelope{
		Error: &openAIErrorDetail{
			Type:    probe.Error.Type,
			Message: probe.Error.Message,
		},
	})
	if err != nil {
		return nil, false
	}
	return body, true
}

// openAIErrorBody reports whether responseBody is an OpenAI-shape error
// ({"error":{...}}) and returns the parsed envelope. Success bodies (and
// bodies where "error" is null or a non-object) return ok=false.
//
// This is the exit half of the relay: it recognizes error bodies in the
// pipeline's OpenAI shape — whether produced by anthropicErrorToOpenAIBody
// or returned natively by an OpenAI-format backend — so the Anthropic
// emitter re-wraps them instead of flattening them into an empty message.
func openAIErrorBody(responseBody []byte) (*openAIErrorEnvelope, bool) {
	var probe openAIErrorEnvelope
	if err := json.Unmarshal(responseBody, &probe); err != nil || probe.Error == nil {
		return nil, false
	}
	return &probe, true
}

// anthropicErrorType maps an arbitrary error-type token to one of
// Anthropic's defined error types. Anthropic's own tokens pass through
// unchanged (the Anthropic-to-Anthropic round-trip case); known
// OpenAI-only tokens map to their closest Anthropic equivalent; anything
// else — including an absent type — falls back to the catch-all
// "api_error".
func anthropicErrorType(token string) string {
	switch token {
	case "invalid_request_error", "authentication_error", "billing_error",
		"permission_error", "not_found_error", "request_too_large",
		"rate_limit_error", "timeout_error", "api_error", "overloaded_error":
		return token
	case "rate_limit_exceeded":
		return "rate_limit_error"
	case "insufficient_quota":
		return "billing_error"
	default:
		return anthropicErrorTypeAPI
	}
}
