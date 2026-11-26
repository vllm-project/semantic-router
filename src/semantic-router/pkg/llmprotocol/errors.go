package llmprotocol

import "fmt"

type ErrorCategory string

const (
	ErrorInvalidRequest      ErrorCategory = "invalid_request"
	ErrorAuthentication      ErrorCategory = "authentication"
	ErrorPermission          ErrorCategory = "permission"
	ErrorNotFound            ErrorCategory = "not_found"
	ErrorConflict            ErrorCategory = "conflict"
	ErrorUnsupportedFeature  ErrorCategory = "unsupported_feature"
	ErrorRateLimited         ErrorCategory = "rate_limited"
	ErrorUpstreamUnavailable ErrorCategory = "upstream_unavailable"
	ErrorUpstreamTimeout     ErrorCategory = "upstream_timeout"
	ErrorInternal            ErrorCategory = "internal"
)

type ProtocolError struct {
	Category  ErrorCategory
	Code      string
	Message   string
	Parameter string
	// ProviderFramed reports that Code, Message or Parameter was copied from a
	// provider's error payload rather than authored by the Router. Those fields
	// can echo request content, so diagnostic logs name only Category for such
	// an error; the client still reads the fields it always read.
	ProviderFramed bool
	RetryAfter     int64
	Cause          error
}

// TransportError is the protocol-neutral representation of an HTTP error
// response. It is deliberately separate from Response.Error: the latter is a
// model-generation result (for example, a Responses resource whose status is
// "failed"), while TransportError represents a non-2xx API response.
type TransportError struct {
	Error             *ProtocolError
	ProviderRequestID string
}

// ErrorDiagnostic is a Router-authored explanation attached to a ProtocolError's
// cause. Its text is built from Router constants and counts, never from a
// provider payload, so it is the only cause an operator log may print: a JSON
// or transport error can echo provider input in its message.
type ErrorDiagnostic string

func (diagnostic ErrorDiagnostic) Error() string { return string(diagnostic) }

func (err *ProtocolError) Error() string {
	if err == nil {
		return ""
	}
	if err.Code != "" {
		return fmt.Sprintf("%s: %s", err.Code, err.Message)
	}
	return err.Message
}

func (err *ProtocolError) Unwrap() error {
	if err == nil {
		return nil
	}
	return err.Cause
}

func NewError(category ErrorCategory, code, message string, cause error) *ProtocolError {
	return &ProtocolError{Category: category, Code: code, Message: message, Cause: cause}
}
