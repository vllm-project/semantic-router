package routing

import "errors"

// ErrUnsupported reports a routing-core reply that this contract cannot
// express, so no gateway applies part of it silently.
var ErrUnsupported = errors.New("unsupported by the routing contract")

// Phase names one step of the routing pipeline.
type Phase string

const (
	PhaseRequestHeaders  Phase = "request_headers"
	PhaseRequestBody     Phase = "request_body"
	PhaseResponseHeaders Phase = "response_headers"
	PhaseResponseBody    Phase = "response_body"
)

// BodyMode is how a body reaches the routing core.
type BodyMode string

const (
	// BodyModeBuffered delivers the whole body in one phase call; the
	// headers are held until that call returns, so it may still change them.
	BodyModeBuffered BodyMode = "buffered"
	// BodyModeStreamed delivers the body chunk by chunk after the headers
	// have been forwarded.
	BodyModeStreamed BodyMode = "streamed"
)

// Effect is what the routing core asks the gateway to do after one phase.
type Effect struct {
	// Header mutates the headers of the message being processed. A non-nil
	// empty mutation still counts as present for ClearRouteCache.
	Header *HeaderMutation `json:"header,omitempty"`
	// Body mutates the body of the current body phase. Header phases carry
	// no body mutation the gateway applies.
	Body *BodyMutation `json:"body,omitempty"`
	// ClearRouteCache asks the gateway to choose the route again from the
	// mutated request headers. It takes effect only with a Header mutation.
	ClearRouteCache bool `json:"clear_route_cache,omitempty"`
	// ResponseBodyMode, set on a response-headers effect, switches the
	// response body to streamed processing.
	ResponseBodyMode BodyMode `json:"response_body_mode,omitempty"`
	// Immediate answers the client instead of continuing the request.
	Immediate *ImmediateResponse `json:"immediate,omitempty"`
}

// HeaderMutation removes headers, then sets headers in order.
type HeaderMutation struct {
	Set    []HeaderOption `json:"set,omitempty"`
	Remove []string       `json:"remove,omitempty"`
}

// HeaderOption sets one header. Without Append it replaces every existing
// value; with Append it adds another value when the header already exists.
type HeaderOption struct {
	Name   string `json:"name"`
	Value  string `json:"value"`
	Append bool   `json:"append,omitempty"`
}

// BodyMutation clears the body or replaces it with Body.
type BodyMutation struct {
	Clear bool   `json:"clear,omitempty"`
	Body  []byte `json:"body,omitempty"`
}

// ImmediateResponse answers the client without the upstream, or instead of
// the upstream's response.
type ImmediateResponse struct {
	Status int             `json:"status"`
	Header *HeaderMutation `json:"header,omitempty"`
	Body   []byte          `json:"body,omitempty"`
	// Details explains the response for access logs; clients don't see it.
	Details string `json:"details,omitempty"`
}
