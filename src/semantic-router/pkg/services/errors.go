package services

import (
	"errors"
	"strings"
)

// ErrEmptyText is returned by classification services when the request text is
// empty or whitespace-only. Handlers map it to HTTP 400 (client error) rather
// than 500, matching the documented OpenAPI contract and sibling endpoints.
var ErrEmptyText = errors.New("text cannot be empty")

// ErrInvalidRequestFacts is returned when request metadata or other extracted
// facts violate the classification request contract. Handlers map it to HTTP
// 400 (client error).
var ErrInvalidRequestFacts = errors.New("invalid request facts")

// ErrUnknownRoutingModel is returned when a request names no configured
// routing entrypoint. Handlers map it to HTTP 400 (client error).
var ErrUnknownRoutingModel = errors.New("unknown routing model")

// ErrModelNotReady is returned by classification services when the underlying
// model has not been loaded (init failed or was skipped). Handlers map it to
// HTTP 503 (service unavailable) so callers can distinguish a not-ready condition
// from a genuine runtime/inference failure (500).
var ErrModelNotReady = errors.New("model not ready")

// blankText reports whether s is empty or whitespace-only.
func blankText(s string) bool {
	return strings.TrimSpace(s) == ""
}
