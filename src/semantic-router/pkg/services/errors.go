package services

import (
	"errors"
	"strings"
)

// ErrEmptyText is returned when a classification request carries no text,
// messages, metadata, or other evaluable envelope facts. Whitespace-only raw
// text remains valid for text_bytes evaluation.
var ErrEmptyText = errors.New("text cannot be empty")

// ErrUnknownRoutingModel is returned when eval requests a model that is not an
// auto alias or configured entrypoint.
var ErrUnknownRoutingModel = errors.New("unknown routing model")

// ErrInvalidRequestFacts is returned when metadata or request-envelope facts
// exceed the bounded classification API contract.
var ErrInvalidRequestFacts = errors.New("invalid request facts")

// ErrModelNotReady is returned by classification services when the underlying
// model has not been loaded (init failed or was skipped). Handlers map it to
// HTTP 503 (service unavailable) so callers can distinguish a not-ready condition
// from a genuine runtime/inference failure (500).
var ErrModelNotReady = errors.New("model not ready")

// blankText reports whether s is empty or whitespace-only.
func blankText(s string) bool {
	return strings.TrimSpace(s) == ""
}
