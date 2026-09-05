package services

import (
	"errors"
	"strings"
)

// ErrEmptyText is returned by classification services when the request text is
// empty or whitespace-only. Handlers map it to HTTP 400 (client error) rather
// than 500, matching the documented OpenAPI contract and sibling endpoints.
var ErrEmptyText = errors.New("text cannot be empty")

// ErrModelNotReady is returned by classification services when the underlying
// model has not been loaded (init failed or was skipped). Handlers map it to
// HTTP 503 (service unavailable) so callers can distinguish a not-ready condition
// from a genuine runtime/inference failure (500).
var ErrModelNotReady = errors.New("model not ready")

// blankText reports whether s is empty or whitespace-only.
func blankText(s string) bool {
	return strings.TrimSpace(s) == ""
}
