package services

import (
	"errors"
	"strings"
)

// ErrEmptyText is returned by classification services when the request text is
// empty or whitespace-only. Handlers map it to HTTP 400 (client error) rather
// than 500, matching the documented OpenAPI contract and sibling endpoints.
var ErrEmptyText = errors.New("text cannot be empty")

// ErrInvalidImageInput is returned when an inline image selected from an HTTP
// classify/eval request uses an allowlisted data-URI shape but has an empty or
// malformed base64 payload. Keep the error content-free: handlers use this
// sentinel to return a stable 400 response without reflecting request data.
var ErrInvalidImageInput = errors.New("invalid image input")

// blankText reports whether s is empty or whitespace-only.
func blankText(s string) bool {
	return strings.TrimSpace(s) == ""
}
