package services

import (
	"errors"
	"strings"
)

// ErrEmptyText is returned by classification services when the request text is
// empty or whitespace-only. Handlers map it to HTTP 400 (client error) rather
// than 500, matching the documented OpenAPI contract and sibling endpoints.
var ErrEmptyText = errors.New("text cannot be empty")

// blankText reports whether s is empty or whitespace-only.
func blankText(s string) bool {
	return strings.TrimSpace(s) == ""
}
