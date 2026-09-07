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

// blankText reports whether s is empty or whitespace-only.
func blankText(s string) bool {
	return strings.TrimSpace(s) == ""
}
