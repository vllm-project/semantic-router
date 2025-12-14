package responsestore

import "errors"

// Common errors returned by ResponseStore implementations.
var (
	// ErrNotFound is returned when a requested item doesn't exist.
	ErrNotFound = errors.New("item not found")

	// ErrAlreadyExists is returned when trying to create an item that already exists.
	ErrAlreadyExists = errors.New("item already exists")

	// ErrStoreDisabled is returned when the store is not enabled.
	ErrStoreDisabled = errors.New("store is disabled")

	// ErrInvalidID is returned when an ID is invalid or malformed.
	ErrInvalidID = errors.New("invalid ID format")

	// ErrConnectionFailed is returned when the store connection fails.
	ErrConnectionFailed = errors.New("store connection failed")

	// ErrStoreFull is returned when the store has reached its capacity.
	ErrStoreFull = errors.New("store is full")

	// ErrInvalidInput is returned when input validation fails.
	ErrInvalidInput = errors.New("invalid input")

	// ErrConversationNotEmpty is returned when trying to delete a non-empty conversation.
	ErrConversationNotEmpty = errors.New("conversation is not empty")
)
