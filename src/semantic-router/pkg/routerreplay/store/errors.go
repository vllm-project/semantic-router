package store

import "errors"

// Standard errors for replay store operations.
var (
	// ErrNotFound is returned when a record is not found.
	ErrNotFound = errors.New("record not found")

	// ErrAlreadyExists is returned when a record already exists.
	ErrAlreadyExists = errors.New("record already exists")

	// ErrStoreDisabled is returned when the store is disabled.
	ErrStoreDisabled = errors.New("store is disabled")

	// ErrInvalidID is returned when the record ID is invalid.
	ErrInvalidID = errors.New("invalid ID format")

	// ErrConnectionFailed is returned when the store connection fails.
	ErrConnectionFailed = errors.New("store connection failed")

	// ErrStoreFull is returned when the store is full and cannot accept new records.
	ErrStoreFull = errors.New("store is full")

	// ErrInvalidInput is returned when the input is invalid.
	ErrInvalidInput = errors.New("invalid input")
)
