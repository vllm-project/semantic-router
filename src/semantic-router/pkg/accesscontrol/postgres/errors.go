package postgres

import "errors"

var (
	// ErrNotFound is returned for a namespace-scoped read with no matching row.
	ErrNotFound = errors.New("access-control resource not found")
	// ErrAlreadyExists identifies a uniqueness collision on a create command.
	ErrAlreadyExists = errors.New("access-control resource already exists")
	// ErrRevisionConflict covers both a stale expected revision and a missing
	// mutation target, avoiding an extra existence probe around CAS writes.
	ErrRevisionConflict = errors.New("access-control revision conflict")
)
