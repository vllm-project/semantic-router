package namespacemanagement

import "errors"

var (
	ErrInvalidRequest      = errors.New("Namespace Management request is invalid")
	ErrIdempotencyConflict = errors.New("Namespace Management idempotency conflict")
	ErrNotFound            = errors.New("Namespace was not found")
	ErrAlreadyExists       = errors.New("Namespace already exists")
	ErrRevisionConflict    = errors.New("Namespace resource revision conflict")
	ErrDependency          = errors.New("Namespace still has active resources")
	ErrAssurance           = errors.New("management authentication assurance is insufficient")
	ErrUnavailable         = errors.New("Namespace Management is unavailable")
)
