package accessmanagement

import "errors"

var (
	ErrInvalidRequest   = errors.New("invalid access management request")
	ErrNotFound         = errors.New("access management resource not found")
	ErrRevisionConflict = errors.New("access management revision conflict")
	ErrUnavailable      = errors.New("access management state unavailable")
)
