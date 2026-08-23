package management

import "errors"

var (
	ErrInvalidRequest   = errors.New("provider credential request is invalid")
	ErrProviderMismatch = errors.New("provider credential binding does not match the active Provider")
	ErrUnsafeOrigin     = errors.New("provider credential origin is denied")
	ErrUnavailable      = errors.New("provider credential Management service is unavailable")
)
