package delegationmanagement

import "errors"

var (
	ErrInvalidRequest      = errors.New("delegated inference session request is invalid")
	ErrNotFound            = errors.New("delegated inference session not found")
	ErrUnavailable         = errors.New("delegated inference session service is unavailable")
	ErrNotEligible         = errors.New("inference key is not eligible for delegation")
	ErrSessionLimit        = errors.New("maximum delegated inference sessions reached")
	ErrSecretResultExpired = errors.New("delegated inference session secret result has expired")
	ErrCredentialInactive  = errors.New("delegated inference credential is inactive")
)
