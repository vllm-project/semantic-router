package apikeymanagement

import "errors"

var (
	ErrInvalidRequest        = errors.New("API-key Management request is invalid")
	ErrNotFound              = errors.New("API-key resource not found")
	ErrUnavailable           = errors.New("API-key Management service is unavailable")
	ErrSecretResultExpired   = errors.New("API-key secret result has expired")
	ErrRevealDisabled        = errors.New("API-key reveal is disabled")
	ErrCredentialUnavailable = errors.New("API-key credential is unavailable")
	ErrLastActiveCredential  = errors.New("the final active credential cannot be revoked while the key is enabled")
	ErrRevisionConflict      = errors.New("API-key revision conflict")
)
