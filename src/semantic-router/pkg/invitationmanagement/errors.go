package invitationmanagement

import "errors"

var (
	ErrInvalidRequest    = errors.New("invitation request is invalid")
	ErrNotFound          = errors.New("invitation was not found")
	ErrConflict          = errors.New("invitation state conflicts with the request")
	ErrRevisionConflict  = errors.New("invitation revision changed")
	ErrExpired           = errors.New("invitation has expired")
	ErrAlreadyAccepted   = errors.New("identity is already onboarded")
	ErrIdentityMismatch  = errors.New("invitation identity does not match")
	ErrSecretExpired     = errors.New("invitation secret result has expired")
	ErrDelegationDenied  = errors.New("invitation role delegation is not authorized")
	ErrDefaultsChanged   = errors.New("invitation onboarding defaults changed")
	ErrUnavailable       = errors.New("invitation management is unavailable")
	ErrInvalidToken      = errors.New("invitation token is invalid")
	ErrPepperUnavailable = errors.New("invitation token pepper is unavailable")
)
