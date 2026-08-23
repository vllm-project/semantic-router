package subjectmanagement

import "errors"

var (
	ErrInvalidRequest             = errors.New("subject Management request is invalid")
	ErrNotFound                   = errors.New("subject Management resource not found")
	ErrAlreadyExists              = errors.New("subject Management resource already exists")
	ErrRevisionConflict           = errors.New("subject Management revision conflict")
	ErrDefaultsUnavailable        = errors.New("Team default policies are unavailable")
	ErrPolicySelectionUnavailable = errors.New("Team policy selection is unavailable")
	ErrUnavailable                = errors.New("subject Management service is unavailable")
)
