package accesscontrol

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// ErrInvalid marks a domain value that violates an access-control invariant.
var ErrInvalid = errors.New("invalid access-control resource")

// ValidationError identifies one invalid field without coupling the domain to
// an API error representation.
type ValidationError struct {
	Field   string
	Problem string
}

func (e ValidationError) Error() string {
	return fmt.Sprintf("%s: %s", e.Field, e.Problem)
}

func (e ValidationError) Unwrap() error { return ErrInvalid }

func invalid(field, problem string) error {
	return ValidationError{Field: field, Problem: problem}
}

func joinValidation(errs ...error) error {
	filtered := errs[:0]
	for _, err := range errs {
		if err != nil {
			filtered = append(filtered, err)
		}
	}
	return errors.Join(filtered...)
}

func validateRequired(field, value string) error {
	if strings.TrimSpace(value) == "" {
		return invalid(field, "must not be empty")
	}
	if strings.TrimSpace(value) != value {
		return invalid(field, "must not have surrounding whitespace")
	}
	return nil
}

func validateRevision(revision Revision) error {
	if revision == 0 {
		return invalid("revision", "must be positive")
	}
	return nil
}

func validateTimestamps(createdAt, updatedAt time.Time) error {
	if createdAt.IsZero() {
		return invalid("created_at", "must be set")
	}
	if updatedAt.IsZero() {
		return invalid("updated_at", "must be set")
	}
	if updatedAt.Before(createdAt) {
		return invalid("updated_at", "must not precede created_at")
	}
	return nil
}
