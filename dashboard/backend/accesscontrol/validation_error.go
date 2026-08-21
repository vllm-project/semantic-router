package accesscontrol

import "fmt"

type ValidationError struct {
	message string
}

func (e *ValidationError) Error() string { return e.message }

func validationError(message string) error {
	return &ValidationError{message: message}
}

func validationErrorf(format string, args ...any) error {
	return validationError(fmt.Sprintf(format, args...))
}
