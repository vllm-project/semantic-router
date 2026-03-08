package configlifecycle

import "net/http"

// Error carries an HTTP-shaped failure for handler adapters.
type Error struct {
	StatusCode int
	Code       string
	Message    string
}

func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	return e.Message
}

func newBadRequestError(message string) *Error {
	return &Error{StatusCode: http.StatusBadRequest, Message: message}
}
