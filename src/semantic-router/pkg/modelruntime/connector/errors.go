package connector

import (
	"errors"
	"fmt"
)

// ErrResponseTooLarge marks a KindResponse failure whose body exceeded
// Options.MaxResponseBytes, so callers can classify it without parsing text.
var ErrResponseTooLarge = errors.New("response body exceeds limit")

// ErrRedirectRejected marks a KindRedirect failure: the remote answered with a
// redirect, which the connector never follows because it would carry the
// request body and credential headers to an origin the caller did not
// configure.
var ErrRedirectRejected = errors.New("redirect rejected")

// ErrorKind identifies the stage at which a connector operation failed.
type ErrorKind string

const (
	KindRequest       ErrorKind = "request"
	KindAuthorization ErrorKind = "authorization"
	KindTransport     ErrorKind = "transport"
	KindStatus        ErrorKind = "status"
	KindResponse      ErrorKind = "response"
	KindRedirect      ErrorKind = "redirect"
)

// Error describes a connector failure without exposing a remote response body
// through Error(). Call ResponseBody when a protocol adapter needs that body
// for a bounded diagnostic.
type Error struct {
	Kind       ErrorKind
	Operation  string
	StatusCode int
	Attempt    int
	Retryable  bool
	Cause      error

	body      []byte
	truncated bool
}

func (e *Error) Error() string {
	prefix := fmt.Sprintf("connector operation %q failed", e.Operation)
	if e.Attempt > 0 {
		prefix = fmt.Sprintf("%s on attempt %d", prefix, e.Attempt)
	}
	if e.StatusCode != 0 {
		return fmt.Sprintf("%s with HTTP status %d", prefix, e.StatusCode)
	}
	if e.Cause != nil {
		return fmt.Sprintf("%s: %v", prefix, e.Cause)
	}
	return prefix
}

func (e *Error) Unwrap() error {
	return e.Cause
}

// ResponseBody returns a copy of the bounded non-success response body and
// reports whether the connector truncated it.
func (e *Error) ResponseBody() ([]byte, bool) {
	return append([]byte(nil), e.body...), e.truncated
}
