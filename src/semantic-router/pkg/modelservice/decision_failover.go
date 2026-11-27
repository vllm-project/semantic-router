package modelservice

import (
	"context"
	"errors"
	"io"
	"net"
)

// retryableDecisionError distinguishes a transport or HTTP 5xx failure from
// other errors reported as ErrFailed, without changing the original error.
type retryableDecisionError struct{ error }

func (e retryableDecisionError) Unwrap() error { return e.error }

// CanFailOverDecision reports whether another deployment may answer this
// call. An exhausted or cancelled caller must still stop the chain.
func CanFailOverDecision(err error) bool {
	var retryable retryableDecisionError
	return !errors.Is(err, context.Canceled) && (errors.Is(err, context.DeadlineExceeded) ||
		errors.Is(err, ErrUnavailable) || errors.Is(err, ErrOverloaded) || errors.As(err, &retryable))
}

func decisionHTTPError(status int, err error) error {
	if status >= 500 && status < 600 {
		return retryableDecisionError{err}
	}
	return err
}

func decisionTransportError(err error) error {
	// The runtime client wraps authorization and other non-network errors
	// in url.Error too, so the outer net.Error interface is not sufficient.
	var network *net.OpError
	var dns *net.DNSError
	if errors.As(err, &network) || errors.As(err, &dns) || errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return retryableDecisionError{err}
	}
	return err
}
