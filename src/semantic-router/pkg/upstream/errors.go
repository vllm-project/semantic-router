package upstream

import (
	"context"
	"errors"
	"fmt"
	"net/http"
)

// Kind classifies why an upstream call produced no usable response.
type Kind string

const (
	// KindConnectFailure: no connection could be opened (refused,
	// unreachable, name resolution, TLS handshake, or the connect timeout).
	KindConnectFailure Kind = "connect_failure"
	// KindRefusedStream: the server refused the request before processing it.
	// Only HTTP/2 reports it; over HTTP/1.1 the transport itself resends a
	// request the server closed its connection on before reading.
	KindRefusedStream Kind = "refused_stream"
	// KindReset: the connection failed after the request was sent, before a
	// complete response arrived.
	KindReset Kind = "reset"
	// KindTimeout: a timeout fired; Error.Stage names which one.
	KindTimeout Kind = "timeout"
	// KindOverflow: a circuit breaker rejected the request.
	KindOverflow Kind = "overflow"
	// KindNoHealthyUpstream: the cluster had no endpoint to pick.
	KindNoHealthyUpstream Kind = "no_healthy_upstream"
	// KindNoRoute: no cluster serves the route key and there is no default.
	KindNoRoute Kind = "no_route"
	// KindCanceled: the caller canceled the call.
	KindCanceled Kind = "canceled"
	// KindClosed: the Set was closed and accepts no new calls.
	KindClosed Kind = "closed"
	// KindInvalidRequest: the request cannot be sent as given.
	KindInvalidRequest Kind = "invalid_request"
	// KindBudgetExhausted: the shared request-wide model call limit was reached.
	KindBudgetExhausted Kind = "budget_exhausted"
)

// TimeoutStage names the timeout behind a KindTimeout error.
type TimeoutStage string

const (
	StageFirstByte TimeoutStage = "first_byte"
	StagePerTry    TimeoutStage = "per_try"
	StageTotal     TimeoutStage = "total"
	StageIdle      TimeoutStage = "idle"
)

// Error is a classified upstream failure.
type Error struct {
	Kind Kind
	// Stage is set for KindTimeout.
	Stage TimeoutStage
	// Cluster and Endpoint name where the failure happened, when known.
	Cluster  string
	Endpoint string
	Err      error
}

func (e *Error) Error() string {
	msg := "upstream " + string(e.Kind)
	if e.Stage != "" {
		msg += " (" + string(e.Stage) + ")"
	}
	if e.Cluster != "" {
		msg += " cluster=" + e.Cluster
	}
	if e.Endpoint != "" {
		msg += " endpoint=" + e.Endpoint
	}
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

func (e *Error) Unwrap() error { return e.Err }

// StatusCode is the HTTP status of the local reply Envoy sends for the same
// failure: 504 for a timeout before the response started, 503 otherwise.
func (e *Error) StatusCode() int {
	switch e.Kind {
	case KindTimeout:
		return http.StatusGatewayTimeout
	case KindInvalidRequest:
		return http.StatusBadRequest
	default:
		return http.StatusServiceUnavailable
	}
}

// KindOf returns the classification of err, or "" when err is not an
// upstream error.
func KindOf(err error) Kind {
	var upstreamErr *Error
	if errors.As(err, &upstreamErr) && upstreamErr != nil {
		return upstreamErr.Kind
	}
	return ""
}

// timeoutCause marks the context cancellation of a timeout this package
// armed, so the error it produces can be told apart from the caller's own.
type timeoutCause struct{ stage TimeoutStage }

func (c *timeoutCause) Error() string { return "upstream " + string(c.stage) + " timeout" }

var errSetClosed = errors.New("upstream set is closed")

func invalidRequest(format string, args ...any) *Error {
	return &Error{Kind: KindInvalidRequest, Err: fmt.Errorf(format, args...)}
}

// contextError classifies the end of a context: a timeout this package armed,
// the deadline of the caller's context (its total budget), or a cancellation.
func contextError(ctx context.Context) *Error {
	cause := context.Cause(ctx)
	var timeout *timeoutCause
	switch {
	case errors.As(cause, &timeout):
		return &Error{Kind: KindTimeout, Stage: timeout.stage, Err: cause}
	case errors.Is(cause, context.DeadlineExceeded):
		return &Error{Kind: KindTimeout, Stage: StageTotal, Err: cause}
	case errors.Is(cause, errSetClosed):
		return &Error{Kind: KindClosed, Err: cause}
	default:
		return &Error{Kind: KindCanceled, Err: cause}
	}
}
