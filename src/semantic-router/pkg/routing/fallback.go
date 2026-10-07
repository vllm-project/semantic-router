package routing

import (
	"context"
	"time"
)

// Fallback continues a planned call after an upstream attempt failed: it
// prepares the next candidate's request the way the primary's was prepared,
// or ends the chain. The routing core implements it; whoever executes calls
// (the upstream layer for the native gateway, a request-graph call node) asks
// it only before any response byte reached the client.
type Fallback interface {
	Next(ctx context.Context, outcome Outcome) (FallbackStep, error)
}

// Outcome describes the attempt that failed.
type Outcome struct {
	// Route is the route key the attempt used.
	Route string
	// Status is the response status, or the status of the local reply the
	// gateway produced for a failure without a response.
	Status int
	// Local reports that no backend answered; Failure then classifies why
	// (for example connect_failure, reset, timeout, overflow).
	Local   bool
	Failure string
	// Header and Body are the response's, bounded by the executor.
	Header Header
	Body   []byte
	// Duration is how long the attempt took.
	Duration time.Duration
}

// FallbackStep is what to do next: send Call, answer with Immediate, or, with
// neither set, return the failed response as it is.
type FallbackStep struct {
	Call      *Call
	Immediate *Response
}

// FallbackSession is a Session that can prepare fallback candidates for its
// planned call. An engine whose caller executes fallback attaches it to
// Call.Fallback; from then on the session no longer falls back in its
// response phases, so a request keeps one fallback authority.
type FallbackSession interface {
	Session
	Fallback() Fallback
}
