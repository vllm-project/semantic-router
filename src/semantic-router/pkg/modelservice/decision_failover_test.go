package modelservice

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/url"
	"testing"
	"time"
)

func TestDecisionFailoverDistinguishesRetryableFailures(t *testing.T) {
	for _, test := range []struct {
		name string
		err  error
		want bool
	}{
		{"not ready", ErrUnavailable, true},
		{"native 429", statusError(http.StatusTooManyRequests, nil), true},
		{"native 502", statusError(http.StatusBadGateway, nil), true},
		{"System One 500", systemOneStatusError(http.StatusInternalServerError), true},
		{"transport", transportError(t.Context(), &url.Error{Err: &net.OpError{Op: "dial", Err: errors.New("connection failed")}}), true},
		{"attempt deadline", context.DeadlineExceeded, true},
		{"cancelled", context.Canceled, false},
		{"invalid request", ErrRejected, false},
		{"auth rejected", statusError(http.StatusForbidden, nil), false},
		{"missing credential", transportError(t.Context(), &url.Error{Err: errors.New("no credential")}), false},
		{"bad response", ErrFailed, false},
		{"unknown deployment", ErrUnknownDeployment, false},
		{"success", nil, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := CanFailOverDecision(test.err); got != test.want {
				t.Fatalf("CanFailOverDecision(%v) = %v, want %v", test.err, got, test.want)
			}
		})
	}
}

func TestDecisionFailoverAttemptDoesNotWaitForSharedExchange(t *testing.T) {
	carrier, end := context.WithCancel(t.Context())
	defer end()
	ctx, cancel := context.WithDeadline(WithOwnDecisionDeadline(t.Context()), time.Now().Add(-time.Second))
	defer cancel()
	call := bundleCall{ctx: ctx, carrier: carrier, done: make(chan struct{})}
	err := make(chan error, 1)
	go func() { err <- call.waitCarrier() }()
	select {
	case got := <-err:
		if !errors.Is(got, context.DeadlineExceeded) || carrier.Err() != nil {
			t.Fatalf("attempt returned %v, shared exchange returned %v", got, carrier.Err())
		}
	case <-time.After(time.Second):
		t.Fatal("attempt waited after its deadline for the shared exchange")
	}
}
