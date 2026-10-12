package upstream

import (
	"context"
	"net"
	"net/http"
	"os"
	"syscall"
	"testing"
)

func TestEnvoyReplyStrings(t *testing.T) {
	const reset = "upstream connect error or disconnect/reset before headers. "
	refused := &connectError{err: &net.OpError{Op: "dial", Net: "tcp", Err: os.NewSyscallError("connect", syscall.ECONNREFUSED)}}
	tests := []struct {
		name    string
		failure *Error
		retried bool
		status  int
		body    string
	}{
		{"timeout", &Error{Kind: KindTimeout, Stage: StagePerTry}, true, http.StatusGatewayTimeout, "upstream request timeout"},
		{"no healthy upstream", &Error{Kind: KindNoHealthyUpstream}, false, http.StatusServiceUnavailable, "no healthy upstream"},
		{"overflow", &Error{Kind: KindOverflow}, false, http.StatusServiceUnavailable, reset + "reset reason: overflow"},
		{
			"retried reset", &Error{Kind: KindReset}, true, http.StatusServiceUnavailable,
			reset + "retried and the latest reset reason: connection termination",
		},
		{
			"refused", &Error{Kind: KindConnectFailure, Err: refused}, false, http.StatusServiceUnavailable,
			reset + "reset reason: remote connection failure, transport failure reason: delayed connect error: 111",
		},
		{
			"connect timeout", &Error{Kind: KindConnectFailure, Err: &connectError{err: context.DeadlineExceeded}}, false,
			http.StatusServiceUnavailable, reset + "reset reason: connection timeout",
		},
		{
			"unresolved name", &Error{Kind: KindConnectFailure, Err: &connectError{err: &net.DNSError{Err: "no such host", IsNotFound: true}}},
			false, http.StatusServiceUnavailable, "no healthy upstream",
		},
	}
	for _, tt := range tests {
		status, body := envoyReply(tt.failure, tt.retried)
		if status != tt.status || body != tt.body {
			t.Errorf("%s: reply = %d %q, want %d %q", tt.name, status, body, tt.status, tt.body)
		}
	}
}

func TestOnlyFailuresWithoutAResponseAreAnswered(t *testing.T) {
	for kind, want := range map[Kind]bool{
		KindConnectFailure: true, KindRefusedStream: true, KindReset: true, KindTimeout: true,
		KindOverflow: true, KindNoHealthyUpstream: true,
		KindCanceled: false, KindClosed: false, KindNoRoute: false, KindInvalidRequest: false,
	} {
		if got := (&Error{Kind: kind}).answered(); got != want {
			t.Errorf("%s answered = %v, want %v", kind, got, want)
		}
	}
}
