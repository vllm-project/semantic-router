package upstream

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"syscall"
)

// Envoy 1.35's router local-reply bodies.
const (
	replyTimeout       = "upstream request timeout"
	replyNoHealthyHost = "no healthy upstream"
	replyResetPrefix   = "upstream connect error or disconnect/reset before headers. "
	replyRetried       = "retried and the latest "
)

// Envoy's reset reasons, by name.
const (
	resetConnectionFailure = "remote connection failure"
	resetConnectionTimeout = "connection timeout"
	resetTermination       = "connection termination"
	resetRefusedStream     = "remote refused stream reset"
	resetOverflow          = "overflow"
)

// answered reports whether Envoy answers this failure with a local reply.
// Other failures end the call: the caller went away, the Set closed, or the
// request could not be routed or sent at all.
func (e *Error) answered() bool {
	switch e.Kind {
	case KindConnectFailure, KindRefusedStream, KindReset, KindTimeout, KindOverflow, KindNoHealthyUpstream:
		return true
	default:
		return false
	}
}

// localReply is the response Envoy's router sends when a call ends without an
// upstream response: the same status, text/plain body and headers, so the
// router core's response phases translate it exactly as they do in Envoy
// mode. Response.Local carries the failure.
func (c *call) localReply(failure *Error) *Response {
	status, body := envoyReply(failure, len(c.attempts) > 1)
	header := http.Header{
		"Content-Type":   {"text/plain"},
		"Content-Length": {strconv.Itoa(len(body))},
	}
	if failure.Kind == KindOverflow {
		header.Set("X-Envoy-Overloaded", "true")
	}
	return &Response{
		StatusCode:   status,
		Header:       header,
		Body:         io.NopCloser(strings.NewReader(body)),
		Trailer:      http.Header{},
		Cluster:      c.cluster.spec.Name,
		DefaultRoute: c.defaultRoute,
		Attempts:     c.attempts,
		Local:        failure,
	}
}

func envoyReply(failure *Error, retried bool) (int, string) {
	switch failure.Kind {
	case KindTimeout:
		return http.StatusGatewayTimeout, replyTimeout
	case KindNoHealthyUpstream:
		return http.StatusServiceUnavailable, replyNoHealthyHost
	case KindOverflow:
		return http.StatusServiceUnavailable, resetReply(retried, resetOverflow, "")
	case KindRefusedStream:
		return http.StatusServiceUnavailable, resetReply(retried, resetRefusedStream, "")
	case KindConnectFailure:
		reason, transport, resolved := connectFailure(failure.Err)
		if !resolved {
			// A name that never resolved leaves Envoy's DNS cluster without hosts.
			return http.StatusServiceUnavailable, replyNoHealthyHost
		}
		return http.StatusServiceUnavailable, resetReply(retried, reason, transport)
	default:
		return http.StatusServiceUnavailable, resetReply(retried, resetTermination, "")
	}
}

func resetReply(retried bool, reason, transport string) string {
	var b strings.Builder
	b.WriteString(replyResetPrefix)
	if retried {
		b.WriteString(replyRetried)
	}
	b.WriteString("reset reason: ")
	b.WriteString(reason)
	if transport != "" {
		b.WriteString(", transport failure reason: ")
		b.WriteString(transport)
	}
	return b.String()
}

// connectFailure names a connection failure in Envoy's terms: the reset
// reason, the transport failure detail, and whether the host name resolved.
func connectFailure(err error) (string, string, bool) {
	var dns *net.DNSError
	if errors.As(err, &dns) {
		return "", "", false
	}
	var netErr net.Error
	if errors.Is(err, context.DeadlineExceeded) || (errors.As(err, &netErr) && netErr.Timeout()) {
		return resetConnectionTimeout, "", true
	}
	var connect *connectError
	if errors.As(err, &connect) && connect.handshake {
		return resetConnectionFailure, "TLS_error:|" + connect.err.Error() + ":TLS_error_end", true
	}
	var errno syscall.Errno
	if errors.As(err, &errno) {
		return resetConnectionFailure, fmt.Sprintf("delayed connect error: %d", int(errno)), true
	}
	return resetConnectionFailure, "", true
}
