package connector

import (
	"compress/gzip"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zaptest/observer"
	"golang.org/x/net/http2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	tlsutil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/tls"
)

func testBreaker(t *testing.T, threshold int) (*circuitBreaker, func()) {
	t.Helper()
	c, advance := newBreakerTestClient(t, threshold, testOptions())
	return c.breaker, advance
}

func newBreakerTestClient(t *testing.T, threshold int, options Options) (*Client, func()) {
	t.Helper()
	options.CircuitBreaker = &CircuitBreakerOptions{Name: t.Name(), ConsecutiveFailures: threshold, OpenInterval: time.Minute}
	c, err := New("https://classifier.invalid", nil, options)
	require.NoError(t, err)
	t.Cleanup(func() { _ = c.Close() })
	now := time.Unix(100, 0)
	c.breaker.now = func() time.Time { return now }
	return c, func() { now = now.Add(time.Minute) }
}

func TestCircuitBreakerTransitionsAndObservability(t *testing.T) {
	core, logs := observer.New(zap.InfoLevel)
	restore := zap.ReplaceGlobals(zap.New(core))
	defer restore()
	b, advance := testBreaker(t, 2)
	failure := &Error{Kind: KindTransport}
	ticket, _ := b.admit()
	b.complete(ticket, context.Background(), failure)
	require.Equal(t, breakerClosed, b.state)
	require.Equal(t, 1, b.failures)
	skips := metrics.RemoteConnectorBreakerSkipped.WithLabelValues(t.Name())
	skipsBefore := testutil.ToFloat64(skips)
	before := map[breakerState]float64{}
	for _, state := range []breakerState{breakerClosed, breakerOpen, breakerHalfOpen} {
		before[state] = testutil.ToFloat64(metrics.RemoteConnectorBreakerTransitions.WithLabelValues(t.Name(), state.String()))
	}
	for _, outcome := range []error{failure, failure, nil} {
		if b.state == breakerOpen {
			advance()
		}
		ticket, ok := b.admit()
		require.True(t, ok)
		if ticket.probe {
			_, ok = b.admit()
			require.False(t, ok, "second probe admitted")
		}
		b.complete(ticket, context.Background(), outcome)
		want := breakerClosed
		if outcome != nil {
			want = breakerOpen
			_, ok = b.admit()
			require.False(t, ok, "open admitted RPC")
		}
		require.Equal(t, want, b.state)
	}
	require.Equal(t, skipsBefore+4, testutil.ToFloat64(skips))
	for state, increment := range map[breakerState]float64{breakerClosed: 1, breakerOpen: 2, breakerHalfOpen: 2} {
		require.Equal(t, before[state]+increment, testutil.ToFloat64(metrics.RemoteConnectorBreakerTransitions.WithLabelValues(t.Name(), state.String())))
	}
	entries := logs.All()
	require.Len(t, entries, 5)
	for i, state := range []string{"open", "half_open", "open", "half_open", "closed"} {
		fields := entries[i].ContextMap()
		require.Equal(t, "circuit_breaker_transition", fields["event"])
		require.Equal(t, state, fields["to"])
		require.Equal(t, t.Name(), fields["backend"])
	}
}

func TestCircuitBreakerNeutralProbeAndStaleCompletions(t *testing.T) {
	b, advance := testBreaker(t, 1)
	old, _ := b.admit()
	trip, _ := b.admit()
	b.complete(trip, context.Background(), &Error{Kind: KindTransport})
	advance()
	probe, ok := b.admit()
	for _, stale := range []error{nil, &Error{Kind: KindTransport}} {
		b.complete(old, context.Background(), stale)
		require.Equal(t, breakerHalfOpen, b.state)
		require.True(t, b.probing, "old result stole probe")
	}
	for _, kind := range []ErrorKind{KindRequest, KindBudget, KindAuthorization, KindRedirect, KindResponse, KindStatus, KindTransport} {
		require.True(t, ok && probe.probe, "replacement probe denied")
		ctx, cancel := context.WithCancel(context.Background())
		var neutral error = &Error{Kind: kind, StatusCode: 400}
		if kind == KindTransport {
			cancel()
			neutral = nil // A canceled caller cannot establish recovery either.
		}
		b.complete(probe, ctx, neutral)
		cancel()
		require.Equal(t, breakerHalfOpen, b.state)
		require.False(t, b.probing, "neutral probe wedged breaker")
		probe, ok = b.admit()
	}
	require.True(t, ok && probe.probe)
	b.complete(probe, context.Background(), nil)
	require.Equal(t, breakerClosed, b.state)
}

type breakerTransport struct {
	call   func(*http.Request) (*http.Response, error)
	closed bool
}

func (r *breakerTransport) RoundTrip(req *http.Request) (*http.Response, error) { return r.call(req) }
func (r *breakerTransport) CloseIdleConnections()                               { r.closed = true }

func TestCircuitBreakerDisabled(t *testing.T) {
	for _, options := range []*CircuitBreakerOptions{nil, {}} {
		b, err := newCircuitBreaker(options)
		require.NoError(t, err)
		require.Nil(t, b)
	}
}

type breakerBody struct {
	ctx context.Context
	err error
}

func (b breakerBody) Read([]byte) (int, error) {
	if b.err != nil {
		return 0, b.err
	}
	<-b.ctx.Done()
	return 0, b.ctx.Err()
}

func (breakerBody) Close() error { return nil }

func TestCircuitBreakerConnectorOutcomes(t *testing.T) {
	reset := http2.StreamError{StreamID: 1, Code: http2.ErrCodeInternal}
	for _, tc := range []struct {
		name     string
		status   int
		bodyErr  error
		safe     bool
		source   string
		counted  bool
		wantErr  error
		attempts int
	}{
		{"attempt-timeout", 0, nil, false, "", true, context.DeadlineExceeded, 1},
		{"retry-status", 503, nil, true, "", true, nil, 2},
		{"caller-cancel", 0, nil, true, "cancel", false, context.Canceled, 1},
		{"caller-deadline", 0, nil, true, "caller", false, context.DeadlineExceeded, 1},
		{"nested-owned", 0, nil, true, "owned", true, context.DeadlineExceeded, 1},
		{"nested-canceled", 0, nil, true, "owned-cancel", false, context.Canceled, 1},
		{"body-timeout/200", 200, nil, true, "", true, context.DeadlineExceeded, 2},
		{"body-timeout/503", 503, nil, true, "", true, context.DeadlineExceeded, 2},
		{"body-eof/200", 200, io.ErrUnexpectedEOF, true, "", true, io.ErrUnexpectedEOF, 2},
		{"body-eof/503", 503, io.ErrUnexpectedEOF, true, "", true, io.ErrUnexpectedEOF, 2},
		{"body-reset/200", 200, reset, true, "", true, reset, 2},
		{"body-reset/503", 503, fmt.Errorf("wrapped reset: %w", reset), true, "", true, reset, 2},
		{"body-caller/200", 200, nil, true, "caller", false, context.DeadlineExceeded, 1},
		{"body-caller/503", 503, nil, true, "caller", false, context.DeadlineExceeded, 1},
		{"body-size", 200, ErrResponseTooLarge, true, "", false, ErrResponseTooLarge, 1},
		{"body-gzip", 200, gzip.ErrHeader, true, "", false, gzip.ErrHeader, 2},
		{"owned-backoff", 200, gzip.ErrHeader, true, "backoff", false, context.DeadlineExceeded, 1},
		{"unavailable-backoff", 503, nil, true, "backoff", true, context.DeadlineExceeded, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			options := testOptions()
			options.AttemptTimeout, options.MaxRetries = 5*time.Millisecond, 1
			if tc.source != "" {
				options.AttemptTimeout = time.Second
			}
			c, advance := newBreakerTestClient(t, 1, options)
			calls := 0
			var cancelCaller context.CancelFunc
			transport := &breakerTransport{call: func(req *http.Request) (*http.Response, error) {
				calls++
				if strings.Contains(tc.source, "cancel") {
					cancelCaller()
				}
				if tc.status == 0 {
					<-req.Context().Done()
					return nil, req.Context().Err()
				}
				var body io.ReadCloser = http.NoBody
				if tc.status == 200 || strings.HasPrefix(tc.name, "body-") || tc.bodyErr != nil {
					body = breakerBody{ctx: req.Context(), err: tc.bodyErr}
				}
				if errors.Is(tc.bodyErr, ErrResponseTooLarge) {
					body = io.NopCloser(strings.NewReader(strings.Repeat("x", 1025)))
				}
				return &http.Response{StatusCode: tc.status, Body: body}, nil
			}}
			c.http.Transport = transport
			operation := testOperation
			operation.Name, operation.RetrySafe = t.Name(), tc.safe
			if tc.name == "retry-status" {
				c.breaker.options.ConsecutiveFailures = 2
				_, err := c.Do(context.Background(), operation, nil)
				require.Error(t, err)
				require.Equal(t, 1, c.breaker.failures, "retries must count once per call")
				require.Equal(t, breakerClosed, c.breaker.state)
			}
			counter := metrics.RemoteConnectorRequestsTotal.WithLabelValues(operation.Name, "circuit_open")
			histogram := metrics.RemoteConnectorRequestDuration.WithLabelValues(operation.Name).(interface{ Write(*dto.Metric) error })
			before := testutil.ToFloat64(counter)
			var durationBefore, durationAfter dto.Metric
			require.NoError(t, histogram.Write(&durationBefore))
			phases := []bool{false, true}
			if tc.name == "body-size" || tc.name == "body-gzip" {
				phases = phases[1:]
			}
			for _, probe := range phases {
				t.Run(fmt.Sprintf("probe=%t", probe), func(t *testing.T) {
					want := breakerClosed
					if probe {
						if c.breaker.state == breakerClosed {
							ticket, _ := c.breaker.admit()
							c.breaker.complete(ticket, context.Background(), &Error{Kind: KindTransport})
						}
						advance()
						want = breakerHalfOpen
					}
					ctx, cancel := context.WithCancel(context.Background())
					cancelCaller = cancel
					defer cancel()
					if strings.HasPrefix(tc.source, "owned") || tc.source == "backoff" {
						ctx = WithCallerContext(ctx)
					}
					if tc.source != "" {
						ctx, cancel = context.WithTimeout(ctx, 20*time.Millisecond)
						defer cancel()
						ctx = WithCallerContext(ctx) // A nested adapter must retain the original caller.
					}
					callsBefore := calls
					_, err := c.Do(ctx, operation, nil)
					require.Error(t, err)
					if tc.wantErr != nil {
						require.ErrorIs(t, err, tc.wantErr)
					}
					if tc.source == "backoff" {
						var remote *Error
						require.ErrorAs(t, err, &remote)
						require.Equal(t, KindTransport, remote.Kind)
					}
					require.Equal(t, callsBefore+tc.attempts, calls, "retry budget changed")
					if tc.counted {
						want = breakerOpen
						_, err = c.Do(context.Background(), operation, nil)
						require.ErrorIs(t, err, context.DeadlineExceeded)
						require.Equal(t, callsBefore+tc.attempts, calls, "open call issued RPC")
					}
					require.Equal(t, want, c.breaker.state, "availability attribution changed")
					require.Zero(t, c.breaker.failures)
					require.False(t, c.breaker.probing, "probe slot was not released")
				})
			}
			observations := uint64(len(phases))
			if tc.counted {
				observations = 4
				require.Equal(t, before+2, testutil.ToFloat64(counter))
			}
			require.NoError(t, histogram.Write(&durationAfter))
			require.Equal(t, durationBefore.GetHistogram().GetSampleCount()+observations, durationAfter.GetHistogram().GetSampleCount())
			require.NoError(t, c.Close())
			require.True(t, transport.closed, "transport cleanup lost")
		})
	}
}

func TestCircuitBreakerHTTP2InterruptedBody(t *testing.T) {
	cert, err := tlsutil.CreateSelfSignedTLSCertificate()
	require.NoError(t, err)
	serverTLS := &tls.Config{Certificates: []tls.Certificate{cert}, NextProtos: []string{"h2"}}
	// "close" drops the connection without an HTTP/2 frame and is the control for the two frame-level interruptions.
	for _, mode := range []string{"reset", "goaway", "close"} {
		for _, status := range []int{http.StatusOK, http.StatusServiceUnavailable} {
			t.Run(fmt.Sprintf("%s/%d", mode, status), func(t *testing.T) {
				c, advance := newBreakerTestClient(t, 1, testOptions())
				var calls atomic.Int32
				transport := c.http.Transport.(*http.Transport)
				transport.Proxy = nil
				transport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
				t.Cleanup(transport.CloseIdleConnections)
				headers := make(chan struct{}, 1)
				c.http.Transport = &breakerTransport{call: func(req *http.Request) (*http.Response, error) {
					response, roundTripErr := transport.RoundTrip(req)
					select {
					case headers <- struct{}{}:
					default:
					}
					return response, roundTripErr
				}}
				transport.DialContext = func(context.Context, string, string) (net.Conn, error) {
					client, server := net.Pipe()
					go func() {
						defer server.Close()
						secured := tls.Server(server, serverTLS)
						if handshakeErr := secured.Handshake(); handshakeErr != nil {
							return
						}
						(&http2.Server{}).ServeConn(secured, &http2.ServeConnOpts{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
							calls.Add(1)
							w.WriteHeader(status)
							_, _ = w.Write([]byte("partial"))
							w.(http.Flusher).Flush()
							if mode == "reset" {
								panic(http.ErrAbortHandler) // Sends RST_STREAM(INTERNAL_ERROR) after headers.
							}
							<-headers
							if mode == "goaway" {
								_ = http2.NewFramer(secured, nil).WriteGoAway(1, http2.ErrCodeInternal, nil)
							}
							_ = server.Close()
						})})
					}()
					return client, nil
				}
				for _, probe := range []bool{false, true} {
					if probe {
						advance()
					}
					_, err := c.Do(context.Background(), testOperation, nil)
					var remote *Error
					require.ErrorAs(t, err, &remote)
					if mode == "reset" {
						var reset http2.StreamError
						require.ErrorAs(t, err, &reset)
					}
					require.Equal(t, breakerOpen, c.breaker.state, "interrupted HTTP/2 body did not open breaker (probe=%t): %v", probe, err)
					_, err = c.Do(context.Background(), testOperation, nil)
					require.ErrorIs(t, err, context.DeadlineExceeded)
				}
				require.Equal(t, int32(2), calls.Load(), "open calls must skip HTTP/2 transport")
			})
		}
	}
}

func TestCircuitBreakerSharedBackendName(t *testing.T) {
	open := metrics.RemoteConnectorBreakerTransitions.WithLabelValues(t.Name(), "open")
	closed := metrics.RemoteConnectorBreakerTransitions.WithLabelValues(t.Name(), "closed")
	openBefore, closedBefore := testutil.ToFloat64(open), testutil.ToFloat64(closed)
	b, advance := testBreaker(t, 1)
	ticket, _ := b.admit()
	b.complete(ticket, context.Background(), &Error{Kind: KindTransport})
	other, err := newCircuitBreaker(&b.options)
	require.NoError(t, err)
	require.Equal(t, breakerClosed, other.state)
	require.Equal(t, breakerOpen, b.state)
	require.Equal(t, openBefore+1, testutil.ToFloat64(open))
	require.Equal(t, closedBefore, testutil.ToFloat64(closed), "construction is not a transition")
	ticket, _ = other.admit()
	other.complete(ticket, context.Background(), &Error{Kind: KindTransport})
	require.Equal(t, openBefore+2, testutil.ToFloat64(open), "same-name clients aggregate transitions")
	advance()
	ticket, _ = b.admit()
	b.complete(ticket, context.Background(), nil)
	require.Equal(t, breakerOpen, other.state)
	require.Equal(t, closedBefore+1, testutil.ToFloat64(closed))
	require.Equal(t, openBefore+2, testutil.ToFloat64(open))
}
