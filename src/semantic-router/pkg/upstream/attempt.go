package upstream

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptrace"
	"sync"
	"sync/atomic"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// firstChunkSize bounds what an attempt reads ahead to see the first byte.
const firstChunkSize = 4 << 10

// call is the state of one Do.
type call struct {
	req          *Request
	cluster      *cluster
	defaultRoute bool
	policy       Policy
	rnd          Rand
	clock        Clock
	attempts     []Attempt
	// tried lists the endpoints attempted, which a retry avoids.
	tried []*endpoint
	// sent reports whether the last attempt's request reached the endpoint.
	sent bool
	// served is the body of the last attempt that produced a response.
	served *stream
	// retrySlot frees the circuit-breaker retry the call holds, if any.
	retrySlot func()
}

// try runs one attempt and returns its response once it is ready to commit.
// A failure before that point is classified, recorded on the call and
// returned, so a retry decision can still act on it.
func (c *call) try(ctx context.Context) (*Response, *Error) {
	start := time.Now()
	c.cluster.used.Store(true)
	release, overflow := c.cluster.breaker.acquire(ctx)
	if overflow != nil {
		c.record(nil, start, 0, overflow)
		return nil, overflow
	}
	ep := c.cluster.pickAvoiding(c.tried)
	if ep == nil {
		release()
		failure := &Error{Kind: KindNoHealthyUpstream, Cluster: c.cluster.spec.Name}
		c.record(nil, start, 0, failure)
		return nil, failure
	}
	c.tried = append(c.tried, ep)
	r := c.begin(ctx, ep, start, release)
	resp, head, failure := r.exchange(c) //nolint:bodyclose // the body becomes the stream the caller closes
	c.sent = r.conn.wrote.Load()
	c.record(ep, start, statusOf(resp), failure)
	if failure != nil {
		r.finish(string(failure.Kind), failure)
		return nil, failure
	}
	r.span.SetAttributes(attribute.Int(tracing.AttrHTTPStatusCode, resp.StatusCode))
	c.cluster.observe(ep, resp.StatusCode)
	return c.respond(r, resp, head), nil
}

func (c *call) record(ep *endpoint, start time.Time, status int, failure *Error) {
	a := Attempt{Cluster: c.cluster.spec.Name, Start: start, Latency: time.Since(start), StatusCode: status, Err: failure}
	outcome := statusClass(status)
	if failure != nil {
		outcome = string(failure.Kind)
	}
	if ep != nil {
		a.Endpoint, a.Address = ep.spec.Name, ep.spec.Address()
	}
	c.attempts = append(c.attempts, a)
	metrics.RecordUpstreamAttempt(a.Cluster, a.Endpoint, outcome, a.Latency.Seconds())
}

func (c *call) respond(r *run, resp *http.Response, head []byte) *Response {
	removeHopByHop(resp.Header)
	r.served = true
	body := newStream(r, resp.Body, head, c.policy.Timeouts.Idle)
	c.served = body
	return &Response{
		StatusCode:   resp.StatusCode,
		Header:       resp.Header,
		Body:         body,
		Trailer:      resp.Trailer,
		Cluster:      c.cluster.spec.Name,
		DefaultRoute: c.defaultRoute,
		Endpoint:     r.ep.spec,
		Attempts:     c.attempts,
	}
}

// run is one attempt in flight against one endpoint.
type run struct {
	ctx     context.Context
	cancel  context.CancelCauseFunc
	span    trace.Span
	timers  *attemptTimers
	conn    transportTrace
	ep      *endpoint
	owner   *cluster
	cluster string
	// release returns the attempt's circuit-breaker slot.
	release func()
	// served is set once the attempt's response is handed to the caller.
	served bool
	once   sync.Once
}

func (c *call) begin(ctx context.Context, ep *endpoint, start time.Time, release func()) *run {
	name := c.cluster.spec.Name
	ep.active.Add(1)
	metrics.AddUpstreamActiveRequests(name, ep.spec.Name, 1)
	attemptCtx, cancel := context.WithCancelCause(ctx)
	spanCtx, span := tracing.StartSpan(attemptCtx, tracing.SpanUpstreamAttempt,
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithTimestamp(start),
		trace.WithAttributes(
			attribute.String(tracing.AttrUpstreamCluster, name),
			attribute.String(tracing.AttrUpstreamEndpoint, ep.spec.Name),
			attribute.String(tracing.AttrEndpointAddress, ep.spec.Address()),
			attribute.Int(tracing.AttrUpstreamAttempt, len(c.attempts)+1),
		))
	r := &run{ctx: spanCtx, cancel: cancel, span: span, ep: ep, owner: c.cluster, cluster: name, release: release}
	r.timers = &attemptTimers{cancel: cancel}
	r.timers.arm(c.policy.Timeouts.PerTry, StagePerTry)
	r.conn.onWrote = func() { r.timers.arm(c.policy.Timeouts.FirstByte, StageFirstByte) }
	return r
}

// exchange sends the request and waits until the response is ready to
// commit: its headers, and its first body byte when the first-byte timeout
// is set.
func (r *run) exchange(c *call) (*http.Response, []byte, *Error) {
	out, err := outbound(httptrace.WithClientTrace(r.ctx, r.conn.clientTrace()),
		c.req, &c.cluster.spec, &r.ep.spec, c.defaultRoute)
	if err != nil {
		return nil, nil, r.located(asError(err))
	}
	for _, kv := range tracing.InjectSpanContextToSlice(r.ctx) {
		out.Header.Set(kv[0], kv[1])
	}
	resp, err := c.cluster.pool.transport.RoundTrip(out)
	var head []byte
	if err == nil && enabled(c.policy.Timeouts.FirstByte) {
		head, err = firstChunk(resp.Body)
	}
	r.timers.stop()
	if err == nil && r.ctx.Err() != nil {
		err = context.Cause(r.ctx)
	}
	if err != nil {
		if resp != nil {
			_ = resp.Body.Close()
		}
		return nil, nil, r.located(r.classify(err))
	}
	return resp, head, nil
}

// classify names why the exchange failed: a timeout or cancellation of the
// attempt, a connection that could not be opened, or one that broke after
// the request went out.
func (r *run) classify(err error) *Error {
	if r.ctx.Err() != nil {
		return contextError(r.ctx)
	}
	var connect *connectError
	if errors.As(err, &connect) || !r.conn.connected.Load() {
		return &Error{Kind: KindConnectFailure, Err: err}
	}
	return &Error{Kind: KindReset, Err: err}
}

func (r *run) located(e *Error) *Error {
	e.Cluster, e.Endpoint = r.cluster, r.ep.spec.Name
	return e
}

// finish closes the attempt once: it releases the endpoint and the attempt's
// context, and ends its span. A served attempt finishes when its body ends.
func (r *run) finish(outcome string, failure *Error) {
	r.once.Do(func() {
		if code, charged := outlierCode(failure); charged {
			r.owner.observe(r.ep, code)
		}
		r.release()
		r.ep.active.Add(-1)
		metrics.AddUpstreamActiveRequests(r.cluster, r.ep.spec.Name, -1)
		if r.served {
			metrics.RecordUpstreamStream(r.cluster, r.ep.spec.Name, outcome)
		}
		r.span.SetAttributes(attribute.String(tracing.AttrUpstreamOutcome, outcome))
		if failure != nil {
			r.span.SetStatus(codes.Error, failure.Error())
		}
		r.span.End()
		r.cancel(nil)
	})
}

// outlierCode is the status Envoy's router records against an endpoint for a
// failure it caused: 503 for a connection that failed or broke, 504 for a
// timeout of the request. A stream idle timeout, a cancellation or a local
// rejection says nothing about the endpoint.
func outlierCode(failure *Error) (int, bool) {
	if failure == nil {
		return 0, false
	}
	switch failure.Kind {
	case KindConnectFailure, KindRefusedStream, KindReset:
		return http.StatusServiceUnavailable, true
	case KindTimeout:
		return http.StatusGatewayTimeout, failure.Stage != StageIdle
	default:
		return 0, false
	}
}

// firstChunk reads the first piece of a body so that a stream that stalls
// before its first byte fails while the attempt can still be retried.
func firstChunk(body io.Reader) ([]byte, error) {
	buf := make([]byte, firstChunkSize)
	n, err := body.Read(buf)
	if errors.Is(err, io.EOF) {
		err = nil
	}
	return buf[:n], err
}

func statusOf(resp *http.Response) int {
	if resp == nil {
		return 0
	}
	return resp.StatusCode
}

func asError(err error) *Error {
	var upstreamErr *Error
	if errors.As(err, &upstreamErr) {
		return upstreamErr
	}
	return &Error{Kind: KindInvalidRequest, Err: err}
}

// transportTrace notes how far one attempt got through the transport.
type transportTrace struct {
	connected atomic.Bool
	wrote     atomic.Bool
	onWrote   func()
}

func (t *transportTrace) clientTrace() *httptrace.ClientTrace {
	return &httptrace.ClientTrace{
		GotConn: func(httptrace.GotConnInfo) { t.connected.Store(true) },
		WroteRequest: func(info httptrace.WroteRequestInfo) {
			if info.Err == nil {
				t.wrote.Store(true)
				t.onWrote()
			}
		},
	}
}

// attemptTimers are the per-attempt timeouts: each cancels the attempt with
// its stage as the cause, and none fires once the response is ready.
type attemptTimers struct {
	mu      sync.Mutex
	cancel  context.CancelCauseFunc
	timers  []*time.Timer
	stopped bool
}

func (t *attemptTimers) arm(d time.Duration, stage TimeoutStage) {
	if !enabled(d) {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.stopped {
		return
	}
	t.timers = append(t.timers, time.AfterFunc(d, func() { t.cancel(&timeoutCause{stage: stage}) }))
}

func (t *attemptTimers) stop() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.stopped = true
	for _, timer := range t.timers {
		timer.Stop()
	}
}
