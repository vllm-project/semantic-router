package upstream

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// Bounds on what a fallback decision sees of a failed response.
const (
	fallbackBodyLimit   = 64 << 10
	fallbackHeaderLimit = 64
)

// Result is what a planned call produced: the upstream response for the
// router core's response phases, or the client response the call's fallback
// answered with instead.
type Result struct {
	Response  *Response
	Immediate *routing.Response
	// Hops counts the calls sent: one, plus one per fallback candidate.
	Hops int
}

// Execute sends a planned call. While the response is a failure and the call
// carries a fallback, it asks the fallback what to do next and sends the
// candidate it prepares. Every hop goes through Do, so retries, ejection and
// timeouts apply per hop, and the chain moves on only before a response is
// returned: nothing falls back once the caller holds response bytes.
//
// A chain that ends without a candidate's success returns the planned call's
// own failed response, whole, as Envoy passes the primary's response on when
// the router's response-phase fallback runs out.
func (s *Set) Execute(ctx context.Context, call *routing.Call, listener string) (*Result, error) {
	fallback := call.Fallback
	var primary *Response
	started := time.Now()
	for hop := 1; ; hop++ {
		resp, err := s.Do(ctx, RequestFromCall(call, listener))
		switch {
		case err != nil && primary == nil:
			return nil, err
		case err != nil:
			// A candidate that could not be sent ends the chain, unless the
			// request itself is gone.
			if ctx.Err() != nil || KindOf(err) == KindClosed {
				_ = primary.Body.Close()
				return nil, err
			}
			return &Result{Response: primary, Hops: hop}, nil
		case fallback == nil || !failed(resp):
			if primary != nil {
				_ = primary.Body.Close()
			}
			return &Result{Response: resp, Hops: hop}, nil
		}
		outcome, replay := failureOutcome(call.Route, resp, time.Since(started))
		if primary == nil {
			resp.Body, primary = replay, resp
		} else {
			_ = replay.Close()
		}
		step, err := fallback.Next(ctx, outcome)
		if err != nil || (step.Call == nil && step.Immediate == nil) {
			return &Result{Response: primary, Hops: hop}, nil
		}
		if step.Immediate != nil {
			_ = primary.Body.Close()
			return &Result{Immediate: step.Immediate, Hops: hop}, nil
		}
		call, started = step.Call, time.Now()
	}
}

// failed reports whether a response is one the fallback may act on: a local
// reply for a call no backend answered, or any status but 2xx, as the
// response-phase fallback treats the primary's and a candidate's.
func failed(resp *Response) bool {
	return resp.Local != nil || resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices
}

// failureOutcome describes a failed response to the fallback. It reads at most
// fallbackBodyLimit of the body and returns a body that replays those bytes
// ahead of the rest, so the response can still be returned whole.
func failureOutcome(route string, resp *Response, elapsed time.Duration) (routing.Outcome, io.ReadCloser) {
	head, _ := io.ReadAll(io.LimitReader(resp.Body, fallbackBodyLimit))
	outcome := routing.Outcome{
		Route:    route,
		Status:   resp.StatusCode,
		Local:    resp.Local != nil,
		Header:   boundedHeader(resp.Header),
		Body:     head,
		Duration: elapsed,
	}
	if resp.Local != nil {
		outcome.Failure = string(resp.Local.Kind)
	}
	return outcome, replayBody{Reader: io.MultiReader(bytes.NewReader(head), resp.Body), closer: resp.Body}
}

type replayBody struct {
	io.Reader
	closer io.Closer
}

func (b replayBody) Close() error { return b.closer.Close() }

// boundedHeader converts a response header to the router core's shape:
// lowercase names, sorted, at most fallbackHeaderLimit fields.
func boundedHeader(h http.Header) routing.Header {
	names := make([]string, 0, len(h))
	for name := range h {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make(routing.Header, 0, min(len(names), fallbackHeaderLimit))
	for _, name := range names {
		for _, value := range h[name] {
			if len(out) == fallbackHeaderLimit {
				return out
			}
			out = append(out, routing.HeaderField{Name: strings.ToLower(name), Value: value})
		}
	}
	return out
}

// RequestFromCall is the upstream request for a planned call: its method and
// path from the pseudo-headers, every other header as given. The route-level
// rules (host rewrite, route headers, internal header removal) are applied by
// Do.
func RequestFromCall(call *routing.Call, listener string) *Request {
	header := http.Header{}
	for _, field := range call.Request.Header {
		if !strings.HasPrefix(field.Name, ":") {
			header.Add(field.Name, field.Value)
		}
	}
	return &Request{
		Method:      call.Request.Header.Get(":method"),
		Path:        call.Request.Header.Get(":path"),
		Header:      header,
		Body:        call.Request.Body,
		RouteKey:    call.Route,
		Listener:    listener,
		Reliability: call.Reliability,
	}
}
