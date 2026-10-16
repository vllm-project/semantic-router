package routing

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strconv"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// RouteHeader is the request header whose value selects the upstream route:
// the local Envoy template's route table matches it exactly on provider model
// names.
const RouteHeader = headers.SelectedModel

// Engine is the transport-agnostic routing core shared by every gateway mode.
type Engine interface {
	// Plan resolves the entrypoint and recipe, evaluates signals, the decision
	// and request-side plugins, and returns what to do with the request. The
	// caller ends the request with Plan.Finish.
	Plan(ctx context.Context, req *Request) (*Plan, error)
	// Respond runs response-side plugins on the upstream response and returns
	// what the client receives.
	Respond(ctx context.Context, plan *Plan, resp *UpstreamResponse) (*Response, error)
}

// Request is a client request as a gateway's HTTP connection manager presents
// it: pseudo-headers first, then the client's headers.
type Request struct {
	Header Header
	Body   []byte
}

// Plan is what the routing core decided for one request.
type Plan struct {
	// Immediate is the client response when the core answers itself.
	Immediate *Response
	// Call is the upstream call otherwise.
	Call     *Call
	Budget   Budget
	Evidence Evidence

	session Session
	finish  sync.Once
}

// Call is one upstream call.
type Call struct {
	// Route is the route key: the RouteHeader value Envoy's route table would
	// match. Empty means the default route.
	Route string
	// Request is the upstream request after every request-phase effect.
	Request Request
	// Fallback prepares the next candidate when the call fails; nil when the
	// session offers no fallback.
	Fallback Fallback
	// Reliability overrides the provider model's timeouts and retries for
	// this call, lowest first: the decision's, then a request-graph step's.
	// Each merges over the ones before it; none keeps the provider model's.
	Reliability []*Reliability
}

// Budget bounds the work for one request.
type Budget struct {
	// Deadline is when the request must finish; zero means none.
	Deadline time.Time
}

// Response is what the client receives.
type Response struct {
	Status int
	// Header holds regular headers; Status carries ":status".
	Header Header
	// Body is the buffered body. It is nil when Stream is set.
	Body []byte
	// Stream yields the body chunk by chunk for a streamed response.
	Stream BodyStream
}

// BodyStream yields a streamed response body.
type BodyStream interface {
	// Next returns the next chunk for the client, and io.EOF after the last.
	Next() ([]byte, error)
}

// UpstreamResponse is the backend's response as the gateway received it.
type UpstreamResponse struct {
	Status int
	// Header holds regular headers.
	Header Header
	// Body is the response body; nil when the response has none.
	Body io.Reader
}

// Options configure an Engine.
type Options struct {
	// MaxBufferedBodyBytes bounds a buffered upstream response body.
	MaxBufferedBodyBytes int64
	// Limits bound header maps after mutation.
	Limits Limits
	// ExecutesFallback declares that the caller runs Call.Fallback itself (the
	// native gateway does). The engine then attaches the session's fallback to
	// each planned call, and the session stops falling back in its response
	// phases. Without it the session behaves as behind Envoy.
	ExecutesFallback bool
}

// DefaultOptions match the local Envoy template, which buffers up to 500 MiB
// per connection.
var DefaultOptions = Options{MaxBufferedBodyBytes: 500 << 20, Limits: DefaultLimits}

// ErrBufferLimit reports an upstream body larger than MaxBufferedBodyBytes.
var ErrBufferLimit = errors.New("buffered body exceeds the limit")

// ErrResponseCommitted reports an immediate response requested after the
// response headers reached the client, which Envoy cannot honor either.
var ErrResponseCommitted = errors.New("immediate response after the response headers were sent")

// NewEngine returns the Engine that drives p's sessions with the local Envoy
// template's processing mode (headers sent both ways; bodies buffered both
// ways unless a response-headers effect streams the response body) and
// applies their effects with Envoy's rules.
func NewEngine(p Processor, opts Options) Engine {
	if opts.MaxBufferedBodyBytes <= 0 {
		opts.MaxBufferedBodyBytes = DefaultOptions.MaxBufferedBodyBytes
	}
	if opts.Limits == (Limits{}) {
		opts.Limits = DefaultLimits
	}
	return &engine{processor: p, opts: opts}
}

type engine struct {
	processor Processor
	opts      Options
}

// Finish ends the request lifecycle. err is nil when the response was
// delivered and non-nil when the transport ended the request early. Only the
// first call counts.
func (p *Plan) Finish(err error) {
	if p == nil || p.session == nil {
		return
	}
	p.finish.Do(func() { p.session.Close(err) })
}

func (e *engine) Plan(ctx context.Context, req *Request) (*Plan, error) {
	if req == nil {
		return nil, errors.New("routing: nil request")
	}
	session, err := e.processor.Open(ctx)
	if err != nil {
		return nil, err
	}
	plan := &Plan{session: session}
	if deadline, ok := ctx.Deadline(); ok {
		plan.Budget.Deadline = deadline
	}
	upstream := Request{Header: req.Header.Clone(), Body: req.Body}
	reroute := false
	hasBody := len(req.Body) > 0

	effect, err := session.RequestHeaders(req.Header.Clone(), !hasBody)
	if err == nil {
		err = e.applyRequestEffect(plan, &upstream, effect, false, &reroute)
	}
	if err == nil && plan.Immediate == nil && hasBody {
		effect, err = session.RequestBody(req.Body, true)
		if err == nil {
			err = e.applyRequestEffect(plan, &upstream, effect, true, &reroute)
		}
	}
	if err != nil {
		plan.Finish(err)
		return nil, err
	}
	plan.Evidence = session.Evidence()
	if plan.Immediate == nil {
		// Envoy picks the route from the original headers and picks again from
		// the mutated headers only after an effect clears its route cache.
		route := req.Header.Get(RouteHeader)
		if reroute {
			route = upstream.Header.Get(RouteHeader)
		}
		plan.Call = &Call{Route: route, Request: upstream}
		if fallbackSession, ok := session.(FallbackSession); ok && e.opts.ExecutesFallback {
			plan.Call.Fallback = fallbackSession.Fallback()
		}
		if reliabilitySession, ok := session.(ReliabilitySession); ok {
			if override := reliabilitySession.Reliability(); override != nil {
				plan.Call.Reliability = []*Reliability{override}
			}
		}
	}
	return plan, nil
}

func (e *engine) applyRequestEffect(plan *Plan, msg *Request, effect *Effect, bodyPhase bool, reroute *bool) error {
	if effect == nil {
		return nil
	}
	if effect.Immediate != nil {
		plan.Immediate = RenderImmediate(effect.Immediate, e.opts.Limits)
		return nil
	}
	if err := ApplyHeaderMutation(&msg.Header, effect.Header, false, e.opts.Limits); err != nil {
		return err
	}
	if bodyPhase && effect.Body != nil {
		if err := CheckContentLength(msg.Header, effect.Body); err != nil {
			return err
		}
		msg.Body = ApplyBodyMutation(msg.Body, effect.Body)
	}
	if effect.ClearRouteCache && effect.Header != nil {
		*reroute = true
	}
	return nil
}

func (e *engine) Respond(ctx context.Context, plan *Plan, resp *UpstreamResponse) (*Response, error) {
	if plan == nil || plan.session == nil || plan.Call == nil || resp == nil {
		return nil, errors.New("routing: Respond needs a planned call and its upstream response")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	session := plan.session
	defer func() { plan.Evidence = session.Evidence() }()

	header := Header{{Name: ":status", Value: strconv.Itoa(resp.Status)}}
	for _, field := range resp.Header {
		header.Add(field.Name, field.Value)
	}
	hasBody := resp.Body != nil
	effect, err := session.ResponseHeaders(header.Clone(), !hasBody)
	if err != nil {
		return nil, err
	}
	if effect != nil && effect.Immediate != nil {
		return RenderImmediate(effect.Immediate, e.opts.Limits), nil
	}
	streamed := effect != nil && effect.ResponseBodyMode == BodyModeStreamed
	if err = ApplyHeaderMutation(&header, headerMutationOf(effect), streamed, e.opts.Limits); err != nil {
		return nil, err
	}
	if !hasBody {
		return clientResponse(header, nil), nil
	}
	if streamed {
		out := clientResponse(header, nil)
		out.Stream = &bodyStream{session: session, upstream: resp.Body, buf: make([]byte, 32<<10)}
		return out, nil
	}

	body, err := readBounded(resp.Body, e.opts.MaxBufferedBodyBytes)
	if err != nil {
		return nil, err
	}
	effect, err = session.ResponseBody(body, true)
	if err != nil {
		return nil, err
	}
	if effect != nil && effect.Immediate != nil {
		return RenderImmediate(effect.Immediate, e.opts.Limits), nil
	}
	if err = ApplyHeaderMutation(&header, headerMutationOf(effect), false, e.opts.Limits); err != nil {
		return nil, err
	}
	if effect != nil && effect.Body != nil {
		if err = CheckContentLength(header, effect.Body); err != nil {
			return nil, err
		}
		body = ApplyBodyMutation(body, effect.Body)
	}
	return clientResponse(header, body), nil
}

func headerMutationOf(effect *Effect) *HeaderMutation {
	if effect == nil {
		return nil
	}
	return effect.Header
}

func clientResponse(header Header, body []byte) *Response {
	status, _ := strconv.Atoi(header.Get(":status"))
	return &Response{Status: status, Header: header.WithoutPseudo(), Body: body}
}

func readBounded(r io.Reader, limit int64) ([]byte, error) {
	body, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, fmt.Errorf("%w (%d bytes)", ErrBufferLimit, limit)
	}
	return body, nil
}

// bodyStream runs the response-body phase on each upstream chunk once the
// headers have been forwarded. Header mutations in these effects are ignored,
// as Envoy ignores them after the headers are sent.
type bodyStream struct {
	session     Session
	upstream    io.Reader
	buf         []byte
	upstreamEOF bool
	done        bool
}

func (s *bodyStream) Next() ([]byte, error) {
	for !s.done {
		if s.upstreamEOF {
			s.done = true
			out, err := s.process(nil, true)
			if err != nil || len(out) > 0 {
				return out, err
			}
			break
		}
		n, readErr := s.upstream.Read(s.buf)
		if errors.Is(readErr, io.EOF) {
			s.upstreamEOF = true
		} else if readErr != nil {
			return nil, readErr
		}
		if n == 0 {
			continue
		}
		out, err := s.process(append([]byte(nil), s.buf[:n]...), false)
		if err != nil || len(out) > 0 {
			return out, err
		}
	}
	return nil, io.EOF
}

func (s *bodyStream) process(chunk []byte, endOfStream bool) ([]byte, error) {
	effect, err := s.session.ResponseBody(chunk, endOfStream)
	if err != nil {
		return nil, err
	}
	if effect == nil {
		return chunk, nil
	}
	if effect.Immediate != nil {
		return nil, ErrResponseCommitted
	}
	return ApplyBodyMutation(chunk, effect.Body), nil
}
