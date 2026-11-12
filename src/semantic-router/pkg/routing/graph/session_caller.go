package graph

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// Upstream sends a planned call, its fallback chain included. The upstream
// layer implements it through an adapter in the composition, since the
// routing core does not import its executors.
type Upstream interface {
	Send(ctx context.Context, call *routing.Call) (*Delivery, error)
}

// Delivery is what the upstream layer returned for a call.
type Delivery struct {
	// Response is the backend's response, or the gateway's own local reply
	// when Local is set; a local reply skips the response phases, as
	// ext_proc skips Envoy's local replies. Nil when Immediate is set.
	Response *routing.UpstreamResponse
	Local    bool
	// Immediate is the answer a fallback chain gave without a backend.
	Immediate *routing.Response
	// Close releases the response; it may be nil.
	Close func()
}

// DefaultMaxHopBodyBytes bounds a hop's response body, the bound of the
// Looper's buffered model responses.
const DefaultMaxHopBodyBytes int64 = 32 << 20

// SessionCaller sends hops through the routing core and the upstream layer in
// process. Each hop is a routing session marked with its routing.Hop and
// served the way the standalone gateway serves a client request: planned,
// sent, and answered through its response phases. The session's plan pins
// nothing of its own, so the engine must serve the generation the client
// request already pins.
type SessionCaller struct {
	Engine   routing.Engine
	Upstream Upstream
	// MaxBodyBytes bounds a hop's response body; zero means
	// DefaultMaxHopBodyBytes.
	MaxBodyBytes int64
}

// ErrHopBodyLimit reports a hop response larger than the caller's bound.
var ErrHopBodyLimit = errors.New("graph: the hop's response exceeds the body limit")

// Call sends one hop.
func (c *SessionCaller) Call(ctx context.Context, req *HopRequest) (resp *HopResponse, err error) {
	request := &routing.Request{Header: req.Request.Header.Clone(), Body: req.Request.Body}
	if request.Header.Get(headers.RequestID) == "" {
		// Envoy gives a request without one an id, so every hop session has
		// its own, as a loopback hop had.
		request.Header.Add(headers.RequestID, uuid.NewString())
	}
	plan, err := c.Engine.Plan(routing.WithHop(ctx, req.Hop), request)
	if err != nil {
		return nil, err
	}
	defer func() { plan.Finish(err) }()
	if plan.Immediate != nil {
		return c.answer(plan.Immediate)
	}
	call := *plan.Call
	if req.Reliability != nil {
		call.Reliability = append(slices.Clone(call.Reliability), req.Reliability)
	}
	delivery, err := c.Upstream.Send(ctx, &call)
	if err != nil {
		return nil, err
	}
	if delivery.Close != nil {
		defer delivery.Close()
	}
	switch {
	case delivery.Immediate != nil:
		return c.answer(delivery.Immediate)
	case delivery.Response == nil:
		return nil, errors.New("graph: the upstream layer returned no response")
	case delivery.Local:
		body, readErr := c.read(delivery.Response.Body)
		if readErr != nil {
			return nil, readErr
		}
		return &HopResponse{Status: delivery.Response.Status, Header: delivery.Response.Header, Body: body}, nil
	}
	out, err := c.Engine.Respond(ctx, plan, delivery.Response)
	if err != nil {
		return nil, err
	}
	return c.answer(out)
}

func (c *SessionCaller) answer(resp *routing.Response) (*HopResponse, error) {
	body := resp.Body
	if resp.Stream != nil {
		streamed, err := c.drain(resp.Stream)
		if err != nil {
			return nil, err
		}
		body = streamed
	}
	if int64(len(body)) > c.limit() {
		return nil, fmt.Errorf("%w (%d bytes)", ErrHopBodyLimit, c.limit())
	}
	return &HopResponse{Status: resp.Status, Header: resp.Header, Body: body}, nil
}

func (c *SessionCaller) drain(stream routing.BodyStream) ([]byte, error) {
	var body []byte
	for {
		chunk, err := stream.Next()
		body = append(body, chunk...)
		if int64(len(body)) > c.limit() {
			return nil, fmt.Errorf("%w (%d bytes)", ErrHopBodyLimit, c.limit())
		}
		if errors.Is(err, io.EOF) {
			return body, nil
		}
		if err != nil {
			return nil, err
		}
	}
}

func (c *SessionCaller) read(body io.Reader) ([]byte, error) {
	if body == nil {
		return nil, nil
	}
	data, err := io.ReadAll(io.LimitReader(body, c.limit()+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > c.limit() {
		return nil, fmt.Errorf("%w (%d bytes)", ErrHopBodyLimit, c.limit())
	}
	return data, nil
}

func (c *SessionCaller) limit() int64 {
	if c.MaxBodyBytes > 0 {
		return c.MaxBodyBytes
	}
	return DefaultMaxHopBodyBytes
}
