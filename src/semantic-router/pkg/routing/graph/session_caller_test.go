package graph

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// fakeEngine plans a call for every request and records the hop it served.
type fakeEngine struct {
	hops      []routing.Hop
	requests  []*routing.Request
	immediate *routing.Response
	respond   func(*routing.UpstreamResponse) (*routing.Response, error)
}

func (e *fakeEngine) Plan(ctx context.Context, req *routing.Request) (*routing.Plan, error) {
	hop, _ := routing.HopFrom(ctx)
	e.hops, e.requests = append(e.hops, hop), append(e.requests, req)
	plan := &routing.Plan{Immediate: e.immediate}
	if plan.Immediate == nil {
		seconds := time.Second
		plan.Call = &routing.Call{Route: "backend", Request: *req, Reliability: []*routing.Reliability{{TotalTimeout: &seconds}}}
	}
	return plan, nil
}

func (e *fakeEngine) Respond(_ context.Context, _ *routing.Plan, resp *routing.UpstreamResponse) (*routing.Response, error) {
	if e.respond != nil {
		return e.respond(resp)
	}
	body, _ := io.ReadAll(resp.Body)
	return &routing.Response{Status: resp.Status, Header: resp.Header, Body: append([]byte("translated:"), body...)}, nil
}

type fakeUpstream struct {
	calls    []*routing.Call
	delivery *Delivery
	closed   int
}

func (u *fakeUpstream) Send(_ context.Context, call *routing.Call) (*Delivery, error) {
	u.calls = append(u.calls, call)
	delivery := *u.delivery
	delivery.Close = func() { u.closed++ }
	return &delivery, nil
}

func backendDelivery(status int, body string) *Delivery {
	return &Delivery{Response: &routing.UpstreamResponse{Status: status, Body: strings.NewReader(body)}}
}

func hopRequest() *HopRequest {
	retries := 2
	return &HopRequest{
		Hop:         routing.Hop{Decision: "fusion", Recipe: "default", Iteration: 3},
		Model:       "panel-a",
		Request:     routing.Request{Header: ChatHeader(), Body: []byte(`{"model":"panel-a"}`)},
		Reliability: &routing.Reliability{RetryCount: &retries},
	}
}

func TestSessionCallerServesAHopInProcess(t *testing.T) {
	engine := &fakeEngine{}
	upstream := &fakeUpstream{delivery: backendDelivery(200, `{"ok":true}`)}
	caller := &SessionCaller{Engine: engine, Upstream: upstream}
	resp, err := caller.Call(context.Background(), hopRequest())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Status != 200 || string(resp.Body) != `translated:{"ok":true}` {
		t.Fatalf("resp %d %s", resp.Status, resp.Body)
	}
	if engine.hops[0] != (routing.Hop{Decision: "fusion", Recipe: "default", Iteration: 3}) {
		t.Fatalf("the session was not marked as the hop: %+v", engine.hops)
	}
	if engine.requests[0].Header.Get("x-request-id") == "" {
		t.Fatal("a hop session needs a request id of its own")
	}
	layers := upstream.calls[0].Reliability
	if len(layers) != 2 || *layers[0].TotalTimeout != time.Second || *layers[1].RetryCount != 2 {
		t.Fatalf("the step's reliability must layer over the decision's: %+v", layers)
	}
	if upstream.closed != 1 {
		t.Fatalf("closed %d", upstream.closed)
	}
}

func TestSessionCallerAnswersWithoutTheBackend(t *testing.T) {
	engine := &fakeEngine{immediate: &routing.Response{Status: 200, Body: []byte("cached")}}
	upstream := &fakeUpstream{}
	resp, err := (&SessionCaller{Engine: engine, Upstream: upstream}).Call(context.Background(), hopRequest())
	if err != nil || string(resp.Body) != "cached" || len(upstream.calls) != 0 {
		t.Fatalf("err %v resp %+v calls %d", err, resp, len(upstream.calls))
	}

	local := backendDelivery(503, "upstream connect error")
	local.Local = true
	engine = &fakeEngine{respond: func(*routing.UpstreamResponse) (*routing.Response, error) {
		t.Fatal("a local reply skips the response phases")
		return nil, nil
	}}
	resp, err = (&SessionCaller{Engine: engine, Upstream: &fakeUpstream{delivery: local}}).Call(context.Background(), hopRequest())
	if err != nil || resp.Status != 503 || string(resp.Body) != "upstream connect error" {
		t.Fatalf("err %v resp %+v", err, resp)
	}
}

type chunks struct{ parts []string }

func (c *chunks) Next() ([]byte, error) {
	if len(c.parts) == 0 {
		return nil, io.EOF
	}
	part := c.parts[0]
	c.parts = c.parts[1:]
	return []byte(part), nil
}

func TestSessionCallerDrainsAStreamWithinItsLimit(t *testing.T) {
	engine := &fakeEngine{respond: func(*routing.UpstreamResponse) (*routing.Response, error) {
		return &routing.Response{Status: 200, Stream: &chunks{parts: []string{"data: a\n\n", "data: [DONE]\n\n"}}}, nil
	}}
	upstream := &fakeUpstream{delivery: backendDelivery(200, "")}
	resp, err := (&SessionCaller{Engine: engine, Upstream: upstream}).Call(context.Background(), hopRequest())
	if err != nil || string(resp.Body) != "data: a\n\ndata: [DONE]\n\n" {
		t.Fatalf("err %v body %q", err, resp.Body)
	}
	_, err = (&SessionCaller{Engine: engine, Upstream: upstream, MaxBodyBytes: 4}).Call(context.Background(), hopRequest())
	if !errors.Is(err, ErrHopBodyLimit) {
		t.Fatalf("err %v", err)
	}
}
