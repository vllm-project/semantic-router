package extproc

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

// Tests that build a router without the configuration lifecycle point its
// deprecated Looper endpoint at a fake model server; their hops go there.
func init() {
	testLooperHops = func(r *OpenAIRouter) graph.Caller {
		if r.Config == nil || r.Config.Looper.Endpoint == "" {
			return nil
		}
		return directLooperHops(r.Config.Looper.Endpoint)
	}
}

// directLooperHops sends a router's Looper hops straight to the model server
// at endpoint, for tests whose fake server stands in for every backend: the
// hop's own pipeline is not what they test.
func directLooperHops(endpoint string) graph.Caller {
	return directHopCaller{endpoint: endpoint}
}

type directHopCaller struct{ endpoint string }

func (c directHopCaller) Call(ctx context.Context, req *graph.HopRequest) (*graph.HopResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(req.Request.Body))
	if err != nil {
		return nil, err
	}
	for _, field := range req.Request.Header {
		if !strings.HasPrefix(field.Name, ":") && field.Name != "content-length" {
			httpReq.Header.Add(field.Name, field.Value)
		}
	}
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	return &graph.HopResponse{Status: resp.StatusCode, Header: routingHeaderOf(resp.Header), Body: body}, err
}

// Only the process marks a session as a hop; a request header cannot.
func TestARoutingSessionIsAHopOnlyWhenTheProcessMarksIt(t *testing.T) {
	router := &OpenAIRouter{}
	if plain := router.newRoutingSession(context.Background(), nil); plain.ctx.Hop != nil {
		t.Fatalf("a client session is not a hop: %+v", plain.ctx.Hop)
	}
	hop := routing.Hop{Decision: "fusion-route", Recipe: "default", Iteration: 2}
	marked := router.newRoutingSession(routing.WithHop(context.Background(), hop), nil)
	if marked.ctx.Hop == nil || *marked.ctx.Hop != hop {
		t.Fatalf("hop %+v", marked.ctx.Hop)
	}
}

// A hop served in process reads its decision and recipe from its typed
// context; internal headers in its request are dropped, not believed.
func TestAnInProcessHopIgnoresInternalHeaders(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			headers.VSRLooperRequest:  "true",
			headers.VSRLooperDecision: "someone-elses-decision",
			headers.VSRSelectedRecipe: "someone-elses-recipe",
			headers.VSRInternalAuth:   "forged",
		},
		Hop: &routing.Hop{Decision: "fusion-route", Recipe: "default"},
	}
	markLooperHop(ctx)
	if !ctx.LooperRequest {
		t.Fatal("an in-process hop is a Looper request")
	}
	if looperHopDecision(ctx) != "fusion-route" || looperHopRecipe(ctx) != "default" {
		t.Fatalf("decision %q recipe %q", looperHopDecision(ctx), looperHopRecipe(ctx))
	}
	for _, name := range looperInternalContextHeaders {
		if _, ok := ctx.Headers[name]; ok {
			t.Fatalf("%s survived", name)
		}
	}
}

// A request that carries internal context headers, credential included, but
// that the Router did not mark is an ordinary request; the headers go.
func TestInternalHeadersNeverMakeAHop(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		headers.VSRLooperRequest:  "true",
		headers.VSRInternalAuth:   "anything",
		headers.VSRLooperDecision: "fusion-route",
	}}
	markLooperHop(ctx)
	if ctx.LooperRequest || len(ctx.Headers) != 0 {
		t.Fatalf("looper request %t, headers %v", ctx.LooperRequest, ctx.Headers)
	}
}

func TestLooperHopsNeedTheSnapshotsUpstreamSet(t *testing.T) {
	if hops := (&OpenAIRouter{}).looperHops(); hops != nil {
		t.Fatal("a router whose snapshot has no upstream set loops its hops back")
	}
}
