package looper

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

// hopCaller answers every hop with one response and keeps the last request.
type hopCaller struct {
	resp *graph.HopResponse
	wait bool
	last *graph.HopRequest
}

func (c *hopCaller) Call(ctx context.Context, req *graph.HopRequest) (*graph.HopResponse, error) {
	c.last = req
	if c.wait {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	return c.resp, nil
}

type stepFunc func(ctx context.Context, x *graph.Exec, st *graph.State) error

func (f stepFunc) Run(ctx context.Context, x *graph.Exec, st *graph.State) error {
	return f(ctx, x, st)
}

// callAsHop makes one model call through a hop client inside a run.
func callAsHop(t *testing.T, cfg *config.LooperConfig, caller graph.Caller) (*ModelResponse, error) {
	t.Helper()
	var resp *ModelResponse
	var callErr error
	step := stepFunc(func(ctx context.Context, x *graph.Exec, st *graph.State) error {
		ctx = contextWithRoutingRecipe(ctx, "support")
		resp, callErr = NewHopClient(cfg, x).CallModelWithOptions(ctx,
			openai.ChatCompletionNewParams{Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")}},
			ModelTarget{Name: "panel-a", AccessKey: "key-a"},
			CallOptions{DecisionName: "fusion-route", Iteration: 2},
		)
		st.Response = &graph.Response{}
		return nil
	})
	program := &graph.Program{Name: "test", Steps: graph.Sequence{{ID: "call", Type: "test", Node: step}}}
	if _, err := graph.Run(context.Background(), program, graph.Input{}, graph.Options{Caller: caller}); err != nil {
		t.Fatal(err)
	}
	return resp, callErr
}

func TestHopClientCarriesTheRoutingContextInProcess(t *testing.T) {
	caller := &hopCaller{resp: &graph.HopResponse{Status: 200, Body: []byte(
		`{"object":"chat.completion","model":"panel-a","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`)}}
	cfg := &config.LooperConfig{Headers: map[string]string{"X-Static": "static-a"}}
	resp, err := callAsHop(t, cfg, caller)
	if err != nil || resp.Content != "ok" {
		t.Fatalf("resp %+v err %v", resp, err)
	}
	hop := caller.last.Hop
	if hop.Fallback == nil || hop.Fallback.Enabled == nil || *hop.Fallback.Enabled {
		t.Fatalf("a Looper hop never falls back across models: %+v", hop.Fallback)
	}
	hop.Fallback = nil
	if hop != (routing.Hop{Decision: "fusion-route", Recipe: "support", Iteration: 2}) || caller.last.Model != "panel-a" {
		t.Fatalf("hop %+v model %q", caller.last.Hop, caller.last.Model)
	}
	header := caller.last.Request.Header
	if header.Get(":path") != "/v1/chat/completions" || header.Get("authorization") != "Bearer key-a" || header.Get("x-static") != "static-a" {
		t.Fatalf("header %+v", header)
	}
	for _, field := range header {
		if strings.HasPrefix(field.Name, "x-vsr-") {
			t.Fatalf("a hop carries no internal header, got %s", field.Name)
		}
	}
	if !strings.Contains(string(caller.last.Request.Body), `"model":"panel-a"`) {
		t.Fatalf("body %s", caller.last.Request.Body)
	}
}

// A failed hop reads as the same failure over HTTP reads, since traces that
// reach the client quote the error.
func TestHopClientFailsAsTheConnectorDoes(t *testing.T) {
	errorBody := strings.Repeat("e", int(maxErrorBodyBytes)+10)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(errorBody))
	}))
	defer server.Close()
	cfg := &config.LooperConfig{Endpoint: server.URL}
	overHTTP, err := NewConnectorClient(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer overHTTP.Close()
	_, httpErr := overHTTP.CallModelWithOptions(context.Background(),
		openai.ChatCompletionNewParams{Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")}},
		ModelTarget{Name: "panel-a"}, CallOptions{Iteration: 1})

	_, hopErr := callAsHop(t, cfg, &hopCaller{resp: &graph.HopResponse{Status: http.StatusServiceUnavailable, Body: []byte(errorBody)}})
	if httpErr == nil || hopErr == nil || httpErr.Error() != hopErr.Error() {
		t.Fatalf("errors differ:\nhttp: %v\nhop:  %v", httpErr, hopErr)
	}
}

func TestHopClientBoundsEachCallByTheLooperTimeout(t *testing.T) {
	started := time.Now()
	_, err := callAsHop(t, &config.LooperConfig{TimeoutSeconds: 1}, &hopCaller{wait: true})
	if !errors.Is(err, context.DeadlineExceeded) || attemptReasonFromError(err) != AttemptReasonDeadline {
		t.Fatalf("err %v", err)
	}
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("the call ran %v", elapsed)
	}
}

func TestTemplateRunsTheAlgorithmAsOneStep(t *testing.T) {
	program, err := Template(&config.LooperConfig{}, config.DecisionAlgorithmRatings, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(program.Steps) != 2 || program.Steps[0].Type != StepType || program.Steps[1].Type != graph.TypeRespond {
		t.Fatalf("steps %+v", program.Steps)
	}
	if _, unsupported := Template(&config.LooperConfig{}, "static", nil); unsupported == nil {
		t.Fatal("a selection algorithm has no Looper template")
	}
	_, err = graph.Run(context.Background(), program, graph.Input{}, graph.Options{Caller: &hopCaller{}})
	if !errors.Is(err, ErrNoRequest) {
		t.Fatalf("a run without a Looper request: %v", err)
	}
}
