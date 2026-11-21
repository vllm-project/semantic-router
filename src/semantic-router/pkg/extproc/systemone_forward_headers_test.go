package extproc

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// headerRecordingGateway keeps the method and headers of every call before
// the fake System One gateway answers it.
type headerRecordingGateway struct {
	gateway *fakeSystemOneGateway
	mu      sync.Mutex
	calls   []recordedGatewayCall
}

type recordedGatewayCall struct {
	method string
	header http.Header
}

func (g *headerRecordingGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	g.mu.Lock()
	g.calls = append(g.calls, recordedGatewayCall{method: r.Method, header: r.Header.Clone()})
	g.mu.Unlock()
	g.gateway.ServeHTTP(w, r)
}

func (g *headerRecordingGateway) lastQuestion(t *testing.T) http.Header {
	t.Helper()
	g.mu.Lock()
	defer g.mu.Unlock()
	for i := len(g.calls) - 1; i >= 0; i-- {
		if g.calls[i].method == http.MethodPost {
			return g.calls[i].header
		}
	}
	t.Fatal("the gateway was asked no question")
	return nil
}

// A systemone deployment that forwards x-user-id receives the requesting
// client's value on the decision call made for that request, and no other
// client header; a request without it sends none, and a call the Router makes
// on its own behalf never carries it.
func TestSystemOneDecisionCallCarriesTheForwardedClientHeader(t *testing.T) {
	recorder := &headerRecordingGateway{gateway: &fakeSystemOneGateway{}}
	recorder.gateway.choice.Store("engineering")
	gatewayServer := httptest.NewServer(recorder)
	defer gatewayServer.Close()
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	defer backend.Close()
	t.Setenv("REMOTE_DECISION_TEST_KEY", "fake-gateway-key")
	document := strings.NewReplacer(
		"BACKEND", strings.TrimPrefix(backend.URL, "http://"), "GATEWAY", gatewayServer.URL,
		"        timeout_ms: 1500\n", "        timeout_ms: 1500\n        forward_request_headers: [x-user-id]\n",
	).Replace(systemOneRoutingConfig)
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true
	engine := routing.NewEngine(newLooperRouter(t, cfg), opts)
	route := func(user string) string {
		t.Helper()
		header := routing.Header{
			{Name: ":method", Value: "POST"},
			{Name: ":path", Value: "/v1/chat/completions"},
			{Name: ":authority", Value: "router"},
			{Name: ":scheme", Value: "http"},
			{Name: "content-type", Value: "application/json"},
			{Name: "x-other", Value: "client-only"},
		}
		if user != "" {
			header = append(header, routing.HeaderField{Name: "x-user-id", Value: user})
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		plan, err := engine.Plan(ctx, &routing.Request{Header: header, Body: []byte(
			`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"Our app has returned 500 errors since the 09:00 deploy."}]}`)})
		if err != nil {
			t.Fatalf("plan: %v", err)
		}
		defer plan.Finish(nil)
		if plan.Call == nil {
			t.Fatalf("the request was answered instead of routed: %+v", plan.Immediate)
		}
		return plan.Call.Route
	}

	for _, user := range []string{"alice", "bob"} {
		asked := recorder.gateway.asked.Load()
		if got := route(user); got != "deep" {
			t.Fatalf("%s: an engineering answer routes to deep, got %q", user, got)
		}
		if recorder.gateway.asked.Load() != asked+1 {
			t.Fatalf("%s: each request asks the deployment itself; no answer is kept for another request", user)
		}
		sent := recorder.lastQuestion(t)
		if got := sent.Values("X-User-Id"); len(got) != 1 || got[0] != user {
			t.Fatalf("the decision call for %s carried X-User-Id %v", user, got)
		}
		if sent.Get("X-Other") != "" || sent.Get("Ocp-Apim-Subscription-Key") != "fake-gateway-key" {
			t.Fatalf("the decision call carries no other client header and the Router's key: %v", sent)
		}
	}

	route("")
	if got := recorder.lastQuestion(t).Values("X-User-Id"); len(got) != 0 {
		t.Fatalf("a request without x-user-id sends none, got %v", got)
	}
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	for _, call := range recorder.calls {
		if call.method != http.MethodPost && call.header.Get("X-User-Id") != "" {
			t.Fatalf("a %s the Router makes on its own behalf carried X-User-Id", call.method)
		}
	}
}
