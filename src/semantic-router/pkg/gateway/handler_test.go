package gateway

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// fakeProcessor answers each phase with a scripted effect and records what it saw.
type fakeProcessor struct {
	mu       sync.Mutex
	effects  map[routing.Phase][]*routing.Effect
	headers  []routing.Header
	bodies   [][]byte
	closed   []error
	planFail error
	models   []routing.ListenerModels
}

func (p *fakeProcessor) Open(ctx context.Context) (routing.Session, error) {
	models, _ := routing.ListenerModelsFrom(ctx)
	p.mu.Lock()
	p.models = append(p.models, models)
	p.mu.Unlock()
	return &fakeSession{p: p}, nil
}

type fakeSession struct{ p *fakeProcessor }

func (s *fakeSession) next(phase routing.Phase, h routing.Header, body []byte) (*routing.Effect, error) {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()
	if h != nil {
		s.p.headers = append(s.p.headers, h)
	}
	if body != nil {
		s.p.bodies = append(s.p.bodies, body)
	}
	if phase == routing.PhaseRequestHeaders && s.p.planFail != nil {
		return nil, s.p.planFail
	}
	queue := s.p.effects[phase]
	if len(queue) == 0 {
		return &routing.Effect{}, nil
	}
	s.p.effects[phase] = queue[1:]
	return queue[0], nil
}

func (s *fakeSession) RequestHeaders(h routing.Header, _ bool) (*routing.Effect, error) {
	return s.next(routing.PhaseRequestHeaders, h, nil)
}

func (s *fakeSession) RequestBody(b []byte, _ bool) (*routing.Effect, error) {
	return s.next(routing.PhaseRequestBody, nil, b)
}

func (s *fakeSession) ResponseHeaders(h routing.Header, _ bool) (*routing.Effect, error) {
	return s.next(routing.PhaseResponseHeaders, h, nil)
}

func (s *fakeSession) ResponseBody(b []byte, _ bool) (*routing.Effect, error) {
	return s.next(routing.PhaseResponseBody, nil, b)
}

func (s *fakeSession) Evidence() routing.Evidence { return routing.Evidence{} }

func (s *fakeSession) Close(err error) {
	s.p.mu.Lock()
	defer s.p.mu.Unlock()
	s.p.closed = append(s.p.closed, err)
}

// fakeUpstream returns a scripted response and records the request.
type fakeUpstream struct {
	mu        sync.Mutex
	got       []*upstream.Request
	resp      func() *upstream.Response
	immediate *routing.Response
	err       error
}

func (u *fakeUpstream) Execute(_ context.Context, call *routing.Call, listener string) (*upstream.Result, error) {
	u.mu.Lock()
	u.got = append(u.got, upstream.RequestFromCall(call, listener))
	u.mu.Unlock()
	if u.err != nil {
		return nil, u.err
	}
	if u.immediate != nil {
		return &upstream.Result{Immediate: u.immediate, Hops: 2}, nil
	}
	return &upstream.Result{Response: u.resp(), Hops: 1}, nil
}

func newTestHandler(t *testing.T, p *fakeProcessor, u *fakeUpstream, keys ...string) *Handler {
	t.Helper()
	h, err := NewHandler(Options{
		Serving:  Static(Serving{Engine: routing.NewEngine(p, routing.DefaultOptions), Upstream: u, APIKeys: keys}),
		Listener: "http-8899",
	})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func okUpstream(body string, header http.Header) func() *upstream.Response {
	return func() *upstream.Response {
		return &upstream.Response{StatusCode: 200, Header: header.Clone(), Body: io.NopCloser(strings.NewReader(body))}
	}
}

func set(name, value string) routing.HeaderOption {
	return routing.HeaderOption{Name: name, Value: value}
}

func TestHandlerBuildsTheRequestLikeEnvoy(t *testing.T) {
	p := &fakeProcessor{}
	u := &fakeUpstream{resp: okUpstream(`{"ok":true}`, http.Header{"Content-Type": {"application/json"}})}
	server := httptest.NewServer(newTestHandler(t, p, u))
	defer server.Close()

	req, _ := http.NewRequest(http.MethodPost, server.URL+"/v1/chat/completions?trace=1", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Authz-User-Id", "spoofed")
	req.Header.Set("Connection", "keep-alive, X-Drop-Me")
	req.Header.Set("X-Drop-Me", "1")
	req.Header.Add("X-Multi", "a")
	req.Header.Add("X-Multi", "b")
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()

	h := p.headers[0]
	if h.Get(":method") != "POST" || h.Get(":path") != "/v1/chat/completions?trace=1" || h.Get(":scheme") != "http" ||
		!strings.HasPrefix(h.Get(":authority"), "127.0.0.1:") {
		t.Fatalf("pseudo-headers = %v", h)
	}
	if h.Has("x-authz-user-id") || h.Has("connection") || h.Has("x-drop-me") || h.Has("host") {
		t.Fatalf("identity and hop-by-hop headers must not reach the engine: %v", h)
	}
	if got := h.Values("x-multi"); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("repeated headers = %v", got)
	}
	if h.Get("x-forwarded-proto") != "http" || len(h.Get("x-request-id")) != 36 || h.Get("content-length") != "13" {
		t.Fatalf("Envoy-added headers = %v", h)
	}
	if string(p.bodies[0]) != `{"model":"m"}` {
		t.Fatalf("body = %q", p.bodies[0])
	}
	if len(u.got) != 1 || u.got[0].Path != "/v1/chat/completions?trace=1" || u.got[0].Listener != "http-8899" {
		t.Fatalf("upstream request = %+v", u.got)
	}
	if len(p.closed) != 1 || p.closed[0] != nil {
		t.Fatalf("the request must finish once without error, got %v", p.closed)
	}
}

func TestHandlerDropsTheProxyControlHeadersOfClients(t *testing.T) {
	p := &fakeProcessor{}
	u := &fakeUpstream{resp: okUpstream("ok", http.Header{})}
	server := httptest.NewServer(newTestHandler(t, p, u))
	defer server.Close()
	removed := []string{
		"X-Envoy-Internal", "X-Envoy-Retriable-Status-Codes", "X-Envoy-Retriable-Header-Names", "X-Envoy-Retry-On",
		"X-Envoy-Retry-Grpc-On", "X-Envoy-Max-Retries", "X-Envoy-Upstream-Alt-Stat-Name",
		"X-Envoy-Upstream-Rq-Timeout-Ms", "X-Envoy-Upstream-Rq-Per-Try-Timeout-Ms",
		"X-Envoy-Upstream-Rq-Timeout-Alt-Response", "X-Envoy-Expected-Rq-Timeout-Ms", "X-Envoy-Force-Trace",
		"X-Envoy-Ip-Tags", "X-Envoy-Original-Url", "X-Envoy-Hedge-On-Per-Try-Timeout",
	}
	edgeOnly := []string{
		"X-Envoy-Decorator-Operation", "X-Envoy-Downstream-Service-Cluster", "X-Envoy-Downstream-Service-Node",
		"X-Envoy-Original-Path", "X-Envoy-Original-Host",
	}
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	for _, name := range append(append([]string(nil), removed...), edgeOnly...) {
		req.Header.Set(name, "client")
	}
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	h := p.headers[0]
	for _, name := range removed {
		if h.Has(name) {
			t.Errorf("%s reached the engine", name)
		}
	}
	for _, name := range edgeOnly {
		if h.Get(name) != "client" {
			t.Errorf("%s is an edge-only header the template keeps, got %q", name, h.Get(name))
		}
	}
	if forwarded := u.got[0].Header; forwarded.Get("X-Envoy-Max-Retries") != "" || forwarded.Get("X-Envoy-Internal") != "" {
		t.Fatalf("a client control header reached the backend: %v", forwarded)
	}
}

func TestHandlerKeepsAClientRequestID(t *testing.T) {
	p := &fakeProcessor{}
	u := &fakeUpstream{resp: okUpstream("ok", http.Header{})}
	server := httptest.NewServer(newTestHandler(t, p, u))
	defer server.Close()
	req, _ := http.NewRequest(http.MethodGet, server.URL+"/v1/models", nil)
	req.Header.Set("X-Request-Id", "client-id")
	req.Header.Set("X-Forwarded-Proto", "https")
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if h := p.headers[0]; h.Get("x-request-id") != "client-id" || h.Get("x-forwarded-proto") != "https" {
		t.Fatalf("headers = %v", h)
	}
}

func TestHandlerChecksAPIKeysLikeTheTemplate(t *testing.T) {
	tests := []struct {
		name   string
		header http.Header
		status int
	}{
		{"bearer", http.Header{"Authorization": {"Bearer sk-1"}}, 200},
		{"lowercase bearer", http.Header{"Authorization": {"bearer   sk-1"}}, 200},
		{"azure api-key", http.Header{"Api-Key": {"sk-1"}}, 200},
		{"bad bearer falls back to api-key", http.Header{"Authorization": {"Bearer wrong"}, "Api-Key": {"sk-1"}}, 200},
		{"wrong key", http.Header{"Authorization": {"Bearer wrong"}}, 401},
		{"no space", http.Header{"Authorization": {"Bearersk-1"}}, 401},
		{"BEARER is not a match", http.Header{"Authorization": {"BEARER sk-1"}}, 401},
		{"missing", http.Header{}, 401},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := &fakeProcessor{}
			u := &fakeUpstream{resp: okUpstream("ok", http.Header{})}
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{}"))
			req.Header = test.header.Clone()
			newTestHandler(t, p, u, "sk-1", "sk-2").ServeHTTP(rec, req)
			if rec.Code != test.status {
				t.Fatalf("status = %d, want %d", rec.Code, test.status)
			}
			if test.status == 401 {
				if rec.Body.String() != unauthorizedBody || rec.Header().Get("WWW-Authenticate") != `Bearer realm="vllm-semantic-router"` || len(p.headers) != 0 {
					t.Fatalf("401 = %q %v; the engine saw %d requests", rec.Body.String(), rec.Header(), len(p.headers))
				}
				return
			}
			if h := p.headers[0]; h.Has("authorization") || h.Has("api-key") {
				t.Fatalf("client credentials must not reach the engine: %v", h)
			}
		})
	}
}

func TestHandlerHandsTheListenersModelsToTheRoutingCore(t *testing.T) {
	for _, models := range [][]string{nil, {"vllm-sr/auto"}} {
		p := &fakeProcessor{}
		u := &fakeUpstream{resp: okUpstream("ok", http.Header{})}
		h, err := NewHandler(Options{
			Serving: Static(Serving{
				Engine: routing.NewEngine(p, routing.DefaultOptions), Upstream: u,
				APIKeys: []string{"sk-1"}, Models: models,
			}),
			Listener: "public",
		})
		if err != nil {
			t.Fatal(err)
		}
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"a"}`))
		h.ServeHTTP(httptest.NewRecorder(), req)
		if len(p.models) != 0 {
			t.Fatalf("models %v: a request without a key reached the routing core", models)
		}
		req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"a"}`))
		req.Header.Set("Authorization", "Bearer sk-1")
		h.ServeHTTP(httptest.NewRecorder(), req)
		if len(p.models) != 1 || len(p.models[0]) != len(models) || (len(models) > 0 && !p.models[0].Allows(models[0])) {
			t.Fatalf("the routing core saw allow-list %v, want %v", p.models, models)
		}
	}
}

func TestHandlerWritesImmediateResponses(t *testing.T) {
	p := &fakeProcessor{effects: map[routing.Phase][]*routing.Effect{
		routing.PhaseRequestHeaders: {{Immediate: &routing.ImmediateResponse{
			Status: 404,
			Header: &routing.HeaderMutation{Set: []routing.HeaderOption{set("content-type", "application/json")}},
			Body:   []byte(`{"error":{}}`),
		}}},
	}}
	u := &fakeUpstream{resp: okUpstream("unused", http.Header{})}
	rec := httptest.NewRecorder()
	newTestHandler(t, p, u).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/router_replay", nil))
	if rec.Code != 404 || rec.Body.String() != `{"error":{}}` || rec.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("response = %d %q %v", rec.Code, rec.Body.String(), rec.Header())
	}
	if len(u.got) != 0 || len(p.closed) != 1 || p.closed[0] != nil {
		t.Fatalf("an immediate answer never goes upstream: upstream %d, closed %v", len(u.got), p.closed)
	}
}

func TestHandlerWritesTheFallbackChainsAnswerAsIs(t *testing.T) {
	p := &fakeProcessor{}
	u := &fakeUpstream{immediate: &routing.Response{
		Status: http.StatusTooManyRequests,
		Header: routing.Header{{Name: "content-type", Value: "application/json"}},
		Body:   []byte(`{"error":{"message":"busy"}}`),
	}}
	rec := httptest.NewRecorder()
	newTestHandler(t, p, u).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m"}`)))
	if rec.Code != http.StatusTooManyRequests || rec.Body.String() != `{"error":{"message":"busy"}}` {
		t.Fatalf("response = %d %q, want the chain's answer", rec.Code, rec.Body.String())
	}
	if len(p.headers) != 1 {
		t.Fatalf("phases saw %d header sets, want the request's only: the answer is already the client's", len(p.headers))
	}
}

func TestHandlerForwardsTheCallAndRunsResponsePhases(t *testing.T) {
	p := &fakeProcessor{effects: map[routing.Phase][]*routing.Effect{
		routing.PhaseRequestBody: {{
			Header:          &routing.HeaderMutation{Set: []routing.HeaderOption{set("x-selected-model", "model-a"), set(":path", "/v1/messages"), set("content-length", "4")}},
			Body:            &routing.BodyMutation{Body: []byte("next")},
			ClearRouteCache: true,
		}},
		routing.PhaseResponseHeaders: {{Header: &routing.HeaderMutation{Set: []routing.HeaderOption{set("x-vsr-selected-model", "model-a")}}}},
		routing.PhaseResponseBody:    {{Body: &routing.BodyMutation{Body: []byte("translated")}}},
	}}
	u := &fakeUpstream{resp: okUpstream("original", http.Header{"Content-Type": {"application/json"}})}
	rec := httptest.NewRecorder()
	newTestHandler(t, p, u).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("ping")))
	call := u.got[0]
	if call.RouteKey != "model-a" || call.Path != "/v1/messages" || string(call.Body) != "next" || call.Header.Get("X-Selected-Model") != "model-a" {
		t.Fatalf("upstream call = %+v", call)
	}
	if _, pseudo := call.Header[":path"]; pseudo {
		t.Fatal("pseudo-headers must not be sent as headers")
	}
	if rec.Code != 200 || rec.Body.String() != "translated" || rec.Header().Get("X-Vsr-Selected-Model") != "model-a" {
		t.Fatalf("client response = %d %q %v", rec.Code, rec.Body.String(), rec.Header())
	}
	if string(p.bodies[1]) != "original" {
		t.Fatalf("the response body phase saw %q", p.bodies[1])
	}
}

func TestHandlerStreamsChunkByChunk(t *testing.T) {
	p := &fakeProcessor{effects: map[routing.Phase][]*routing.Effect{
		routing.PhaseResponseHeaders: {{ResponseBodyMode: routing.BodyModeStreamed}},
	}}
	release := make(chan struct{})
	pr, pw := io.Pipe()
	go func() {
		_, _ = pw.Write([]byte("data: one\n\n"))
		<-release
		_, _ = pw.Write([]byte("data: [DONE]\n\n"))
		_ = pw.Close()
	}()
	u := &fakeUpstream{resp: func() *upstream.Response {
		return &upstream.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: pr}
	}}
	server := httptest.NewServer(newTestHandler(t, p, u))
	defer server.Close()
	resp, err := server.Client().Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader("{}"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	reader := bufio.NewReader(resp.Body)
	line, err := reader.ReadString('\n')
	if err != nil || line != "data: one\n" {
		t.Fatalf("the first chunk must arrive before the stream ends: %q %v", line, err)
	}
	close(release)
	rest, _ := io.ReadAll(reader)
	if string(rest) != "\ndata: [DONE]\n\n" {
		t.Fatalf("rest = %q", rest)
	}
	if resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("headers = %v", resp.Header)
	}
}

// Envoy's ext_proc ends processing on Envoy's own local replies, so the
// response phases never see one: the client gets it as it is, and the session
// ends as an ext_proc stream ends.
func TestHandlerWritesLocalRepliesAsEnvoyDoes(t *testing.T) {
	const connectFailure = "upstream connect error or disconnect/reset before headers. reset reason: remote connection failure"
	for name, u := range map[string]*fakeUpstream{
		"a walker error": {err: &upstream.Error{Kind: upstream.KindTimeout, Stage: upstream.StageTotal}},
		"an upstream local reply": {resp: func() *upstream.Response {
			return &upstream.Response{
				StatusCode: http.StatusServiceUnavailable,
				Header:     http.Header{"Content-Type": {"text/plain"}, "Content-Length": {strconv.Itoa(len(connectFailure))}},
				Body:       io.NopCloser(strings.NewReader(connectFailure)),
				Local:      &upstream.Error{Kind: upstream.KindConnectFailure},
			}
		}},
	} {
		t.Run(name, func(t *testing.T) {
			p := &fakeProcessor{effects: map[routing.Phase][]*routing.Effect{
				routing.PhaseResponseBody: {{Body: &routing.BodyMutation{Body: []byte("translated")}}},
			}}
			rec := httptest.NewRecorder()
			newTestHandler(t, p, u).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{}")))
			want := map[string]string{"a walker error": "Gateway Timeout", "an upstream local reply": connectFailure}[name]
			if rec.Body.String() != want || rec.Header().Get("Content-Type") != "text/plain" {
				t.Fatalf("response = %d %q %v, want the local reply as it is", rec.Code, rec.Body.String(), rec.Header())
			}
			if len(p.headers) != 1 || len(p.bodies) != 1 {
				t.Fatalf("phases saw %d header sets and %d bodies, want the request's only", len(p.headers), len(p.bodies))
			}
			if len(p.closed) != 1 || p.closed[0] != nil {
				t.Fatalf("session closed with %v, want one clean end", p.closed)
			}
		})
	}
}

func TestHandlerAnswersRoutingFailuresWith500(t *testing.T) {
	p := &fakeProcessor{planFail: errors.New("boom")}
	u := &fakeUpstream{resp: okUpstream("unused", http.Header{})}
	rec := httptest.NewRecorder()
	newTestHandler(t, p, u).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{}")))
	if rec.Code != 500 || !strings.Contains(rec.Body.String(), `"code":"routing_error"`) || len(u.got) != 0 {
		t.Fatalf("response = %d %q", rec.Code, rec.Body.String())
	}
}

func TestHandlerRejectsOversizedBodies(t *testing.T) {
	p := &fakeProcessor{}
	u := &fakeUpstream{resp: okUpstream("unused", http.Header{})}
	h, err := NewHandler(Options{
		Serving:             Static(Serving{Engine: routing.NewEngine(p, routing.DefaultOptions), Upstream: u}),
		MaxRequestBodyBytes: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("four")))
	if rec.Code != http.StatusRequestEntityTooLarge || len(p.headers) != 0 {
		t.Fatalf("response = %d", rec.Code)
	}
}

func TestHandlerFinishesWithTheClientCancellation(t *testing.T) {
	p := &fakeProcessor{effects: map[routing.Phase][]*routing.Effect{
		routing.PhaseResponseHeaders: {{ResponseBodyMode: routing.BodyModeStreamed}},
	}}
	ctx, cancel := context.WithCancel(context.Background())
	pr, pw := io.Pipe()
	u := &fakeUpstream{resp: func() *upstream.Response {
		go func() {
			_, _ = pw.Write([]byte("data: one\n\n"))
			cancel()
		}()
		return &upstream.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: pr}
	}}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{}")).WithContext(ctx)
	go func() {
		<-ctx.Done()
		_ = pw.CloseWithError(context.Canceled)
	}()
	rec := httptest.NewRecorder()
	newTestHandler(t, p, u).ServeHTTP(rec, req)
	if len(p.closed) != 1 || !errors.Is(p.closed[0], context.Canceled) {
		t.Fatalf("the request must finish with the cancellation, got %v", p.closed)
	}
}

func TestHandlerAnswersProbesWithoutRouting(t *testing.T) {
	p := &fakeProcessor{}
	u := &fakeUpstream{resp: okUpstream("unused", http.Header{})}
	ready := false
	h, err := NewHandler(Options{
		Serving: Static(Serving{Engine: routing.NewEngine(p, routing.DefaultOptions), Upstream: u, APIKeys: []string{"k"}}),
		Ready:   func() bool { return ready },
	})
	if err != nil {
		t.Fatal(err)
	}
	probe := func(path string) int {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		return rec.Code
	}
	if probe(HealthPath) != 200 || probe(ReadyPath) != 503 {
		t.Fatal("health must pass and readiness wait for the routing core")
	}
	ready = true
	if probe(ReadyPath) != 200 || len(p.headers) != 0 {
		t.Fatalf("probes must not need API keys or reach the engine, engine saw %d", len(p.headers))
	}
}

func TestHandlerWritesAnAccessRecord(t *testing.T) {
	p := &fakeProcessor{}
	u := &fakeUpstream{resp: func() *upstream.Response {
		return &upstream.Response{
			StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("hello")),
			Endpoint: upstream.EndpointSpec{Host: "192.0.2.10", Port: 8000},
		}
	}}
	var records []AccessRecord
	h, err := NewHandler(Options{
		Serving:   Static(Serving{Engine: routing.NewEngine(p, routing.DefaultOptions), Upstream: u}),
		AccessLog: func(r AccessRecord) { records = append(records, r) },
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("ping"))
	req.Header.Set("User-Agent", "client/1")
	h.ServeHTTP(httptest.NewRecorder(), req)
	if len(records) != 1 {
		t.Fatalf("records = %d", len(records))
	}
	r := records[0]
	if r.Status != 200 || r.BytesSent != 5 || r.BytesReceived != 4 || r.UpstreamHost != "192.0.2.10:8000" ||
		r.UserAgent != "client/1" || len(r.RequestID) != 36 || r.Method != "POST" || r.Path != "/v1/chat/completions" {
		t.Fatalf("record = %+v", r)
	}
}
