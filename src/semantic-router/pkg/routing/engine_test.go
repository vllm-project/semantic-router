package routing

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

// scriptedProcessor opens one session whose phases answer with fixed effects.
type scriptedProcessor struct {
	session *scriptedSession
}

func (p *scriptedProcessor) Open(context.Context) (Session, error) { return p.session, nil }

type phaseCall struct {
	phase       Phase
	endOfStream bool
	header      Header
	body        string
}

type scriptedSession struct {
	effects map[Phase][]*Effect
	errs    map[Phase]error
	calls   []phaseCall
	closed  []error
}

func (s *scriptedSession) answer(call phaseCall) (*Effect, error) {
	s.calls = append(s.calls, call)
	if err := s.errs[call.phase]; err != nil {
		return nil, err
	}
	queue := s.effects[call.phase]
	if len(queue) == 0 {
		return &Effect{}, nil
	}
	s.effects[call.phase] = queue[1:]
	return queue[0], nil
}

func (s *scriptedSession) RequestHeaders(h Header, eos bool) (*Effect, error) {
	return s.answer(phaseCall{phase: PhaseRequestHeaders, endOfStream: eos, header: h})
}

func (s *scriptedSession) RequestBody(b []byte, eos bool) (*Effect, error) {
	return s.answer(phaseCall{phase: PhaseRequestBody, endOfStream: eos, body: string(b)})
}

func (s *scriptedSession) ResponseHeaders(h Header, eos bool) (*Effect, error) {
	return s.answer(phaseCall{phase: PhaseResponseHeaders, endOfStream: eos, header: h})
}

func (s *scriptedSession) ResponseBody(b []byte, eos bool) (*Effect, error) {
	return s.answer(phaseCall{phase: PhaseResponseBody, endOfStream: eos, body: string(b)})
}

func (s *scriptedSession) Evidence() Evidence { return Evidence{Decision: "d"} }

func (s *scriptedSession) Close(err error) { s.closed = append(s.closed, err) }

func newScripted(effects map[Phase][]*Effect) (*scriptedSession, Engine) {
	session := &scriptedSession{effects: effects, errs: map[Phase]error{}}
	return session, NewEngine(&scriptedProcessor{session: session}, DefaultOptions)
}

func chatRequest(body string) *Request {
	return &Request{
		Header: header(":method", "POST", ":path", "/v1/chat/completions", ":authority", "router", "content-length", "4", "x-selected-model", "client-pick"),
		Body:   []byte(body),
	}
}

func set(name, value string) HeaderOption { return HeaderOption{Name: name, Value: value} }

func TestPlanAppliesRequestEffectsAndPicksTheRoute(t *testing.T) {
	session, engine := newScripted(map[Phase][]*Effect{
		PhaseRequestHeaders: {{Header: &HeaderMutation{Set: []HeaderOption{set("accept-encoding", "identity")}}}},
		PhaseRequestBody: {{
			Header:          &HeaderMutation{Set: []HeaderOption{set("x-selected-model", "model-a"), set(":path", "/v1/messages"), set("content-length", "7")}},
			Body:            &BodyMutation{Body: []byte("rewired")},
			ClearRouteCache: true,
		}},
	})
	plan, err := engine.Plan(context.Background(), chatRequest("ping"))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Immediate != nil || plan.Call == nil {
		t.Fatalf("expected a call, got %+v", plan)
	}
	if plan.Call.Route != "model-a" {
		t.Fatalf("route = %q; a cleared route cache must route on the mutated header", plan.Call.Route)
	}
	request := plan.Call.Request
	if request.Header.Get(":path") != "/v1/messages" || request.Header.Get("accept-encoding") != "identity" ||
		request.Header.Get("content-length") != "7" || string(request.Body) != "rewired" {
		t.Fatalf("upstream request = %v %q", request.Header, request.Body)
	}
	if len(session.calls) != 2 || session.calls[0].endOfStream || !session.calls[1].endOfStream {
		t.Fatalf("phases = %+v", session.calls)
	}
	if plan.Evidence.Decision != "d" {
		t.Fatalf("evidence = %+v", plan.Evidence)
	}
	plan.Finish(nil)
	plan.Finish(errors.New("ignored"))
	if len(session.closed) != 1 || session.closed[0] != nil {
		t.Fatalf("Finish must close the session exactly once, got %v", session.closed)
	}
}

func TestPlanKeepsTheOriginalRouteWithoutARouteCacheClear(t *testing.T) {
	_, engine := newScripted(map[Phase][]*Effect{
		PhaseRequestBody: {
			{Header: &HeaderMutation{Set: []HeaderOption{set("x-selected-model", "model-a"), set("content-length", "4")}}},
		},
	})
	plan, err := engine.Plan(context.Background(), chatRequest("ping"))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Call.Route != "client-pick" {
		t.Fatalf("route = %q, want the route chosen from the original headers", plan.Call.Route)
	}

	_, engine = newScripted(map[Phase][]*Effect{
		PhaseRequestBody: {{ClearRouteCache: true}},
	})
	plan, err = engine.Plan(context.Background(), chatRequest("ping"))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Call.Route != "client-pick" {
		t.Fatalf("a clear without a header mutation is ignored, route = %q", plan.Call.Route)
	}
}

func TestPlanAnswersImmediatelyWithoutTheBodyPhase(t *testing.T) {
	session, engine := newScripted(map[Phase][]*Effect{
		PhaseRequestHeaders: {{Immediate: &ImmediateResponse{Status: 404, Body: []byte("nope")}}},
	})
	plan, err := engine.Plan(context.Background(), chatRequest("ping"))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Call != nil || plan.Immediate == nil || plan.Immediate.Status != 404 {
		t.Fatalf("plan = %+v", plan)
	}
	if len(session.calls) != 1 {
		t.Fatalf("the body phase must not run after an immediate response: %+v", session.calls)
	}
}

func TestPlanFailsOnAMismatchedContentLength(t *testing.T) {
	session, engine := newScripted(map[Phase][]*Effect{
		PhaseRequestBody: {{Body: &BodyMutation{Body: []byte("longer body")}}},
	})
	if _, err := engine.Plan(context.Background(), chatRequest("ping")); !errors.Is(err, ErrMutation) {
		t.Fatalf("err = %v, want ErrMutation", err)
	}
	if len(session.closed) != 1 || session.closed[0] == nil {
		t.Fatalf("a failed plan must close its session with the error, got %v", session.closed)
	}
}

func TestPlanSendsBodylessRequestsAsHeadersOnly(t *testing.T) {
	session, engine := newScripted(nil)
	plan, err := engine.Plan(context.Background(), &Request{Header: header(":method", "GET", ":path", "/v1/models")})
	if err != nil {
		t.Fatal(err)
	}
	if len(session.calls) != 1 || !session.calls[0].endOfStream || plan.Call.Route != "" {
		t.Fatalf("calls = %+v, route %q", session.calls, plan.Call.Route)
	}
}

func TestRespondBuffersTheBodyAndAppliesBothPhases(t *testing.T) {
	session, engine := newScripted(map[Phase][]*Effect{
		PhaseResponseHeaders: {{Header: &HeaderMutation{Set: []HeaderOption{set("x-vsr-selected-model", "model-a")}, Remove: []string{"content-length"}}}},
		PhaseResponseBody: {{
			Header: &HeaderMutation{Set: []HeaderOption{set("x-vsr-cost", "1")}},
			Body:   &BodyMutation{Body: []byte("translated")},
		}},
	})
	plan, err := engine.Plan(context.Background(), chatRequest("ping"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := engine.Respond(context.Background(), plan, &UpstreamResponse{
		Status: 200,
		Header: header("Content-Type", "application/json", "content-length", "8"),
		Body:   strings.NewReader("original"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Status != 200 || resp.Stream != nil || string(resp.Body) != "translated" {
		t.Fatalf("response = %d %q", resp.Status, resp.Body)
	}
	if resp.Header.Get("x-vsr-selected-model") != "model-a" || resp.Header.Get("x-vsr-cost") != "1" ||
		resp.Header.Has("content-length") || resp.Header.Has(":status") {
		t.Fatalf("headers = %v", resp.Header)
	}
	last := session.calls[len(session.calls)-1]
	if last.phase != PhaseResponseBody || !last.endOfStream || last.body != "original" {
		t.Fatalf("body phase = %+v", last)
	}
	if session.calls[2].header.Get(":status") != "200" || session.calls[2].header.Get("content-type") != "application/json" {
		t.Fatalf("response headers phase saw %v", session.calls[2].header)
	}
}

type chunkReader struct{ chunks []string }

func (r *chunkReader) Read(p []byte) (int, error) {
	if len(r.chunks) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.chunks[0])
	r.chunks = r.chunks[1:]
	return n, nil
}

func TestRespondStreamsAfterAModeOverride(t *testing.T) {
	session, engine := newScripted(map[Phase][]*Effect{
		PhaseResponseHeaders: {{
			Header:           &HeaderMutation{Set: []HeaderOption{set("content-length", "99"), set("x-a", "1")}, Remove: []string{"content-length"}},
			ResponseBodyMode: BodyModeStreamed,
		}},
		PhaseResponseBody: {
			{Body: &BodyMutation{Body: []byte("A")}},
			{Header: &HeaderMutation{Set: []HeaderOption{set("x-late", "ignored")}}},
			{Body: &BodyMutation{Body: []byte("[DONE]")}},
		},
	})
	plan, err := engine.Plan(context.Background(), chatRequest("ping"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := engine.Respond(context.Background(), plan, &UpstreamResponse{
		Status: 200,
		Header: header("content-type", "text/event-stream", "content-length", "2"),
		Body:   &chunkReader{chunks: []string{"a", "b"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Stream == nil || resp.Body != nil || resp.Header.Has("content-length") || resp.Header.Get("x-a") != "1" {
		t.Fatalf("streamed response = %+v", resp)
	}
	var chunks []string
	for {
		chunk, err := resp.Stream.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, string(chunk))
	}
	if strings.Join(chunks, "|") != "A|b|[DONE]" {
		t.Fatalf("chunks = %q", chunks)
	}
	if resp.Header.Has("x-late") {
		t.Fatal("header mutations after the headers were sent must be ignored")
	}
	bodyCalls := session.calls[3:]
	if len(bodyCalls) != 3 || bodyCalls[0].endOfStream || !bodyCalls[2].endOfStream || bodyCalls[2].body != "" {
		t.Fatalf("streamed body phases = %+v", bodyCalls)
	}
}

func TestRespondRejectsAnImmediateResponseMidStream(t *testing.T) {
	_, engine := newScripted(map[Phase][]*Effect{
		PhaseResponseHeaders: {{ResponseBodyMode: BodyModeStreamed}},
		PhaseResponseBody:    {{Immediate: &ImmediateResponse{Status: 500}}},
	})
	plan, _ := engine.Plan(context.Background(), chatRequest("ping"))
	resp, err := engine.Respond(context.Background(), plan, &UpstreamResponse{Status: 200, Body: &chunkReader{chunks: []string{"a"}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resp.Stream.Next(); !errors.Is(err, ErrResponseCommitted) {
		t.Fatalf("err = %v, want ErrResponseCommitted", err)
	}
}

func TestRespondReplacesTheUpstreamWithAnImmediateResponse(t *testing.T) {
	_, engine := newScripted(map[Phase][]*Effect{
		PhaseResponseHeaders: {{Immediate: &ImmediateResponse{Status: 200, Body: []byte(`{"fallback":true}`)}}},
	})
	plan, _ := engine.Plan(context.Background(), chatRequest("ping"))
	resp, err := engine.Respond(context.Background(), plan, &UpstreamResponse{Status: 503, Body: strings.NewReader("busy")})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Status != 200 || string(resp.Body) != `{"fallback":true}` {
		t.Fatalf("response = %d %q", resp.Status, resp.Body)
	}
}

func TestRespondBoundsTheBufferedBody(t *testing.T) {
	session := &scriptedSession{errs: map[Phase]error{}}
	engine := NewEngine(&scriptedProcessor{session: session}, Options{MaxBufferedBodyBytes: 3})
	plan, _ := engine.Plan(context.Background(), &Request{Header: header(":method", "GET")})
	if _, err := engine.Respond(context.Background(), plan, &UpstreamResponse{Status: 200, Body: strings.NewReader("four")}); !errors.Is(err, ErrBufferLimit) {
		t.Fatalf("err = %v, want ErrBufferLimit", err)
	}
}

func TestPlanCarriesTheDeadline(t *testing.T) {
	_, engine := newScripted(nil)
	ctx, cancel := context.WithTimeout(context.Background(), 1<<30)
	defer cancel()
	plan, err := engine.Plan(ctx, &Request{Header: header(":method", "GET")})
	if err != nil {
		t.Fatal(err)
	}
	if deadline, _ := ctx.Deadline(); !plan.Budget.Deadline.Equal(deadline) {
		t.Fatalf("budget deadline = %v, want %v", plan.Budget.Deadline, deadline)
	}
}

type fallbackSession struct {
	*scriptedSession
	handedOut int
}

func (s *fallbackSession) Fallback() Fallback {
	s.handedOut++
	return fallbackFunc(func(context.Context, Outcome) (FallbackStep, error) { return FallbackStep{}, nil })
}

type fallbackFunc func(context.Context, Outcome) (FallbackStep, error)

func (f fallbackFunc) Next(ctx context.Context, o Outcome) (FallbackStep, error) { return f(ctx, o) }

type fallbackProcessor struct{ session *fallbackSession }

func (p *fallbackProcessor) Open(context.Context) (Session, error) { return p.session, nil }

func TestPlanAttachesTheSessionFallbackOnlyWhenTheCallerExecutesIt(t *testing.T) {
	session := &fallbackSession{scriptedSession: &scriptedSession{errs: map[Phase]error{}}}
	plan, err := NewEngine(&fallbackProcessor{session: session}, DefaultOptions).Plan(context.Background(), chatRequest("ping"))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Call.Fallback != nil || session.handedOut != 0 {
		t.Fatal("an engine behind Envoy must leave fallback to the session's response phases")
	}
	opts := DefaultOptions
	opts.ExecutesFallback = true
	plan, err = NewEngine(&fallbackProcessor{session: session}, opts).Plan(context.Background(), chatRequest("ping"))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Call.Fallback == nil || session.handedOut != 1 {
		t.Fatal("an engine whose caller executes fallback must attach the session's fallback")
	}
}

type reliabilitySession struct {
	*scriptedSession
	reliability *Reliability
}

func (s *reliabilitySession) Reliability() *Reliability { return s.reliability }

type reliabilityProcessor struct{ session *reliabilitySession }

func (p *reliabilityProcessor) Open(context.Context) (Session, error) { return p.session, nil }

func TestPlanCarriesTheSessionReliabilityOverride(t *testing.T) {
	total := 5 * time.Second
	want := &Reliability{TotalTimeout: &total, RetryOn: []string{"reset"}}
	session := &reliabilitySession{scriptedSession: &scriptedSession{errs: map[Phase]error{}}, reliability: want}
	plan, err := NewEngine(&reliabilityProcessor{session: session}, DefaultOptions).Plan(context.Background(), chatRequest("ping"))
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Call.Reliability) != 1 || plan.Call.Reliability[0] != want {
		t.Fatalf("call reliability = %+v, want the session's", plan.Call.Reliability)
	}
}
