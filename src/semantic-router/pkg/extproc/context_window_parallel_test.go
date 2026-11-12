package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount"
)

// testAdmission is a window check of one model with a 1,000-token window
// whose tokenizer counts text exactly.
func testAdmission(t *testing.T, text string) (*OpenAIRouter, *RequestContext, *contextAdmission) {
	t.Helper()
	encoding, err := tokencount.Builtin(tokencount.O200kBase)
	require.NoError(t, err)
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	request := testNeutralRequest("model", text)
	admission := &contextAdmission{done: make(chan struct{}), started: time.Now(), generation: request.Generation}
	admission.parts, admission.reserve = promptParts(request)
	admission.models = []admissionModel{{
		model: "model", window: 1_000, tokenizer: modelTokenizer{encoding: encoding, kind: contextCountExact},
	}}
	return router, &RequestContext{RequestID: "req-admission", SemanticRequest: request, contextAdmission: admission}, admission
}

// requireGoroutinesBack waits for the goroutines a test started to end.
func requireGoroutinesBack(t *testing.T, baseline int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > baseline && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	require.LessOrEqual(t, runtime.NumGoroutine(), baseline, "a goroutine outlived the window check")
}

// A count that rules out every model cancels the signals running beside it,
// and the request is refused with that count; nothing it started outlives it.
func TestACountThatRulesOutEveryModelCancelsTheSignals(t *testing.T) {
	baseline := runtime.NumGoroutine()
	router, ctx, admission := testAdmission(t, promptOfTokens(t, 1_100))
	signals, release := admission.signals(context.Background())
	go admission.run(router, context.Background())

	select {
	case <-signals.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("the signals were not cancelled")
	}
	require.ErrorIs(t, context.Cause(signals), errContextWindowExceeded)
	release()
	refusal := router.contextAdmissionRefusal(ctx)
	require.NotNil(t, refusal)
	require.Equal(t, 1_000, refusal.window)
	require.Equal(t, routerCountOf(promptOfTokens(t, 1_100)), refusal.tokens)
	requireGoroutinesBack(t, baseline)
}

// Signals that start after the counts refused the request are cancelled at
// once.
func TestSignalsAfterARefusalAreCancelledAtOnce(t *testing.T) {
	router, ctx, admission := testAdmission(t, promptOfTokens(t, 1_100))
	admission.run(router, context.Background())
	signals, release := admission.signals(context.Background())
	defer release()
	require.ErrorIs(t, context.Cause(signals), errContextWindowExceeded)
	require.NotNil(t, router.contextAdmissionRefusal(ctx))
}

// A count that fits leaves the signals running, and the window checks after
// the decision reuse it without counting again.
func TestAFittingCountLeavesTheSignalsAndIsReused(t *testing.T) {
	baseline := runtime.NumGoroutine()
	router, ctx, admission := testAdmission(t, promptOfTokens(t, 900))
	signals, release := admission.signals(context.Background())
	admission.run(router, context.Background())
	require.NoError(t, signals.Err(), "a fitting count cancelled the signals")
	release()
	require.ErrorIs(t, context.Cause(signals), context.Canceled)

	logs := observeRouterEvents(t)
	require.Nil(t, router.contextAdmissionRefusal(ctx))
	router.Config.ModelConfig = map[string]config.ModelParams{"model": {ContextWindowSize: 1_000, Tokenizer: tokencount.O200kBase}}
	ctx.contextTokenizers = map[string]modelTokenizer{"model": admission.models[0].tokenizer}
	tokens, kind, ok := router.contextTokens(ctx, nil, "model")
	require.True(t, ok)
	require.Equal(t, contextCountExact, kind)
	require.Equal(t, routerCountOf(promptOfTokens(t, 900)), tokens)
	require.Len(t, logs.FilterMessage("context_tokens_counted").All(), 1, "the prompt was counted again")
	counted := logs.FilterMessage("context_window_counted").All()
	require.Len(t, counted, 1)
	require.Equal(t, false, counted[0].ContextMap()["refused"])
	requireGoroutinesBack(t, baseline)
}

// A request that ends stops its count; the request is not refused.
func TestTheCountStopsWithTheRequest(t *testing.T) {
	router, ctx, admission := testAdmission(t, promptOfTokens(t, 1_100))
	request, cancel := context.WithCancel(context.Background())
	cancel()
	admission.run(router, request)
	require.ErrorIs(t, admission.err, context.Canceled)
	require.Nil(t, router.contextAdmissionRefusal(ctx))
}

// slowGuard is windowedGuard answering each window after delay, as a guard
// model under load does, and counting the windows it is sent.
type slowGuard struct {
	windowedGuard
	delay time.Duration
	mu    sync.Mutex
	sent  int
}

func (g *slowGuard) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		g.mu.Lock()
		g.sent++
		g.mu.Unlock()
		select {
		case <-time.After(g.delay):
		case <-r.Context().Done():
			return
		}
	}
	g.windowedGuard.ServeHTTP(w, r)
}

func (g *slowGuard) windows() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.sent
}

// slowGuardEngine routes with toolOutputGuardConfig's guard, a slow one, in
// front of two models with a 1,048,576-token window that name their
// tokenizer, or with no window when windows is false.
func slowGuardEngine(t *testing.T, delay time.Duration, windows bool) (routing.Engine, *slowGuard) {
	t.Helper()
	guard := &slowGuard{delay: delay}
	server := httptest.NewServer(guard)
	t.Cleanup(server.Close)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	t.Cleanup(backend.Close)
	t.Setenv("LONG_MESSAGE_TEST_KEY", "fake-key")
	yaml := strings.NewReplacer("BACKEND", strings.TrimPrefix(backend.URL, "http://"), "GATEWAY", server.URL).Replace(toolOutputGuardConfig)
	if windows {
		yaml = strings.ReplaceAll(yaml, "protocol: http, weight: 1}]\n",
			"protocol: http, weight: 1}]\n      context_window_size: 1048576\n      tokenizer: o200k_base\n")
	}
	cfg, err := config.ParseYAMLBytes([]byte(yaml))
	require.NoError(t, err)
	return routing.NewEngine(newLooperRouter(t, cfg), routing.DefaultOptions), guard
}

// planChat plans one Chat Completions request of text and how long it took.
func planChat(t *testing.T, engine routing.Engine, id, text string) (*routing.Plan, time.Duration) {
	t.Helper()
	body, err := json.Marshal(map[string]any{"model": "vllm-sr/auto", "messages": []map[string]any{{"role": "user", "content": text}}})
	require.NoError(t, err)
	header := routing.Header{
		{Name: ":method", Value: "POST"},
		{Name: ":path", Value: "/v1/chat/completions"},
		{Name: ":authority", Value: "router"},
		{Name: ":scheme", Value: "http"},
		{Name: "content-type", Value: "application/json"},
		{Name: "x-request-id", Value: id},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	started := time.Now()
	plan, err := engine.Plan(ctx, &routing.Request{Header: header, Body: body})
	elapsed := time.Since(started)
	require.NoError(t, err)
	t.Cleanup(func() { plan.Finish(nil) })
	return plan, elapsed
}

// An oversized request is refused as soon as its count has it, while its
// guard scan is still reading, and the cancelled scan sends no more windows.
func TestAnOversizedRequestStopsItsGuardScan(t *testing.T) {
	engine, guard := slowGuardEngine(t, 100*time.Millisecond, true)
	plan, elapsed := planChat(t, engine, "oversized", promptOfTokens(t, 1_100_000))
	require.NotNil(t, plan.Immediate, "the oversized request was routed")
	require.Equal(t, http.StatusBadRequest, plan.Immediate.Status)
	require.Contains(t, string(plan.Immediate.Body), "context_length_exceeded")
	refusedAfter := guard.windows()
	t.Logf("refused in %v after %d guard windows", elapsed, refusedAfter)
	time.Sleep(500 * time.Millisecond)
	require.Equal(t, refusedAfter, guard.windows(), "the cancelled guard scan kept sending windows")

	_, fitted := planChat(t, engine, "fitting", promptOfTokens(t, 1_040_000))
	require.Greater(t, fitted, elapsed, "the refusal waited for the guard")
	require.Greater(t, guard.windows(), 2*refusedAfter, "a fitting request's scan reads every window")
}

// A fitting request's count runs beside its signals: it adds nothing to the
// request once the signals take longer than the count.
func TestAFittingRequestDoesNotWaitForItsCount(t *testing.T) {
	engine, _ := slowGuardEngine(t, 100*time.Millisecond, true)
	logs := observeRouterEvents(t)
	plan, elapsed := planChat(t, engine, "fitting", promptOfTokens(t, 1_040_000))
	require.NotNil(t, plan.Call, "the fitting request was refused: %+v", plan.Immediate)
	counted := logs.FilterMessage("context_window_counted").All()
	require.Len(t, counted, 1)
	fields := counted[0].ContextMap()
	t.Logf("routed in %v; counted in %v ms beside the signals, waited %v ms", elapsed, fields["count_ms"], fields["wait_ms"])
	require.Less(t, fields["wait_ms"].(float64), 50.0, "the request waited for its count")
	require.Equal(t, false, fields["refused"])
}
