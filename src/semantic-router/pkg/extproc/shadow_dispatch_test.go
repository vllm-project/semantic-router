package extproc

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const (
	shadowTestModel    = "shadow-model"
	shadowTestDecision = "shadow-route"
	shadowTestReply    = "shadow says hi"
)

type shadowTestBackend struct {
	server  *httptest.Server
	mu      sync.Mutex
	bodies  [][]byte
	headers []http.Header
	handler func(w http.ResponseWriter, body []byte)
}

func newShadowTestBackend(t *testing.T) *shadowTestBackend {
	t.Helper()
	return newShadowTestBackendWithTLS(t, false)
}

func newShadowTestBackendWithTLS(t *testing.T, useTLS bool) *shadowTestBackend {
	t.Helper()
	backend := &shadowTestBackend{}
	backend.handler = func(w http.ResponseWriter, _ []byte) {
		writeShadowChatCompletion(w, shadowTestReply)
	}
	serve := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		backend.mu.Lock()
		backend.bodies = append(backend.bodies, body)
		backend.headers = append(backend.headers, r.Header.Clone())
		handler := backend.handler
		backend.mu.Unlock()
		handler(w, body)
	})
	if useTLS {
		backend.server = httptest.NewTLSServer(serve)
	} else {
		backend.server = httptest.NewServer(serve)
	}
	t.Cleanup(backend.server.Close)
	return backend
}

func (b *shadowTestBackend) requestCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.bodies)
}

func (b *shadowTestBackend) setHandler(handler func(w http.ResponseWriter, body []byte)) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.handler = handler
}

func writeShadowChatCompletion(w http.ResponseWriter, content string) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"id":      "cmpl-shadow",
		"object":  "chat.completion",
		"created": 1,
		"model":   shadowTestModel,
		"choices": []map[string]interface{}{{
			"index":         0,
			"message":       map[string]string{"role": "assistant", "content": content},
			"finish_reason": "stop",
		}},
		"usage": map[string]int{"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
	})
}

func newShadowTestRouter(t *testing.T, backend *shadowTestBackend) (*OpenAIRouter, string) {
	t.Helper()
	router, primaryModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	scheme, hostPort, _ := strings.Cut(backend.server.URL, "://")
	host, portText, err := net.SplitHostPort(hostPort)
	if err != nil {
		t.Fatalf("split backend host: %v", err)
	}
	port, _ := strconv.Atoi(portText)
	router.Config.ModelConfig[shadowTestModel] = config.ModelParams{
		PreferredEndpoints: []string{"shadow-backend"},
		APIFormat:          config.APIFormatOpenAI,
	}
	router.Config.VLLMEndpoints = append(router.Config.VLLMEndpoints, config.VLLMEndpoint{
		Name: "shadow-backend", Address: host, Port: port, Protocol: scheme,
	})
	dispatcher := newShadowDispatcher()
	dispatcher.sampler = func() float64 { return 0 }
	router.ShadowDispatcher = dispatcher
	t.Cleanup(func() { _ = dispatcher.Close() })
	return router, primaryModel
}

func shadowTestPluginConfig() *config.ShadowDispatchPluginConfig {
	return &config.ShadowDispatchPluginConfig{
		Enabled:        true,
		Model:          shadowTestModel,
		TimeoutSeconds: 5,
	}
}

type shadowRun struct {
	body     []byte
	replayID string
	recorder *routerreplay.Recorder
	ctx      *RequestContext
}

func runShadowRequest(
	t *testing.T,
	router *OpenAIRouter,
	primaryModel string,
	pluginCfg *config.ShadowDispatchPluginConfig,
	mutate func(ctx *RequestContext),
) shadowRun {
	t.Helper()
	return runShadowRequestWithReasoning(t, router, primaryModel, pluginCfg, mutate, false)
}

func runShadowRequestWithReasoning(
	t *testing.T,
	router *OpenAIRouter,
	primaryModel string,
	pluginCfg *config.ShadowDispatchPluginConfig,
	mutate func(ctx *RequestContext),
	useReasoning bool,
) shadowRun {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	request := testNeutralRequest("virtual", "please shadow this prompt")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	decision := &config.Decision{
		Name:      shadowTestDecision,
		ModelRefs: []config.ModelRef{{Model: primaryModel}},
	}
	ctx.VSRSelectedDecision = decision
	ctx.VSRSelectedDecisionName = decision.Name
	replayID, err := recorder.AddRecord(routerreplay.RoutingRecord{
		RequestID: ctx.RequestID,
		Decision:  decision.Name,
	})
	if err != nil {
		t.Fatalf("add replay record: %v", err)
	}
	ctx.RouterReplayID = replayID
	ctx.RouterReplayRecorder = recorder
	ctx.ShadowDispatchPluginConfig = pluginCfg
	if mutate != nil {
		mutate(ctx)
	}
	response, err := router.handleEntrypointModelRouting(
		request, "virtual", decision.Name, entropy.ReasoningDecision{UseReasoning: useReasoning}, primaryModel, ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting: %v", err)
	}
	return shadowRun{
		body:     response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(),
		replayID: ctx.RouterReplayID,
		recorder: recorder,
		ctx:      ctx,
	}
}

func waitForShadow(t *testing.T, router *OpenAIRouter) {
	t.Helper()
	if !router.ShadowDispatcher.waitIdle(10 * time.Second) {
		t.Fatal("shadow dispatcher did not become idle")
	}
}

func shadowOutcomes(t *testing.T, run shadowRun) []routerreplay.Outcome {
	t.Helper()
	record, ok := run.recorder.GetRecord(run.replayID)
	if !ok {
		t.Fatalf("replay record %q missing", run.replayID)
	}
	var outcomes []routerreplay.Outcome
	for _, outcome := range record.Outcomes {
		if outcome.Source == shadowDispatchOutcomeSource {
			outcomes = append(outcomes, outcome)
		}
	}
	return outcomes
}

func singleShadowOutcome(t *testing.T, run shadowRun) routerreplay.Outcome {
	t.Helper()
	outcomes := shadowOutcomes(t, run)
	if len(outcomes) != 1 {
		t.Fatalf("expected exactly one shadow outcome, got %d", len(outcomes))
	}
	return outcomes[0]
}

func shadowCounter(decision, result, reason string) float64 {
	return testutil.ToFloat64(metrics.ShadowDispatchTotal.WithLabelValues(decision, result, reason))
}

func TestShadowDispatchCompletesWithoutChangingPrimaryResponse(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)

	baseline := runShadowRequest(t, router, primaryModel, nil, nil)
	shadowed := runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), nil)
	if !bytes.Equal(baseline.body, shadowed.body) {
		t.Fatalf("primary body changed with shadow enabled:\n%s\n%s", baseline.body, shadowed.body)
	}
	waitForShadow(t, router)

	assertShadowWireRequest(t, backend, shadowed.ctx.RequestID)
	assertShadowCompletedProvenance(t, singleShadowOutcome(t, shadowed), shadowed.ctx.RequestID, primaryModel)
}

// assertShadowWireRequest checks what the shadow backend actually received:
// the shadow model, non-streaming, a derived request id, and no credential.
func assertShadowWireRequest(t *testing.T, backend *shadowTestBackend, primaryRequestID string) {
	t.Helper()
	if got := backend.requestCount(); got != 1 {
		t.Fatalf("shadow backend requests = %d, want 1", got)
	}
	var wire struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(backend.bodies[0], &wire); err != nil {
		t.Fatalf("decode shadow request: %v", err)
	}
	if wire.Model != shadowTestModel || wire.Stream {
		t.Fatalf("shadow request model=%q stream=%v, want %q non-streaming", wire.Model, wire.Stream, shadowTestModel)
	}
	if got := backend.headers[0].Get(headers.RequestID); !strings.HasPrefix(got, primaryRequestID+"-shadow") {
		t.Fatalf("shadow request id = %q, want derived from %q", got, primaryRequestID)
	}
	if got := backend.headers[0].Get("Authorization"); got != "" {
		t.Fatalf("shadow request forwarded a credential: %q", got)
	}
}

// assertShadowCompletedProvenance checks the audit fields on a completed
// outcome without any response text leaking into the record.
func assertShadowCompletedProvenance(t *testing.T, outcome routerreplay.Outcome, primaryRequestID, primaryModel string) {
	t.Helper()
	if outcome.Verdict != shadowVerdictCompleted || outcome.Reason != shadowReasonCompleted {
		t.Fatalf("outcome verdict=%q reason=%q", outcome.Verdict, outcome.Reason)
	}
	if outcome.Target != shadowDispatchOutcomeTarget || outcome.TargetRef != shadowTestModel {
		t.Fatalf("outcome target=%q ref=%q", outcome.Target, outcome.TargetRef)
	}
	sum := sha256.Sum256([]byte(shadowTestReply))
	expectations := map[string]string{
		"primary_request_id": primaryRequestID,
		"primary_model":      primaryModel,
		"shadow_model":       shadowTestModel,
		"shadow_backend":     "shadow-backend",
		"decision":           shadowTestDecision,
		"status_code":        "200",
		"attempts":           "1",
		"stop_reason":        string(llmprotocol.StopEndTurn),
		"output_tokens":      "4",
		"input_tokens":       "3",
		"response_sha256":    hex.EncodeToString(sum[:]),
	}
	for key, want := range expectations {
		if got := outcome.Metadata[key]; got != want {
			t.Fatalf("metadata[%s] = %q, want %q", key, got, want)
		}
	}
	if _, ok := outcome.Metadata["response_excerpt"]; ok {
		t.Fatal("response excerpt stored without capture_response_body")
	}
	if outcome.Metadata["shadow_request_id"] == "" || outcome.Metadata["latency_ms"] == "" {
		t.Fatalf("outcome metadata missing identity or timing: %v", outcome.Metadata)
	}
}

func TestShadowDispatchCapturesBoundedExcerpt(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	cfg := shadowTestPluginConfig()
	cfg.CaptureResponseBody = true
	cfg.MaxCaptureBytes = 6

	run := runShadowRequest(t, router, primaryModel, cfg, nil)
	waitForShadow(t, router)

	outcome := singleShadowOutcome(t, run)
	if got := outcome.Metadata["response_excerpt"]; got != shadowTestReply[:6] {
		t.Fatalf("excerpt = %q, want %q", got, shadowTestReply[:6])
	}
	if outcome.Metadata["response_excerpt_truncated"] != "true" {
		t.Fatal("expected excerpt truncation marker")
	}
}

func TestShadowDispatchUpstreamErrorIsIsolatedAndRetried(t *testing.T) {
	backend := newShadowTestBackend(t)
	backend.setHandler(func(w http.ResponseWriter, _ []byte) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	router, primaryModel := newShadowTestRouter(t, backend)
	cfg := shadowTestPluginConfig()
	cfg.MaxRetries = 1

	baseline := runShadowRequest(t, router, primaryModel, nil, nil)
	run := runShadowRequest(t, router, primaryModel, cfg, nil)
	if !bytes.Equal(baseline.body, run.body) {
		t.Fatal("primary body changed when shadow backend failed")
	}
	waitForShadow(t, router)

	if got := backend.requestCount(); got != 2 {
		t.Fatalf("shadow attempts = %d, want 2", got)
	}
	outcome := singleShadowOutcome(t, run)
	if outcome.Verdict != shadowVerdictFailed || outcome.Reason != shadowReasonUpstreamStatus {
		t.Fatalf("outcome verdict=%q reason=%q", outcome.Verdict, outcome.Reason)
	}
	if outcome.Metadata["status_code"] != "500" || outcome.Metadata["attempts"] != "2" {
		t.Fatalf("metadata = %v", outcome.Metadata)
	}
	if _, ok := outcome.Metadata["response_sha256"]; ok {
		t.Fatal("failed outcome must not carry response provenance")
	}
}

func TestShadowDispatchTimeoutIsBounded(t *testing.T) {
	backend := newShadowTestBackend(t)
	release := make(chan struct{})
	backend.setHandler(func(w http.ResponseWriter, _ []byte) {
		<-release
		writeShadowChatCompletion(w, shadowTestReply)
	})
	defer close(release)
	router, primaryModel := newShadowTestRouter(t, backend)
	cfg := shadowTestPluginConfig()
	cfg.TimeoutSeconds = 1

	started := time.Now()
	run := runShadowRequest(t, router, primaryModel, cfg, nil)
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("primary path waited on shadow: %v", elapsed)
	}
	waitForShadow(t, router)

	outcome := singleShadowOutcome(t, run)
	if outcome.Verdict != shadowVerdictFailed || outcome.Reason != shadowReasonTimeout {
		t.Fatalf("outcome verdict=%q reason=%q", outcome.Verdict, outcome.Reason)
	}
}

func TestShadowDispatchRejectsOversizedResponse(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	cfg := shadowTestPluginConfig()
	cfg.MaxResponseBytes = 16

	run := runShadowRequest(t, router, primaryModel, cfg, nil)
	waitForShadow(t, router)

	outcome := singleShadowOutcome(t, run)
	if outcome.Verdict != shadowVerdictFailed || outcome.Reason != shadowReasonResponseTooLarge {
		t.Fatalf("outcome verdict=%q reason=%q", outcome.Verdict, outcome.Reason)
	}
}

func TestShadowDispatchQueueFullDropsWithoutBlocking(t *testing.T) {
	backend := newShadowTestBackend(t)
	release := make(chan struct{})
	backend.setHandler(func(w http.ResponseWriter, _ []byte) {
		<-release
		writeShadowChatCompletion(w, shadowTestReply)
	})
	router, primaryModel := newShadowTestRouter(t, backend)
	cfg := shadowTestPluginConfig()
	cfg.MaxConcurrency = 1
	cfg.MaxQueueDepth = 1

	before := shadowCounter(shadowTestDecision, metrics.ShadowDispatchResultDropped, shadowReasonQueueFull)
	first := runShadowRequest(t, router, primaryModel, cfg, nil)
	deadline := time.Now().Add(5 * time.Second)
	for backend.requestCount() == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	queued := runShadowRequest(t, router, primaryModel, cfg, nil)
	dropped := runShadowRequest(t, router, primaryModel, cfg, nil)
	close(release)
	waitForShadow(t, router)

	if got := backend.requestCount(); got != 2 {
		t.Fatalf("shadow backend requests = %d, want 2 (third must be dropped)", got)
	}
	if got := shadowCounter(shadowTestDecision, metrics.ShadowDispatchResultDropped, shadowReasonQueueFull) - before; got != 1 {
		t.Fatalf("queue_full drops = %v, want 1", got)
	}
	for name, run := range map[string]shadowRun{"first": first, "queued": queued} {
		if len(shadowOutcomes(t, run)) != 1 {
			t.Fatalf("%s request should record one completed shadow outcome", name)
		}
	}
	if len(shadowOutcomes(t, dropped)) != 0 {
		t.Fatal("dropped shadow must not write a replay outcome")
	}
}

func TestShadowDispatchSamplingAndSkipRules(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)

	zero := 0.0
	sampledOut := shadowTestPluginConfig()
	sampledOut.SampleRate = &zero
	runShadowRequest(t, router, primaryModel, sampledOut, nil)

	samePrimary := shadowTestPluginConfig()
	samePrimary.Model = primaryModel
	runShadowRequest(t, router, primaryModel, samePrimary, nil)

	runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), func(ctx *RequestContext) {
		ctx.LooperRequest = true
	})

	disabled := shadowTestPluginConfig()
	disabled.Enabled = false
	runShadowRequest(t, router, primaryModel, disabled, nil)

	waitForShadow(t, router)
	if got := backend.requestCount(); got != 0 {
		t.Fatalf("shadow backend requests = %d, want 0", got)
	}
}

func TestShadowDispatchUnknownModelRecordsBackendUnresolved(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	cfg := shadowTestPluginConfig()
	cfg.Model = "ghost-model"

	run := runShadowRequest(t, router, primaryModel, cfg, nil)
	waitForShadow(t, router)

	outcome := singleShadowOutcome(t, run)
	if outcome.Verdict != shadowVerdictFailed || outcome.Reason != shadowReasonBackendUnresolved {
		t.Fatalf("outcome verdict=%q reason=%q", outcome.Verdict, outcome.Reason)
	}
	if backend.requestCount() != 0 {
		t.Fatal("unresolved shadow model must not send a request")
	}
}

func TestShadowDispatchWithoutReplayStillObserves(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)

	before := shadowCounter(shadowTestDecision, metrics.ShadowDispatchResultCompleted, shadowReasonCompleted)
	runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), func(ctx *RequestContext) {
		ctx.RouterReplayID = ""
		ctx.RouterReplayRecorder = nil
	})
	waitForShadow(t, router)

	if backend.requestCount() != 1 {
		t.Fatal("shadow should still run without a replay record")
	}
	if got := shadowCounter(shadowTestDecision, metrics.ShadowDispatchResultCompleted, shadowReasonCompleted) - before; got != 1 {
		t.Fatalf("completed counter delta = %v, want 1", got)
	}
}

func TestShadowDispatcherCloseStopsNewWork(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	if err := router.ShadowDispatcher.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	before := shadowCounter(shadowTestDecision, metrics.ShadowDispatchResultDropped, shadowReasonRouterClosing)
	runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), nil)
	if backend.requestCount() != 0 {
		t.Fatal("closed dispatcher must not send shadow requests")
	}
	if got := shadowCounter(shadowTestDecision, metrics.ShadowDispatchResultDropped, shadowReasonRouterClosing) - before; got != 1 {
		t.Fatalf("router_closing drops = %v, want 1", got)
	}
}

// The same-model routing branch leaves the neutral request untouched, so the
// chat codec replays the client's original bytes for the primary. The shadow
// must still be rendered for the shadow model rather than replayed. Client
// extension fields that only survive a verbatim replay are not carried, which
// matches every primary dispatch the router itself mutates.
func TestShadowDispatchRendersShadowModelWhenPrimaryReplaysClientBytes(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	primaryParams := router.Config.ModelConfig[primaryModel]
	primaryParams.ExternalModelIDs = nil
	router.Config.ModelConfig[primaryModel] = primaryParams

	clientBytes := []byte(`{"model":"` + primaryModel + `","messages":[{"role":"user","content":"replay me"}],"stream":false,"x_client_marker":true}`)
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	request := testNeutralRequest(primaryModel, "replay me")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ProtocolEnvelope = llmprotocol.Envelope{
		Format:     llmprotocol.OpenAIChatV1,
		Generation: request.Generation,
		Request:    clientBytes,
	}
	decision := &config.Decision{Name: shadowTestDecision, ModelRefs: []config.ModelRef{{Model: primaryModel}}}
	ctx.VSRSelectedDecision = decision
	ctx.VSRSelectedDecisionName = decision.Name
	replayID, err := recorder.AddRecord(routerreplay.RoutingRecord{RequestID: ctx.RequestID, Decision: decision.Name})
	if err != nil {
		t.Fatalf("add replay record: %v", err)
	}
	ctx.RouterReplayID = replayID
	ctx.RouterReplayRecorder = recorder
	ctx.ShadowDispatchPluginConfig = shadowTestPluginConfig()

	response, err := router.handleEntrypointModelRouting(
		request, primaryModel, decision.Name, entropy.ReasoningDecision{}, primaryModel, ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting: %v", err)
	}
	primaryBody := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var primaryWire struct {
		Model  string `json:"model"`
		Marker *bool  `json:"x_client_marker"`
	}
	if err := json.Unmarshal(primaryBody, &primaryWire); err != nil {
		t.Fatalf("decode primary request: %v", err)
	}
	if primaryWire.Model != primaryModel || primaryWire.Marker == nil {
		t.Fatalf("primary body should keep the client's model and unknown fields: %s", primaryBody)
	}
	waitForShadow(t, router)

	if got := backend.requestCount(); got != 1 {
		t.Fatalf("shadow backend requests = %d, want 1", got)
	}
	var shadowWire struct {
		Model  string `json:"model"`
		Marker *bool  `json:"x_client_marker"`
	}
	if err := json.Unmarshal(backend.bodies[0], &shadowWire); err != nil {
		t.Fatalf("decode shadow request: %v", err)
	}
	if shadowWire.Model != shadowTestModel {
		t.Fatalf("shadow request model = %q, want %q: %s", shadowWire.Model, shadowTestModel, backend.bodies[0])
	}
	if shadowWire.Marker != nil {
		t.Fatalf("shadow request replayed client bytes instead of rendering the neutral request: %s", backend.bodies[0])
	}
}

func TestShadowDispatchAppliesShadowReasoningAndDecisionHeaders(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	shadowParams := router.Config.ModelConfig[shadowTestModel]
	shadowParams.ReasoningFamily = "openai"
	router.Config.ModelConfig[shadowTestModel] = shadowParams
	router.Config.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{
		"openai": {Type: config.ReasoningFamilyTypeTopLevelReasoningEffort, Parameter: "reasoning_effort"},
	}

	runShadowRequestWithReasoning(t, router, primaryModel, shadowTestPluginConfig(), func(ctx *RequestContext) {
		ctx.VSRSelectedDecision.Plugins = []config.DecisionPlugin{{
			Type: config.DecisionPluginHeaderMutation,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"add": []map[string]string{{"name": "x-tenant", "value": "acme"}},
			}),
		}}
	}, true)
	waitForShadow(t, router)

	if got := backend.requestCount(); got != 1 {
		t.Fatalf("shadow backend requests = %d, want 1", got)
	}
	var wire struct {
		ReasoningEffort string `json:"reasoning_effort"`
	}
	if err := json.Unmarshal(backend.bodies[0], &wire); err != nil {
		t.Fatalf("decode shadow request: %v", err)
	}
	if wire.ReasoningEffort == "" {
		t.Fatalf("shadow body lacks the shadow model's reasoning adaptation: %s", backend.bodies[0])
	}
	if got := backend.headers[0].Get("x-tenant"); got != "acme" {
		t.Fatalf("x-tenant = %q, want decision header mutation applied", got)
	}
}

func TestShadowDispatchUsesOnlyStaticCredentials(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	primaryParams := router.Config.ModelConfig[primaryModel]
	primaryParams.AccessKey = "primary-key"
	router.Config.ModelConfig[primaryModel] = primaryParams
	// A fail-closed chain, as configured when authz.providers is set.
	resolver := authz.NewCredentialResolver(
		authz.NewHeaderInjectionProvider(authz.DefaultHeaderMap()),
		authz.NewStaticConfigProvider(router.Config),
	)
	resolver.SetFailOpen(false)
	router.CredentialResolver = resolver

	keyless := runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), nil)
	waitForShadow(t, router)
	if outcome := singleShadowOutcome(t, keyless); outcome.Verdict != shadowVerdictCompleted {
		t.Fatalf("keyless shadow backend verdict=%q reason=%q, want completed", outcome.Verdict, outcome.Reason)
	}
	if got := backend.headers[0].Get("Authorization"); got != "" {
		t.Fatalf("keyless shadow sent a credential: %q", got)
	}

	shadowParams := router.Config.ModelConfig[shadowTestModel]
	shadowParams.AccessKey = "shadow-key"
	router.Config.ModelConfig[shadowTestModel] = shadowParams
	runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), nil)
	waitForShadow(t, router)
	if got := backend.headers[1].Get("Authorization"); got != "Bearer shadow-key" {
		t.Fatalf("Authorization = %q, want the shadow model's static key", got)
	}
}

func TestShadowDispatchTLSVerificationIsOptIn(t *testing.T) {
	backend := newShadowTestBackendWithTLS(t, true)
	router, primaryModel := newShadowTestRouter(t, backend)

	verified := runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), nil)
	waitForShadow(t, router)
	outcome := singleShadowOutcome(t, verified)
	if outcome.Verdict != shadowVerdictFailed || outcome.Reason != shadowReasonTransportError {
		t.Fatalf("self-signed backend verdict=%q reason=%q, want transport_error", outcome.Verdict, outcome.Reason)
	}
	if backend.requestCount() != 0 {
		t.Fatal("verification failure must not reach the backend")
	}

	skip := shadowTestPluginConfig()
	skip.TLSSkipVerify = true
	skipped := runShadowRequest(t, router, primaryModel, skip, nil)
	waitForShadow(t, router)
	if outcome := singleShadowOutcome(t, skipped); outcome.Verdict != shadowVerdictCompleted {
		t.Fatalf("tls_skip_verify verdict=%q reason=%q, want completed", outcome.Verdict, outcome.Reason)
	}
	if state := backend.server.TLS; state == nil || len(state.Certificates) == 0 {
		t.Fatal("test backend did not serve TLS")
	}
	_ = tls.VersionTLS12
}

func TestShadowDispatchRetryDoesNotInheritEarlierStatus(t *testing.T) {
	backend := newShadowTestBackend(t)
	var attempts int
	backend.setHandler(func(w http.ResponseWriter, _ []byte) {
		backend.mu.Lock()
		attempts++
		current := attempts
		backend.mu.Unlock()
		if current == 1 {
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		}
		hijacker, ok := w.(http.Hijacker)
		if !ok {
			http.Error(w, "no hijack", http.StatusInternalServerError)
			return
		}
		conn, _, err := hijacker.Hijack()
		if err == nil {
			_ = conn.Close()
		}
	})
	router, primaryModel := newShadowTestRouter(t, backend)
	cfg := shadowTestPluginConfig()
	cfg.MaxRetries = 1

	run := runShadowRequest(t, router, primaryModel, cfg, nil)
	waitForShadow(t, router)

	outcome := singleShadowOutcome(t, run)
	if outcome.Verdict != shadowVerdictFailed || outcome.Reason != shadowReasonTransportError {
		t.Fatalf("outcome verdict=%q reason=%q", outcome.Verdict, outcome.Reason)
	}
	if status, ok := outcome.Metadata["status_code"]; ok {
		t.Fatalf("transport failure carried stale status_code=%s", status)
	}
	if outcome.Metadata["attempts"] != "2" {
		t.Fatalf("attempts = %q, want 2", outcome.Metadata["attempts"])
	}
}

func TestShadowDispatcherCloseDrainsInflightCalls(t *testing.T) {
	backend := newShadowTestBackend(t)
	release := make(chan struct{})
	backend.setHandler(func(w http.ResponseWriter, _ []byte) {
		<-release
		writeShadowChatCompletion(w, shadowTestReply)
	})
	router, primaryModel := newShadowTestRouter(t, backend)

	run := runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), nil)
	deadline := time.Now().Add(5 * time.Second)
	for backend.requestCount() == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	closed := make(chan struct{})
	go func() {
		_ = router.ShadowDispatcher.Close()
		close(closed)
	}()
	time.Sleep(200 * time.Millisecond)
	close(release)
	select {
	case <-closed:
	case <-time.After(10 * time.Second):
		t.Fatal("Close did not return")
	}
	if outcome := singleShadowOutcome(t, run); outcome.Verdict != shadowVerdictCompleted {
		t.Fatalf("in-flight shadow at close verdict=%q reason=%q, want completed", outcome.Verdict, outcome.Reason)
	}
	if err := router.ShadowDispatcher.ctx.Err(); err != context.Canceled {
		t.Fatalf("dispatcher context after close = %v, want canceled", err)
	}
}
