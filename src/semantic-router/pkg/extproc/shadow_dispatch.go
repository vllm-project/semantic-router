package extproc

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	mathrand "math/rand"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// Shadow dispatch sends a bounded, sampled copy of the approved request to a
// secondary configured model. It runs strictly after the primary dispatch
// response has been built, never blocks the request path beyond a
// non-blocking slot check, and records a deterministic outcome for replay.
// This file owns the request-path hook, the bounded lanes, and outcome
// recording; shadow_dispatch_call.go owns the HTTP call itself.

const (
	shadowDispatchOutcomeSource = "shadow_dispatch"
	shadowDispatchOutcomeTarget = "model"

	shadowVerdictCompleted = "completed"
	shadowVerdictFailed    = "failed"
	shadowVerdictDropped   = "dropped"

	shadowReasonCompleted            = "completed"
	shadowReasonSampledOut           = "sampled_out"
	shadowReasonSameAsPrimary        = "same_as_primary"
	shadowReasonInternalRequest      = "internal_request"
	shadowReasonRequestUnavailable   = "request_unavailable"
	shadowReasonQueueFull            = "queue_full"
	shadowReasonQueueTimeout         = "queue_timeout"
	shadowReasonRouterClosing        = "router_closing"
	shadowReasonBackendUnresolved    = "backend_unresolved"
	shadowReasonCredentialUnresolved = "credential_unresolved" //nolint:gosec // outcome reason code, not a secret
	shadowReasonEncodeFailed         = "encode_failed"
	shadowReasonTimeout              = "timeout"
	shadowReasonTransportError       = "transport_error"
	shadowReasonUpstreamStatus       = "upstream_status"
	shadowReasonResponseTooLarge     = "response_too_large"
	shadowReasonMalformedResponse    = "malformed_response"

	// shadowDispatchDrainTimeout is how long Close lets in-flight shadows
	// finish before cancelling them. Their primary requests already completed
	// and their upstream compute is already spent, so finishing is cheaper
	// than aborting; the per-job deadline bounds the wait regardless.
	shadowDispatchDrainTimeout   = 5 * time.Second
	shadowDispatchCancelGrace    = 2 * time.Second
	shadowDispatchErrorTextLimit = 200
)

// shadowDispatcher owns the bounded execution lanes for shadow calls. One
// lane exists per routing decision so a slow shadow backend on one route
// cannot starve another route's observations.
type shadowDispatcher struct {
	httpClient *http.Client
	sampler    func() float64
	now        func() time.Time

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu             sync.Mutex
	lanes          map[string]*shadowLane
	insecureClient *http.Client
	closed         bool
}

type shadowLane struct {
	slots    chan struct{}
	maxQueue int
	waiting  atomic.Int32
}

// shadowRequestEncoder renders the approved neutral request for one shadow
// target exactly as a primary dispatch to that model would.
type shadowRequestEncoder func(request llmprotocol.Request, target *shadowTarget) ([]byte, error)

// shadowJob is everything a shadow call needs, captured synchronously so the
// worker never reads mutable request state after the primary path moves on.
type shadowJob struct {
	cfg             config.ShadowDispatchPluginConfig
	routerConfig    *config.RouterConfig
	engine          *protocolcodec.Engine
	encode          shadowRequestEncoder
	request         llmprotocol.Request
	extraHeaders    map[string]string
	decision        string
	recipe          string
	primaryRequest  string
	primaryModel    string
	primaryBackend  string
	shadowRequestID string
	replayID        string
	recorder        *routerreplay.Recorder
	enqueuedAt      time.Time
	deadline        time.Time
}

type shadowResult struct {
	verdict       string
	reason        string
	shadowBackend string
	startedAt     time.Time
	finishedAt    time.Time
	attempts      int
	statusCode    int
	responseBytes int
	stopReason    string
	inputTokens   int64
	outputTokens  int64
	text          string
	err           string
}

func newShadowDispatcher() *shadowDispatcher {
	ctx, cancel := context.WithCancel(context.Background())
	return &shadowDispatcher{
		// Per-call deadlines come from the job context, so the client itself
		// carries no global timeout.
		httpClient: &http.Client{},
		sampler:    mathrand.Float64, //nolint:gosec // sampling only, not security sensitive
		now:        time.Now,
		ctx:        ctx,
		cancel:     cancel,
		lanes:      make(map[string]*shadowLane),
	}
}

// clientFor returns the HTTP client for a target. Certificate verification is
// skipped only when the operator opted in for that decision.
func (d *shadowDispatcher) clientFor(skipVerify bool) *http.Client {
	if !skipVerify {
		return d.httpClient
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.insecureClient == nil {
		transport := http.DefaultTransport.(*http.Transport).Clone()
		transport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true} //nolint:gosec // explicit operator opt-in matching Envoy's upstream posture
		d.insecureClient = &http.Client{Transport: transport}
	}
	return d.insecureClient
}

// Close stops accepting new shadow calls, drains in-flight ones for a bounded
// time, and only then cancels whatever is still running.
func (d *shadowDispatcher) Close() error {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		return nil
	}
	d.closed = true
	d.mu.Unlock()
	if d.waitIdle(shadowDispatchDrainTimeout) {
		d.cancel()
		return nil
	}
	logging.ComponentWarnEvent("extproc", "shadow_dispatch_drain_timeout", map[string]interface{}{
		"timeout": shadowDispatchDrainTimeout.String(),
	})
	d.cancel()
	d.waitIdle(shadowDispatchCancelGrace)
	return nil
}

// waitIdle blocks until every shadow worker has exited or the timeout passes.
func (d *shadowDispatcher) waitIdle(timeout time.Duration) bool {
	done := make(chan struct{})
	go func() {
		d.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		return true
	case <-time.After(timeout):
		return false
	}
}

// dispatchShadowIfConfigured is the single request-path hook. It runs after
// finalizeProviderDispatchResponse so the shadow starts from the same approved
// neutral request the primary was rendered from. Nothing here can fail the
// request.
func (r *OpenAIRouter) dispatchShadowIfConfigured(ctx *RequestContext, dispatch *providerDispatch) {
	if r == nil || r.ShadowDispatcher == nil || ctx == nil || dispatch == nil {
		return
	}
	pluginCfg := ctx.ShadowDispatchPluginConfig
	if pluginCfg == nil || !pluginCfg.Enabled {
		return
	}
	engine, err := r.protocolEngine()
	if err != nil {
		engine = nil
	}
	r.ShadowDispatcher.submit(ctx, dispatch, pluginCfg.WithDefaults(), shadowSubmitDeps{
		routerConfig: r.Config,
		engine:       engine,
		encode:       r.shadowRequestEncoder(ctx, dispatch, engine),
		extraHeaders: r.shadowExtraHeaders(ctx),
	})
}

type shadowSubmitDeps struct {
	routerConfig *config.RouterConfig
	engine       *protocolcodec.Engine
	encode       shadowRequestEncoder
	extraHeaders map[string]string
}

// shadowRequestEncoder mirrors the primary pipeline for the shadow model:
// semantic reasoning mode for non-chat targets before encoding, then the
// provider-dialect reasoning rewrite for chat targets after encoding.
func (r *OpenAIRouter) shadowRequestEncoder(
	ctx *RequestContext,
	dispatch *providerDispatch,
	engine *protocolcodec.Engine,
) shadowRequestEncoder {
	decision := ctx.VSRSelectedDecision
	decisionName := dispatch.decisionName
	useReasoning := dispatch.useReasoning
	envelope := ctx.ProtocolEnvelope
	return func(request llmprotocol.Request, target *shadowTarget) ([]byte, error) {
		request.Model = target.upstreamModel
		request.Stream = false
		request.StreamOptions = llmprotocol.StreamOptions{}
		// The neutral request now differs from the client bytes, so the codec
		// must render it instead of replaying the envelope's original body.
		request.Generation++
		if decisionName != "" && target.format != llmprotocol.OpenAIChatV1 {
			r.applySemanticReasoningMode(&request, target.logicalModel, target.format, useReasoning, decision)
		}
		encoded, err := engine.EncodeRequest(target.format, request, envelope)
		if err != nil {
			return nil, err
		}
		if decisionName == "" || target.format != llmprotocol.OpenAIChatV1 {
			return encoded.Body, nil
		}
		return r.setReasoningModeToRequestBodyForModelAndProvider(
			encoded.Body, target.logicalModel, useReasoning, decision, target.profile,
		)
	}
}

// shadowExtraHeaders carries the decision's header mutations and the trace
// context to the shadow backend, the same additions the primary receives.
// Client headers are never included.
func (r *OpenAIRouter) shadowExtraHeaders(ctx *RequestContext) map[string]string {
	extra := make(map[string]string)
	if ctx.TraceContext != nil {
		for _, pair := range tracing.InjectTraceContextToSlice(ctx.TraceContext) {
			extra[pair[0]] = pair[1]
		}
	}
	if ctx.VSRSelectedDecision != nil {
		setHeaders, _ := r.buildHeaderMutations(ctx.VSRSelectedDecision)
		for _, option := range setHeaders {
			header := option.GetHeader()
			if header == nil || strings.HasPrefix(header.GetKey(), ":") {
				continue
			}
			extra[header.GetKey()] = string(header.GetRawValue())
		}
	}
	return extra
}

func (d *shadowDispatcher) submit(
	ctx *RequestContext,
	dispatch *providerDispatch,
	cfg config.ShadowDispatchPluginConfig,
	deps shadowSubmitDeps,
) {
	decision := dispatch.decisionName
	if decision == "" {
		decision = ctx.VSRSelectedDecisionName
	}
	recipe := string(ctx.Routing.RecipeName())
	if recipe == "" {
		recipe = string(config.DefaultRecipeName)
	}
	dropEarly := func(reason string) {
		metrics.RecordShadowDispatch(decision, metrics.ShadowDispatchResultDropped, reason)
		logging.ComponentDebugEvent("extproc", "shadow_dispatch_skipped", map[string]interface{}{
			"request_id": ctx.RequestID,
			"decision":   decision,
			"reason":     reason,
		})
	}
	switch {
	case ctx.LooperRequest:
		dropEarly(shadowReasonInternalRequest)
		return
	case cfg.Model == "" || cfg.Model == dispatch.logicalModel:
		dropEarly(shadowReasonSameAsPrimary)
		return
	case ctx.SemanticRequest == nil || deps.routerConfig == nil || deps.engine == nil || deps.encode == nil:
		dropEarly(shadowReasonRequestUnavailable)
		return
	}
	rate := cfg.EffectiveSampleRate()
	if rate <= 0 || d.sampler() >= rate {
		metrics.RecordShadowDispatch(decision, metrics.ShadowDispatchResultSampledOut, shadowReasonSampledOut)
		return
	}

	now := d.now()
	job := &shadowJob{
		cfg:             cfg,
		routerConfig:    deps.routerConfig,
		engine:          deps.engine,
		encode:          deps.encode,
		request:         *ctx.SemanticRequest,
		extraHeaders:    deps.extraHeaders,
		decision:        decision,
		recipe:          recipe,
		primaryRequest:  ctx.RequestID,
		primaryModel:    dispatch.logicalModel,
		primaryBackend:  dispatch.backendName,
		shadowRequestID: shadowRequestID(ctx.RequestID),
		replayID:        ctx.RouterReplayID,
		recorder:        ctx.RouterReplayRecorder,
		enqueuedAt:      now,
		deadline:        now.Add(time.Duration(cfg.TimeoutSeconds) * time.Second),
	}
	d.enqueue(config.RoutingDecisionKey(config.RecipeName(recipe), decision), job)
}

func shadowRequestID(primary string) string {
	var buf [4]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return primary + "-shadow"
	}
	return primary + "-shadow-" + hex.EncodeToString(buf[:])
}

func (d *shadowDispatcher) laneFor(key string, cfg config.ShadowDispatchPluginConfig) *shadowLane {
	lane := d.lanes[key]
	if lane == nil || cap(lane.slots) != cfg.MaxConcurrency || lane.maxQueue != cfg.MaxQueueDepth {
		lane = &shadowLane{
			slots:    make(chan struct{}, cfg.MaxConcurrency),
			maxQueue: cfg.MaxQueueDepth,
		}
		d.lanes[key] = lane
	}
	return lane
}

// enqueue applies the concurrency and queue bounds. The only synchronous work
// is a non-blocking channel send and a counter check.
func (d *shadowDispatcher) enqueue(key string, job *shadowJob) {
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		d.recordDrop(job, shadowReasonRouterClosing)
		return
	}
	lane := d.laneFor(key, job.cfg)
	queued := false
	select {
	case lane.slots <- struct{}{}:
	default:
		if int(lane.waiting.Load()) >= lane.maxQueue {
			d.mu.Unlock()
			d.recordDrop(job, shadowReasonQueueFull)
			return
		}
		lane.waiting.Add(1)
		metrics.ShadowDispatchQueued.WithLabelValues(job.decision).Inc()
		queued = true
	}
	d.wg.Add(1)
	d.mu.Unlock()

	goSafely("shadow_dispatch", func() {
		defer d.wg.Done()
		if queued {
			acquired, reason := d.waitForSlot(lane, job)
			lane.waiting.Add(-1)
			metrics.ShadowDispatchQueued.WithLabelValues(job.decision).Dec()
			if !acquired {
				d.recordDrop(job, reason)
				return
			}
		}
		defer func() { <-lane.slots }()
		d.execute(job)
	})
}

func (d *shadowDispatcher) waitForSlot(lane *shadowLane, job *shadowJob) (bool, string) {
	timer := time.NewTimer(time.Until(job.deadline))
	defer timer.Stop()
	select {
	case lane.slots <- struct{}{}:
		return true, ""
	case <-timer.C:
		return false, shadowReasonQueueTimeout
	case <-d.ctx.Done():
		return false, shadowReasonRouterClosing
	}
}

// recordDrop reports a shadow call that never reached a worker. Drops are
// bounded-resource signals, so they are observable through metrics and logs
// rather than replay-store writes that could amplify an overload.
func (d *shadowDispatcher) recordDrop(job *shadowJob, reason string) {
	metrics.RecordShadowDispatch(job.decision, metrics.ShadowDispatchResultDropped, reason)
	logging.ComponentWarnEvent("extproc", "shadow_dispatch_dropped", map[string]interface{}{
		"request_id":        job.primaryRequest,
		"shadow_request_id": job.shadowRequestID,
		"decision":          job.decision,
		"recipe":            job.recipe,
		"shadow_model":      job.cfg.Model,
		"reason":            reason,
	})
}

func (d *shadowDispatcher) execute(job *shadowJob) {
	metrics.ShadowDispatchInflight.WithLabelValues(job.decision).Inc()
	defer metrics.ShadowDispatchInflight.WithLabelValues(job.decision).Dec()

	callCtx, cancel := context.WithDeadline(d.ctx, job.deadline)
	defer cancel()
	result := d.call(callCtx, job)
	d.record(job, result)
}

// record persists the observation. Metrics always fire; the replay outcome
// is appended when the primary request has a replay record, and a structured
// event covers routes without replay so the observation is never silent.
func (d *shadowDispatcher) record(job *shadowJob, result shadowResult) {
	metricResult := metrics.ShadowDispatchResultFailed
	if result.verdict == shadowVerdictCompleted {
		metricResult = metrics.ShadowDispatchResultCompleted
	}
	metrics.RecordShadowDispatch(job.decision, metricResult, result.reason)
	latency := result.finishedAt.Sub(result.startedAt)
	if latency > 0 {
		metrics.RecordShadowDispatchLatency(job.decision, latency.Seconds())
	}

	metadata := shadowOutcomeMetadata(job, result, latency)
	if job.recorder != nil && job.replayID != "" {
		outcome := routerreplay.Outcome{
			Timestamp: result.finishedAt.UTC(),
			Source:    shadowDispatchOutcomeSource,
			Target:    shadowDispatchOutcomeTarget,
			TargetRef: job.cfg.Model,
			Verdict:   result.verdict,
			Reason:    result.reason,
			Metadata:  metadata,
		}
		if err := job.recorder.AppendOutcome(job.replayID, outcome); err != nil {
			logging.ComponentErrorEvent("extproc", "shadow_dispatch_outcome_persist_failed", map[string]interface{}{
				"request_id": job.primaryRequest,
				"replay_id":  job.replayID,
				"error":      err.Error(),
			})
		}
	}

	fields := make(map[string]interface{}, len(metadata)+3)
	for key, value := range metadata {
		if key == "response_excerpt" || key == "response_excerpt_truncated" {
			continue
		}
		fields[key] = value
	}
	fields["request_id"] = job.primaryRequest
	fields["replay_id"] = job.replayID
	fields["verdict"] = result.verdict
	if result.verdict == shadowVerdictCompleted {
		logging.ComponentDebugEvent("extproc", "shadow_dispatch_outcome", fields)
	} else {
		logging.ComponentWarnEvent("extproc", "shadow_dispatch_outcome", fields)
	}
}

// shadowOutcomeMetadata is the bounded provenance stored with the outcome.
// It carries identities, timing, sizes, and a content hash. Response text is
// included only when the operator enabled capture, and then truncated.
func shadowOutcomeMetadata(job *shadowJob, result shadowResult, latency time.Duration) map[string]string {
	metadata := map[string]string{
		"primary_request_id": job.primaryRequest,
		"shadow_request_id":  job.shadowRequestID,
		"primary_model":      job.primaryModel,
		"primary_backend":    job.primaryBackend,
		"shadow_model":       job.cfg.Model,
		"shadow_backend":     result.shadowBackend,
		"decision":           job.decision,
		"recipe":             job.recipe,
		"sample_rate":        strconv.FormatFloat(job.cfg.EffectiveSampleRate(), 'f', -1, 64),
		"enqueued_at":        job.enqueuedAt.UTC().Format(time.RFC3339Nano),
		"started_at":         result.startedAt.UTC().Format(time.RFC3339Nano),
		"finished_at":        result.finishedAt.UTC().Format(time.RFC3339Nano),
		"queue_wait_ms":      strconv.FormatInt(result.startedAt.Sub(job.enqueuedAt).Milliseconds(), 10),
		"latency_ms":         strconv.FormatInt(latency.Milliseconds(), 10),
		"attempts":           strconv.Itoa(result.attempts),
	}
	if result.statusCode != 0 {
		metadata["status_code"] = strconv.Itoa(result.statusCode)
	}
	if result.err != "" {
		metadata["error"] = result.err
	}
	if result.verdict != shadowVerdictCompleted {
		return metadata
	}
	metadata["response_bytes"] = strconv.Itoa(result.responseBytes)
	metadata["stop_reason"] = result.stopReason
	metadata["input_tokens"] = strconv.FormatInt(result.inputTokens, 10)
	metadata["output_tokens"] = strconv.FormatInt(result.outputTokens, 10)
	metadata["response_chars"] = strconv.Itoa(utf8.RuneCountInString(result.text))
	sum := sha256.Sum256([]byte(result.text))
	metadata["response_sha256"] = hex.EncodeToString(sum[:])
	if job.cfg.CaptureResponseBody {
		if len(result.text) > job.cfg.MaxCaptureBytes {
			metadata["response_excerpt_truncated"] = "true"
		}
		metadata["response_excerpt"] = truncateShadowText(result.text, job.cfg.MaxCaptureBytes)
	}
	return metadata
}

// truncateShadowText cuts on a rune boundary so stored text stays valid UTF-8.
func truncateShadowText(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	cut := value[:limit]
	for len(cut) > 0 && !utf8.ValidString(cut) {
		cut = cut[:len(cut)-1]
	}
	return cut
}
