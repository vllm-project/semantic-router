package extproc

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	mathrand "math/rand"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	callbudget "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
)

// Shadow dispatch, router_replay's shadow, sends a bounded, sampled copy of the
// approved request to a secondary configured model and appends its outcome to
// the primary's replay record. It runs strictly after the primary dispatch
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
	shadowReasonReplayQueueFull      = "replay_queue_full"
	shadowReasonQueueTimeout         = "queue_timeout"
	shadowReasonRouterClosing        = "router_closing"
	shadowReasonBackendUnresolved    = "backend_unresolved"
	shadowReasonCredentialUnresolved = "credential_unresolved" //nolint:gosec // outcome reason code, not a secret
	shadowReasonEncodeFailed         = "encode_failed"
	shadowReasonTimeout              = "timeout"
	shadowReasonTransportError       = "transport_error"
	shadowReasonUpstreamStatus       = "upstream_status"
	shadowReasonRedirectRejected     = "redirect_rejected"
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
	sampler func() float64
	now     func() time.Time

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu      sync.Mutex
	lanes   map[string]*shadowLane
	clients map[string]*connector.Client
	closed  bool
}

type shadowLane struct {
	slots    chan struct{}
	groups   chan struct{}
	maxQueue int
	waiting  atomic.Int32
}

// shadowRequestEncoder renders the approved neutral request for one shadow
// target exactly as a primary dispatch to that model would.
type shadowRequestEncoder func(request llmprotocol.Request, target *shadowTarget) ([]byte, error)

// shadowJob is everything a shadow call needs, captured synchronously so the
// worker never reads mutable request state after the primary path moves on.
type shadowJob struct {
	cfg             config.RouterReplayShadowConfig
	model           string
	context         context.Context
	responseLimit   int
	captureLimit    int
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
	receipt         *routerreplay.OutcomeReservation
	enqueuedAt      time.Time
	timeout         time.Duration
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
	inputKnown    bool
	outputKnown   bool
	text          string
	err           string
}

func newShadowDispatcher() *shadowDispatcher {
	ctx, cancel := context.WithCancel(context.Background())
	return &shadowDispatcher{
		sampler: mathrand.Float64, //nolint:gosec // sampling only, not security sensitive
		now:     time.Now,
		ctx:     ctx,
		cancel:  cancel,
		lanes:   make(map[string]*shadowLane),
		clients: make(map[string]*connector.Client),
	}
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
	if !d.waitIdle(shadowDispatchDrainTimeout) {
		logging.ComponentWarnEvent("extproc", "shadow_dispatch_drain_timeout", map[string]interface{}{
			"timeout": shadowDispatchDrainTimeout.String(),
		})
		d.cancel()
		d.waitIdle(shadowDispatchCancelGrace)
	} else {
		d.cancel()
	}
	d.closeConnectors()
	return nil
}

// closeConnectors releases idle connections held by every shadow connector.
func (d *shadowDispatcher) closeConnectors() {
	d.mu.Lock()
	defer d.mu.Unlock()
	for key, client := range d.clients {
		_ = client.Close()
		delete(d.clients, key)
	}
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
// neutral request the primary was rendered from. The shadow belongs to the
// decision's router_replay policy, so it runs only while the decision records
// replays. Nothing here can fail the request.
func (r *OpenAIRouter) dispatchShadowIfConfigured(ctx *RequestContext, dispatch *providerDispatch) {
	if r == nil || r.ShadowDispatcher == nil || ctx == nil || dispatch == nil {
		return
	}
	replay := ctx.RouterReplayPluginConfig
	if replay == nil || !replay.Enabled || replay.Shadow == nil {
		return
	}
	engine, err := r.protocolEngine()
	if err != nil {
		engine = nil
	}
	cfg := replay.Shadow.WithDefaults()
	cfg.Capture.Enabled = cfg.Capture.Enabled && replayResponseContentAllowed(ctx)
	r.ShadowDispatcher.submit(ctx, dispatch, cfg, shadowSubmitDeps{
		routerConfig: r.Config,
		engine:       engine,
		encode:       r.shadowRequestEncoder(ctx, dispatch, engine),
		extraHeaders: r.shadowExtraHeaders(ctx, dispatch, cfg.ForwardHeaders),
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
// provider-dialect rewrites for chat targets after encoding.
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
		projected, err := r.projectRequestForBackend(request, target.logicalModel, target.format)
		if err != nil {
			return nil, err
		}
		encoded, err := engine.EncodeRequest(target.format, projected, envelope)
		if err != nil {
			return nil, err
		}
		body := encoded.Body
		if decisionName != "" && target.format == llmprotocol.OpenAIChatV1 {
			body, err = r.setReasoningModeToRequestBodyForModelAndProvider(
				body, target.logicalModel, useReasoning, decision, target.profile,
			)
			if err != nil {
				return nil, err
			}
		}
		return adaptOllamaOutputLimit(body, target.format, target.profile)
	}
}

// shadowHeaderIsSensitive reports whether a header name may carry a
// credential: a known credential carrier, or the resolved auth header of one
// of the provider profiles involved, so a custom auth_header is covered too.
func shadowHeaderIsSensitive(name string, authHeaders ...string) bool {
	if config.IsShadowCredentialHeader(name) {
		return true
	}
	trimmed := strings.TrimSpace(name)
	for _, authHeader := range authHeaders {
		if authHeader != "" && strings.EqualFold(trimmed, authHeader) {
			return true
		}
	}
	return false
}

// shadowProfileAuthHeader returns the auth header name a provider profile
// resolves to, or "" when it cannot be resolved. A nil profile means the
// OpenAI default, which the credential set already covers.
func shadowProfileAuthHeader(profile *config.ProviderProfile) string {
	if profile == nil {
		return ""
	}
	_, providerAuth, err := resolveProviderAuth(profile)
	if err != nil {
		return ""
	}
	return providerAuth.Header
}

// shadowExtraHeaders carries W3C span context and allowed decision mutations.
func (r *OpenAIRouter) shadowExtraHeaders(ctx *RequestContext, dispatch *providerDispatch, allowed []string) map[string]string {
	extra := make(map[string]string)
	if ctx.TraceContext != nil {
		for _, pair := range tracing.InjectSpanContextToSlice(ctx.TraceContext) {
			extra[pair[0]] = pair[1]
		}
	}
	if len(allowed) == 0 {
		return extra
	}
	primaryAuth := shadowProfileAuthHeader(dispatch.profile)
	mutations, _ := r.buildHeaderMutations(ctx)
	for _, mutation := range mutations {
		header := mutation.GetHeader()
		name := header.GetKey()
		if strings.HasPrefix(name, ":") || shadowHeaderIsSensitive(name, primaryAuth) {
			continue
		}
		for _, allowedName := range allowed {
			if strings.EqualFold(name, strings.TrimSpace(allowedName)) {
				extra[name] = string(header.GetRawValue())
				break
			}
		}
	}
	return extra
}

func (d *shadowDispatcher) submit(
	ctx *RequestContext,
	dispatch *providerDispatch,
	cfg config.RouterReplayShadowConfig,
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
	targets := make([]string, 0, len(cfg.Models))
	for _, model := range cfg.Models {
		if model != dispatch.logicalModel {
			targets = append(targets, model)
		}
	}
	if len(targets) == 0 {
		dropEarly(shadowReasonSameAsPrimary)
		return
	}
	snapshot, err := cloneSemanticRequestForReplay(ctx.SemanticRequest)
	if err != nil {
		dropEarly(shadowReasonRequestUnavailable)
		return
	}
	timeout, _ := time.ParseDuration(cfg.Execution.Timeout) // Validated when the config was loaded.
	groupCtx, cancel := context.WithDeadline(d.ctx, now.Add(timeout))
	groupCtx, _ = callbudget.WithLimit(groupCtx, len(targets))
	if ctx.cancelAbandonedWork != nil {
		ctx.cancelAbandonedWork()
	}
	ctx.cancelAbandonedWork = cancel
	jobs := make([]*shadowJob, 0, len(targets))
	for i, model := range targets {
		jobs = append(jobs, &shadowJob{
			cfg: cfg, model: model, context: groupCtx,
			responseLimit: shadowShare(*cfg.Budget.MaxResponseBytes, len(targets), i),
			captureLimit:  shadowShare(*cfg.Capture.MaxBytes, len(targets), i),
			routerConfig:  deps.routerConfig, engine: deps.engine, encode: deps.encode,
			request: *snapshot, extraHeaders: deps.extraHeaders, decision: decision, recipe: recipe,
			primaryRequest: ctx.RequestID, primaryModel: dispatch.logicalModel, primaryBackend: dispatch.backendName,
			shadowRequestID: shadowRequestID(ctx.RequestID), replayID: ctx.RouterReplayID, recorder: ctx.RouterReplayRecorder,
			enqueuedAt: now, timeout: timeout,
		})
	}
	d.enqueueGroup(config.RoutingDecisionKey(config.RecipeName(recipe), decision), jobs, cancel)
}

// Reservations partition each request's byte budget before any worker starts.
// Failed or unreadable responses cannot give another worker extra allowance.
func shadowShare(total, count, index int) int {
	share := total / count
	if index < total%count {
		share++
	}
	return share
}

func shadowRequestID(primary string) string {
	var buf [4]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return primary + "-shadow"
	}
	return primary + "-shadow-" + hex.EncodeToString(buf[:])
}

func (d *shadowDispatcher) laneFor(key string, cfg config.RouterReplayShadowConfig) *shadowLane {
	lane := d.lanes[key]
	if lane == nil || cap(lane.slots) != *cfg.Execution.MaxConcurrency || lane.maxQueue != *cfg.Execution.MaxQueueSize {
		lane = &shadowLane{
			slots:    make(chan struct{}, *cfg.Execution.MaxConcurrency),
			groups:   make(chan struct{}, *cfg.Execution.MaxConcurrency),
			maxQueue: *cfg.Execution.MaxQueueSize,
		}
		d.lanes[key] = lane
	}
	return lane
}

// enqueueGroup bounds both active groups and the pending group queue. Each
// active group competes for the lane's shared HTTP slots; candidates wait
// rather than being discarded when another candidate holds a slot.
func (d *shadowDispatcher) enqueueGroup(key string, jobs []*shadowJob, cancel context.CancelFunc) {
	first := jobs[0]
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		cancel()
		d.recordDrop(first, shadowReasonRouterClosing)
		return
	}
	lane := d.laneFor(key, first.cfg)
	queued := false
	select {
	case lane.groups <- struct{}{}:
	default:
		if int(lane.waiting.Load()) >= lane.maxQueue {
			d.mu.Unlock()
			cancel()
			d.recordDrop(first, shadowReasonQueueFull)
			return
		}
		lane.waiting.Add(1)
		queued = true
		metrics.ShadowDispatchQueued.WithLabelValues(first.decision).Inc()
	}
	d.wg.Add(1)
	d.mu.Unlock()
	goSafely("shadow_dispatch", func() {
		defer d.wg.Done()
		defer cancel()
		if queued {
			acquired := false
			select {
			case lane.groups <- struct{}{}:
				acquired = true
			case <-first.context.Done():
			}
			lane.waiting.Add(-1)
			metrics.ShadowDispatchQueued.WithLabelValues(first.decision).Dec()
			if !acquired {
				for _, job := range jobs {
					d.recordUnstarted(job)
				}
				return
			}
		}
		defer func() { <-lane.groups }()
		var workers sync.WaitGroup
		for _, job := range jobs {
			workers.Add(1)
			goSafely("shadow_candidate", func() {
				defer workers.Done()
				select {
				case lane.slots <- struct{}{}:
				case <-job.context.Done():
					d.recordUnstarted(job)
					return
				}
				defer func() { <-lane.slots }()
				d.execute(job)
			})
		}
		workers.Wait()
	})
}

func (d *shadowDispatcher) recordDrop(job *shadowJob, reason string) {
	metrics.RecordShadowDispatch(job.decision, metrics.ShadowDispatchResultDropped, reason)
	logging.ComponentDebugEvent("extproc", "shadow_dispatch_dropped", map[string]interface{}{
		"request_id": job.primaryRequest, "decision": job.decision, "reason": reason, "candidates": len(job.cfg.Models),
	})
}

func (d *shadowDispatcher) recordUnstarted(job *shadowJob) {
	reason := shadowReasonQueueTimeout
	if d.ctx.Err() != nil {
		reason = shadowReasonRouterClosing
	} else if job.context.Err() == context.Canceled {
		reason = "canceled"
	}
	now := d.now()
	d.record(job, shadowResult{verdict: shadowVerdictFailed, reason: reason, startedAt: now, finishedAt: now})
}

func (d *shadowDispatcher) execute(job *shadowJob) {
	// Reserve terminal evidence before spending a model call. Storage runs on
	// Replay's bounded writer, independently of the candidate execution slots.
	if job.recorder != nil && job.replayID != "" {
		job.receipt = job.recorder.TryReserveOutcome(job.replayID)
		if job.receipt == nil {
			d.recordDrop(job, shadowReasonReplayQueueFull)
			return
		}
	}
	metrics.ShadowDispatchInflight.WithLabelValues(job.decision).Inc()
	defer metrics.ShadowDispatchInflight.WithLabelValues(job.decision).Dec()
	d.record(job, d.call(job.context, job))
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
			TargetRef: job.model,
			Verdict:   result.verdict,
			Reason:    result.reason,
			Metadata:  metadata,
		}
		if job.receipt != nil {
			job.receipt.Finish(outcome)
		} else if !job.recorder.TryAppendOutcome(job.replayID, outcome) {
			// Unstarted work has no reservation and uses best-effort reporting.
			metrics.RecordPluginExecution("shadow_dispatch_receipt", job.decision, "dropped", 0)
			logging.ComponentWarnEvent("extproc", "shadow_dispatch_outcome_queue_full", map[string]interface{}{
				"request_id": job.primaryRequest,
				"replay_id":  job.replayID,
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
		"shadow_model":       job.model,
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
	if result.inputKnown {
		metadata["input_tokens"] = strconv.FormatInt(result.inputTokens, 10)
	}
	if result.outputKnown {
		metadata["output_tokens"] = strconv.FormatInt(result.outputTokens, 10)
	}
	metadata["usage_known"] = strconv.FormatBool(result.inputKnown && result.outputKnown)
	if result.inputKnown && result.outputKnown && job.routerConfig != nil {
		prompt, completion, currency, known := job.routerConfig.GetModelPricing(job.model)
		if known {
			metadata["estimated_cost"] = strconv.FormatFloat((float64(result.inputTokens)*prompt+float64(result.outputTokens)*completion)/1e6, 'g', -1, 64)
			metadata["cost_currency"] = currency
		}
	}
	metadata["response_chars"] = strconv.Itoa(utf8.RuneCountInString(result.text))
	sum := sha256.Sum256([]byte(result.text))
	metadata["response_sha256"] = hex.EncodeToString(sum[:])
	if job.cfg.Capture.Enabled {
		if len(result.text) > job.captureLimit {
			metadata["response_excerpt_truncated"] = "true"
		}
		metadata["response_excerpt"] = truncateShadowText(result.text, job.captureLimit)
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
