package extproc

import (
	"context"
	"errors"
	"fmt"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

var _ routing.FallbackSession = (*routingSession)(nil)

// Fallback hands the request's cross-model fallback to the caller, which
// sends the calls it prepares. From here on the response phases do not fall
// back, so the request keeps one fallback authority.
func (s *routingSession) Fallback() routing.Fallback {
	s.ctx.fallbackExecutedByCaller = true
	if s.fallback == nil {
		s.fallback = &sessionFallback{session: s}
	}
	return s.fallback
}

// sessionFallback is the native gateway's cross-model fallback for one
// request. It runs the response-phase fallback's policy (pkg/fallback),
// candidate order, request preparation and request-context bookkeeping; only
// the sending moves to the caller. Like the session's phases, its methods run
// on the request's goroutine.
//
// As in the response-phase fallback, the request context switches to a
// candidate only once the candidate succeeds; a chain that ends without
// success leaves the primary's context and the primary's failed response.
type sessionFallback struct {
	session *routingSession
	// primary is the context's dispatch state before the first candidate,
	// put back whenever a candidate does not succeed.
	primary *dispatchState
	// current is the candidate whose call is in flight; nil for the primary.
	current *fallbackCandidate
	sent    time.Time
}

// dispatchState is the part of the request context that preparing a
// candidate changes; the response-phase fallback saves and restores the same.
type dispatchState struct {
	selected    *config.ModelRef
	request     *llmprotocol.Request
	rateLimit   *ratelimit.Context
	diagnostics llmprotocol.Diagnostics
}

// Next records the failed call and prepares the next eligible candidate, or
// ends the chain when the policy, the candidates or the budget run out.
func (f *sessionFallback) Next(callCtx context.Context, outcome routing.Outcome) (routing.FallbackStep, error) {
	if f.session.ended || f.session.closed {
		return routing.FallbackStep{}, errSessionEnded
	}
	r, ctx := f.session.router, f.session.ctx
	orch := r.fallbackOrchestratorForContext(ctx)
	if !nativeFallbackApplies(orch, ctx) {
		return f.end(), nil
	}
	record := f.record(orch)
	evaluation := orch.EvaluateAttempt(callCtx, record, f.failedAttempt(outcome), fallbackCommitState(ctx))
	if !evaluation.CanFallback || totalTimeoutSpent(orch, ctx, record) {
		return f.end(), nil
	}
	for {
		index, err := orch.SelectNextCandidateIndex(record, len(ctx.VSREligibleModelRefs),
			func(i int) string { return candidateModelIdentity(ctx.VSREligibleModelRefs[i]) },
			func(model string) (string, error) { return r.fallbackBackendName(ctx, model), nil })
		if err != nil {
			return f.end(), nil
		}
		step, prepared, err := f.prepare(callCtx, orch, &ctx.VSREligibleModelRefs[index])
		if err == nil {
			return step, nil
		}
		// A candidate that cannot be prepared is skipped, unless the request's
		// time has run out, as in the response-phase fallback.
		if !prepared.CanFallback {
			if record.FinalStatus == "" {
				record.FinalStatus = fallbackHaltStatus(err)
			}
			return f.end(), nil
		}
	}
}

// prepare builds the candidate's call from the primary's dispatch state.
func (f *sessionFallback) prepare(
	callCtx context.Context, orch *fallback.Orchestrator, ref *config.ModelRef,
) (routing.FallbackStep, fallback.EvaluationResult, error) {
	s, ctx := f.session, f.session.ctx
	f.restore()
	if f.primary == nil {
		f.primary = &dispatchState{
			selected: ctx.VSRSelectedCandidate, request: ctx.SemanticRequest,
			rateLimit: ctx.RateLimitCtx, diagnostics: ctx.ProtocolDiagnostics,
		}
	}
	ctx.VSRSelectedCandidate = ref
	candidate, prepared, err := s.router.prepareFallbackCandidate(callCtx, ref, ctx, false)
	if err != nil {
		f.restore()
		return routing.FallbackStep{}, prepared, err
	}
	call, err := s.candidateCall(candidate)
	if err != nil {
		f.restore()
		return routing.FallbackStep{}, fallback.EvaluationResult{CanFallback: true}, err
	}
	call.Reliability = nil
	if override := f.candidateReliability(orch, call.Route); override != nil {
		call.Reliability = []*routing.Reliability{override}
	}
	f.current, f.sent = candidate, time.Now()
	return routing.FallbackStep{Call: call}, prepared, nil
}

// candidateReliability bounds a candidate by the policy's per_attempt_timeout
// and by what is left of its total_timeout, as the response-phase fallback
// bounds its HTTP attempt. A native candidate streams, so the bound is a
// per-try timeout, which never cuts a started response short; it only
// tightens the decision's or provider model's own per-try timeout.
func (f *sessionFallback) candidateReliability(orch *fallback.Orchestrator, route string) *routing.Reliability {
	ctx := f.session.ctx
	override := callReliability(ctx.VSRSelectedDecision)
	bound := orch.Policy().PerAttemptTimeout
	if total := orch.Policy().TotalTimeout; total > 0 && !ctx.ProcessingStartTime.IsZero() {
		if left := total - time.Since(ctx.ProcessingStartTime); bound <= 0 || left < bound {
			bound = max(left, time.Millisecond)
		}
	}
	if bound <= 0 {
		return override
	}
	var perTry *time.Duration
	if cfg := f.session.router.Config; cfg != nil {
		perTry = reliabilityDuration(cfg.ModelConfig[route].Reliability.PerTryTimeout)
	}
	if override != nil && override.PerTryTimeout != nil {
		perTry = override.PerTryTimeout
	}
	if perTry != nil && *perTry > 0 && *perTry <= bound {
		return override
	}
	if override == nil {
		override = &routing.Reliability{}
	}
	override.PerTryTimeout = &bound
	return override
}

// restore puts back the primary's dispatch state after a candidate that did
// not succeed.
func (f *sessionFallback) restore() {
	f.current = nil
	if f.primary == nil {
		return
	}
	ctx := f.session.ctx
	ctx.VSRSelectedCandidate = f.primary.selected
	ctx.SemanticRequest = f.primary.request
	ctx.RateLimitCtx = f.primary.rateLimit
	ctx.ProtocolDiagnostics = f.primary.diagnostics
}

// end closes the chain without a candidate's success: the caller returns the
// primary's failed response, which the primary's context describes.
func (f *sessionFallback) end() routing.FallbackStep {
	f.restore()
	f.session.router.recordFallbackExecution(f.session.ctx)
	return routing.FallbackStep{}
}

// nativeFallbackApplies mirrors shouldAttemptFallback for a caller that runs
// the chain: a response that reaches the caller has not been committed. A
// request-graph hop follows its resolved policy too.
func nativeFallbackApplies(orch *fallback.Orchestrator, ctx *RequestContext) bool {
	return orch != nil && orch.Policy().Enabled && ctx.SemanticRequest != nil &&
		len(ctx.VSREligibleModelRefs) > 1
}

// totalTimeoutSpent ends the chain once the policy's total budget is spent,
// recording it as the response-phase fallback does.
func totalTimeoutSpent(orch *fallback.Orchestrator, ctx *RequestContext, record *fallback.ExecutionRecord) bool {
	total := orch.Policy().TotalTimeout
	if total <= 0 || ctx.ProcessingStartTime.IsZero() {
		return false
	}
	elapsed := time.Since(ctx.ProcessingStartTime)
	if elapsed < total {
		return false
	}
	record.TotalDuration = elapsed
	record.FinalStatus = "total_deadline_exceeded"
	fallback.CorrelateExecution(record)
	return true
}

func (f *sessionFallback) record(orch *fallback.Orchestrator) *fallback.ExecutionRecord {
	ctx := f.session.ctx
	if ctx.FallbackRecord == nil {
		ctx.FallbackRecord = orch.NewExecutionRecord(ctx.RequestID, ctx.VSRSelectedDecisionName, f.model())
		ctx.FallbackRecord.SessionID = ctx.SessionID
		ctx.FallbackRecord.ConversationID = ctx.SemanticRequest.ConversationID
		if ctx.FallbackRecord.ConversationID == "" && ctx.ResponseObjectState != nil {
			ctx.FallbackRecord.ConversationID = ctx.ResponseObjectState.ConversationID
		}
	}
	return ctx.FallbackRecord
}

// failedAttempt describes a failed call as the response-phase fallback
// records it: the primary's from its upstream response, a candidate's from
// its dispatch.
func (f *sessionFallback) failedAttempt(outcome routing.Outcome) fallback.AttemptOutcome {
	attempt := fallback.AttemptOutcome{
		Model:      f.model(),
		Backend:    f.backend(),
		StatusCode: outcome.Status,
		Duration:   outcome.Duration,
		Discarded:  true,
	}
	if f.current != nil {
		attempt.ErrorMessage = string(outcome.Body)
		return attempt
	}
	message := fmt.Sprintf("upstream returned status %d", outcome.Status)
	if len(outcome.Body) > 0 {
		message = fmt.Sprintf("upstream returned status %d: %s", outcome.Status, outcome.Body)
	}
	attempt.Error, attempt.ErrorMessage = errors.New(message), message
	return attempt
}

// succeeded switches the request to the in-flight candidate once its
// successful response arrives, and closes the record.
func (f *sessionFallback) succeeded(status int) {
	candidate := f.current
	if candidate == nil {
		return
	}
	f.current, f.primary = nil, nil
	r, ctx := f.session.router, f.session.ctx
	if ctx.InflightToken != 0 {
		inflight.End(ctx.InflightModel, ctx.InflightToken)
		ctx.InflightToken = 0
	}
	ctx.SemanticRequest = candidate.request
	ctx.TargetFormat = candidate.dispatch.targetFormat
	ctx.ResponseVendor = resolveResponseVendor(candidate.dispatch.profile)
	ctx.RequestModel = candidate.model
	ctx.VSRSelectedModel = candidate.model
	ctx.ResponsePath = headers.ResponsePathFallback

	orch := r.fallbackOrchestratorForContext(ctx)
	if orch == nil || ctx.FallbackRecord == nil {
		return
	}
	attempt := fallback.AttemptOutcome{
		Model: candidate.model, Backend: candidate.dispatch.backendName,
		StatusCode: status, Duration: time.Since(f.sent),
	}
	if orch.EvaluateAttempt(ctx.TraceContext, ctx.FallbackRecord, attempt, fallbackCommitState(ctx)).Succeeded {
		metrics.RecordModelRouting(ctx.FallbackRecord.InitialModel, candidate.model)
	}
	r.recordFallbackExecution(ctx)
}

func (f *sessionFallback) model() string {
	if f.current != nil {
		return f.current.model
	}
	ctx := f.session.ctx
	if ctx.VSRSelectedModel != "" {
		return ctx.VSRSelectedModel
	}
	return ctx.RequestModel
}

func (f *sessionFallback) backend() string {
	if f.current != nil {
		return f.current.dispatch.backendName
	}
	return f.session.router.primaryBackendForAccounting(f.session.ctx, f.model())
}

// fallbackBackendName resolves a candidate's backend for the circuit breaker,
// as the response-phase fallback does.
func (r *OpenAIRouter) fallbackBackendName(ctx *RequestContext, model string) string {
	dispatch, err := r.resolveProviderDispatch(model, ctx.VSRSelectedDecisionName, r.candidateReasoningChoice(ctx, model))
	if err == nil {
		return dispatch.backendName
	}
	for _, ref := range ctx.VSREligibleModelRefs {
		if candidateModelIdentity(ref) == model && ref.Model != "" {
			if base, baseErr := r.resolveProviderDispatch(ref.Model, ctx.VSRSelectedDecisionName, false); baseErr == nil {
				return base.backendName
			}
		}
	}
	return model
}

func fallbackCommitState(ctx *RequestContext) fallback.CommitState {
	return fallback.CommitState{
		HasNonIdempotentSideEffects: ctx.hasNonIdempotentToolExecution(),
		BodyReplayable:              ctx.SemanticRequest != nil,
	}
}

// candidateCall builds the upstream request for a prepared candidate from the
// client request as the header phase left it, with the provider dispatch the
// primary's body phase would have produced for that candidate: routing
// headers, path, provider metadata, credentials, decision headers and body.
// Starting from the client request keeps the primary provider's credentials
// and headers off the candidate's request.
func (s *routingSession) candidateCall(candidate *fallbackCandidate) (*routing.Call, error) {
	if s.ctx.UpstreamSpan != nil {
		s.ctx.UpstreamSpan.End()
	}
	response := s.router.buildProviderDispatchResponse(candidate.dispatch, s.ctx)
	common := response.GetRequestBody().GetResponse()
	if common == nil || common.HeaderMutation == nil {
		return nil, fmt.Errorf("fallback candidate %q has no provider dispatch", candidate.model)
	}
	appendContentLengthHeader(&common.HeaderMutation.SetHeaders, len(candidate.body))
	common.BodyMutation = &ext_proc.BodyMutation{Mutation: &ext_proc.BodyMutation_Body{Body: candidate.body}}
	effect, err := routingEffect(response)
	if err != nil {
		return nil, err
	}
	header := s.clientHeader.Clone()
	if err := routing.ApplyHeaderMutation(&header, effect.Header, false, routing.DefaultLimits); err != nil {
		return nil, err
	}
	return &routing.Call{
		Route:   header.Get(routing.RouteHeader),
		Request: routing.Request{Header: header, Body: routing.ApplyBodyMutation(nil, effect.Body)},
	}, nil
}
