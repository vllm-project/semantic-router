package extproc

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageaccounting"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

type inferenceSettlementPlan struct {
	attemptEvidence accessruntime.AttemptEvidenceSnapshot
	aggregate       usageaccounting.Aggregate
	payload         string
	digest          string
	fenceID         string
}

func (r *OpenAIRouter) completeAndSettlePrimaryInference(
	ctx *RequestContext,
	usage responseUsageMetrics,
	statusCode int,
) error {
	if !r.nativeAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil || ctx.DispatchState == nil {
		return nil
	}
	state := ctx.DispatchState
	state.mu.Lock()
	dispatchID := state.selectedDispatchID
	state.mu.Unlock()
	if dispatchID == "" {
		return r.settleInference(ctx, statusCode, terminalErrorCode(statusCode), nil)
	}
	recorded := usageFromResponse(usage)
	r.completeInferenceDispatch(ctx, dispatchID, recorded, recorded.Reason)
	served := recorded.Usage
	if recorded.State != usageaccounting.EvidenceKnownActual {
		served = usageaccounting.ActualUsage{}
	}
	return r.settleInference(ctx, statusCode, terminalErrorCode(statusCode), &served)
}

func (r *OpenAIRouter) settleLooperInference(ctx *RequestContext, statusCode int) error {
	if !r.nativeAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil || ctx.DispatchState == nil {
		return nil
	}
	state := ctx.DispatchState
	state.mu.Lock()
	hasDispatch := len(state.dispatches) > 0
	state.mu.Unlock()
	if !hasDispatch {
		return r.settleNoBackendInference(ctx, statusCode, "looper_no_backend_dispatch")
	}
	// The dispatch journal already contains the authoritative neutral terminal
	// for every physical call. Do not reconstruct served usage from the public
	// Looper payload: that payload is a presentation artifact and can represent
	// a synthesis rather than a single backend completion.
	return r.settleInference(ctx, statusCode, terminalErrorCode(statusCode), nil)
}

func (r *OpenAIRouter) settleUnknownInference(ctx *RequestContext, statusCode int, reason string) error {
	if !r.nativeAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil || ctx.DispatchState == nil {
		return nil
	}
	state := ctx.DispatchState
	state.mu.Lock()
	if len(state.dispatches) == 0 {
		state.mu.Unlock()
		return r.settleNoBackendInference(ctx, statusCode, reason)
	}
	eligible := ensureSettlementDispatchLocked(state, true)
	if eligible == nil {
		state.mu.Unlock()
		return r.settleNoBackendInference(ctx, statusCode, reason)
	}
	for _, dispatch := range state.dispatches {
		if !dispatch.settlementEligible || !dispatch.completedAt.IsZero() {
			continue
		}
		dispatch.completedAt = time.Now().UTC()
		dispatch.state = usageaccounting.EvidenceUnknown
		dispatch.reason = canonicalUsageReason(reason)
	}
	state.mu.Unlock()
	return r.settleInference(ctx, statusCode, canonicalUsageReason(reason), nil)
}

func (r *OpenAIRouter) settleNoBackendInference(ctx *RequestContext, statusCode int, reason string) error {
	if !r.nativeAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil || ctx.DispatchState == nil {
		return nil
	}
	accessState := ctx.InferenceAccess
	accessState.mu.Lock()
	admitted := accessState.admission != nil
	settlementModel := accessState.settlementModel
	accessState.mu.Unlock()
	dispatchState := ctx.DispatchState
	dispatchState.mu.Lock()
	hasDispatch := len(dispatchState.dispatches) > 0
	if hasDispatch {
		ensureSettlementDispatchLocked(dispatchState, false)
	}
	dispatchState.mu.Unlock()
	if !admitted {
		return nil
	}
	if !hasDispatch {
		model := settlementModel
		if model == "" {
			model = ctx.VSRSelectedModel
		}
		if model == "" {
			model = ctx.RequestModel
		}
		dispatch, err := r.beginInferenceDispatch(ctx.TraceContext, ctx, model)
		if err != nil {
			return err
		}
		dispatchState.mu.Lock()
		dispatch.attempted = false
		dispatch.settlementEligible = true
		dispatch.attemptEvidenceRequired = false
		dispatchState.mu.Unlock()
		r.completeInferenceDispatch(ctx, dispatch.id, usageaccounting.DispatchUsage{State: usageaccounting.EvidenceKnownZero}, "")
	}
	zero := usageaccounting.ActualUsage{InputKnown: true, OutputKnown: true, CacheReadKnown: true, CacheWriteKnown: true}
	served := &zero
	if ctx.VSRCacheHit {
		// A cache hit has no backend usage, but its cached neutral response may
		// carry authoritative client-served token evidence. Missing evidence is
		// intentionally left unknown so served_* enforce rules open a fence.
		served = cacheHitSettlementUsage(ctx)
	}
	return r.settleInference(ctx, statusCode, canonicalUsageReason(reason), served)
}

func (r *OpenAIRouter) settleImmediateInference(
	ctx *RequestContext,
	response *ext_proc.ProcessingResponse,
	reason string,
) error {
	if !r.nativeAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil || ctx.DispatchState == nil {
		return nil
	}
	state := ctx.InferenceAccess
	state.mu.Lock()
	if state.admission == nil || state.finalized {
		state.mu.Unlock()
		return nil
	}
	dispatchState := ctx.DispatchState
	dispatchState.mu.Lock()
	hasDispatch := len(dispatchState.dispatches) > 0
	hasOpenDispatch := false
	hasSettlementDispatch := false
	for _, dispatch := range dispatchState.dispatches {
		if dispatch == nil || !dispatch.settlementEligible {
			continue
		}
		hasSettlementDispatch = true
		if dispatch.completedAt.IsZero() {
			hasOpenDispatch = true
			break
		}
	}
	dispatchState.mu.Unlock()
	state.mu.Unlock()

	statusCode := immediateInferenceStatus(response)
	if !hasDispatch || !hasSettlementDispatch {
		return r.settleNoBackendInference(ctx, statusCode, reason)
	}
	if hasOpenDispatch {
		return r.settleUnknownInference(ctx, statusCode, reason)
	}
	return r.settleInference(ctx, statusCode, terminalErrorCode(statusCode), nil)
}

func immediateInferenceStatus(response *ext_proc.ProcessingResponse) int {
	if response == nil || response.GetImmediateResponse() == nil || response.GetImmediateResponse().Status == nil {
		return 500
	}
	status := int(response.GetImmediateResponse().Status.Code)
	if status < 100 || status > 599 {
		return 500
	}
	return status
}

func (r *OpenAIRouter) settleInference(
	ctx *RequestContext,
	statusCode int,
	errorCode string,
	served *usageaccounting.ActualUsage,
) error {
	state := ctx.InferenceAccess
	stopInferenceAdmissionHeartbeat(state)
	state.mu.Lock()
	if state.finalized {
		state.mu.Unlock()
		return nil
	}
	if active := state.settlementRun; active != nil {
		active.waiters++
		state.mu.Unlock()
		err := waitForInferenceSettlement(ctx, active)
		state.mu.Lock()
		active.waiters--
		state.mu.Unlock()
		return err
	}
	if state.admission == nil {
		state.mu.Unlock()
		return nil
	}
	run := &inferenceSettlementRun{done: make(chan struct{})}
	state.settlementRun = run
	admission := *state.admission
	plan := state.settlement
	state.mu.Unlock()

	settlementContext, cancel := newInferenceSettlementContext(ctx)
	defer cancel()
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		if plan == nil {
			plan, err = r.buildInferenceSettlement(
				settlementContext, ctx, state, admission, statusCode, errorCode, served,
			)
			if errors.Is(err, quotaruntime.ErrEvidenceChanged) && attempt < 2 {
				plan = nil
				continue
			}
			if err != nil {
				break
			}
		}
		_, err = r.InferenceAccess.Settle(settlementContext, accessruntime.SettlementRequest{
			Admission: admission, AttemptEvidence: plan.attemptEvidence,
			Aggregate: plan.aggregate, FinalizationDigest: plan.digest,
			Event: plan.payload, FenceID: plan.fenceID,
		})
		if errors.Is(err, quotaruntime.ErrEvidenceChanged) && attempt < 2 {
			plan = nil
			continue
		}
		break
	}
	result := canonicalInferenceSettlementError(err)
	state.mu.Lock()
	if err == nil {
		state.finalized = true
		state.settlement = plan
	} else if !errors.Is(err, quotaruntime.ErrEvidenceChanged) && plan != nil {
		// A store acknowledgement can be lost after an atomic commit. Retain
		// the exact plan so the next call reuses its finalization digest.
		state.settlement = plan
	}
	run.err = result
	state.settlementRun = nil
	close(run.done)
	state.mu.Unlock()
	return result
}

func newInferenceSettlementContext(request *RequestContext) (context.Context, context.CancelFunc) {
	base := context.Background()
	if request != nil && request.TraceContext != nil {
		base = request.TraceContext
	}
	return context.WithTimeout(context.WithoutCancel(base), inferenceSettlementTimeout)
}

func waitForInferenceSettlement(request *RequestContext, run *inferenceSettlementRun) error {
	waitContext, cancel := newInferenceSettlementContext(request)
	defer cancel()
	select {
	case <-run.done:
		return run.err
	case <-waitContext.Done():
		return errInferenceSettlementUnavailable
	}
}

var errInferenceSettlementUnavailable = errors.New("inference usage settlement is temporarily unavailable")

func canonicalInferenceSettlementError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, quotaruntime.ErrEvidenceChanged) {
		return quotaruntime.ErrEvidenceChanged
	}
	return errInferenceSettlementUnavailable
}

func (r *OpenAIRouter) buildInferenceSettlement(
	settlementContext context.Context,
	ctx *RequestContext,
	state *inferenceRequestAccess,
	admission accessruntime.Admission,
	statusCode int,
	errorCode string,
	served *usageaccounting.ActualUsage,
) (*inferenceSettlementPlan, error) {
	dispatchState := ctx.DispatchState
	if dispatchState == nil {
		return nil, fmt.Errorf("settlement dispatch state is unavailable")
	}
	dispatchState.mu.Lock()
	if len(dispatchState.dispatches) == 0 {
		dispatchState.mu.Unlock()
		return nil, fmt.Errorf("settlement has no dispatch journal")
	}
	dispatches := make([]*inferenceDispatch, 0, len(dispatchState.dispatches))
	for index, dispatch := range dispatchState.dispatches {
		if dispatch == nil {
			dispatchState.mu.Unlock()
			return nil, fmt.Errorf("settlement dispatch %d is unavailable", index)
		}
		if !dispatch.settlementEligible {
			continue
		}
		clone := *dispatch
		clone.attempts = append([]usageledger.Attempt(nil), dispatch.attempts...)
		if clone.completedAt.IsZero() {
			clone.completedAt = time.Now().UTC()
			clone.state = usageaccounting.EvidenceUnknown
			clone.reason = "request_terminated"
		}
		dispatches = append(dispatches, &clone)
	}
	dispatchState.mu.Unlock()
	if len(dispatches) == 0 {
		return nil, fmt.Errorf("settlement has no attempted dispatch")
	}

	attemptEvidence, buildInferenceSettlementErr := r.InferenceAccess.ReadAttemptEvidence(
		settlementContext,
		attemptEvidenceRequest(admission, dispatches),
	)
	if buildInferenceSettlementErr != nil {
		return nil, buildInferenceSettlementErr
	}
	if err := reconcileAttemptEvidence(dispatches, attemptEvidence); err != nil {
		return nil, err
	}
	aggregator := usageaccounting.NewAggregator()
	for _, dispatch := range dispatches {
		if err := aggregator.RecordDispatch(usageaccounting.DispatchUsage{
			DispatchID: dispatch.id, ModelID: dispatch.modelID, ModelRevision: dispatch.modelRevision,
			State: dispatch.state, Usage: dispatch.usage, Pricing: dispatch.pricing, Reason: dispatch.reason,
		}); err != nil {
			return nil, err
		}
	}
	servedUsage := usageaccounting.ServedUsage{}
	if served != nil {
		servedUsage = usageaccounting.ServedUsage{
			Input: served.InputTotal, InputKnown: served.InputKnown,
			Output: served.Output, OutputKnown: served.OutputKnown,
		}
	}
	if err := aggregator.SetServedUsage(servedUsage); err != nil {
		return nil, err
	}
	aggregate, buildInferenceSettlementErr := aggregator.Finalize()
	if buildInferenceSettlementErr != nil {
		return nil, buildInferenceSettlementErr
	}
	fenceID := ""
	if settlementNeedsFence(admission, aggregate) {
		fenceID = uuid.NewString()
	}
	event, buildInferenceSettlementErr := r.buildTerminalUsageEvent(
		ctx, state, dispatches, aggregate, statusCode, errorCode, fenceID,
	)
	if buildInferenceSettlementErr != nil {
		return nil, buildInferenceSettlementErr
	}
	payload, buildInferenceSettlementErr := usageledger.EncodeTerminalEvent(event)
	if buildInferenceSettlementErr != nil {
		return nil, buildInferenceSettlementErr
	}
	return &inferenceSettlementPlan{
		attemptEvidence: attemptEvidence, aggregate: aggregate, payload: payload,
		digest: event.FinalizationDigest, fenceID: fenceID,
	}, nil
}

func settlementNeedsFence(admission accessruntime.Admission, aggregate usageaccounting.Aggregate) bool {
	for _, binding := range admission.Rules {
		if binding.Rule.Accounting == quota.AccountingResponseActual && !aggregate.Metric(binding.Rule.Metric).Complete {
			return true
		}
	}
	return false
}

func canonicalUsageReason(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "request_terminated"
	}
	value = strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_' || r == '-' || r == '.' {
			return r
		}
		if r >= 'A' && r <= 'Z' {
			return r + ('a' - 'A')
		}
		return '_'
	}, value)
	if len(value) > 128 {
		value = value[:128]
	}
	return value
}

func terminalErrorCode(status int) string {
	if status >= 200 && status < 400 {
		return ""
	}
	return fmt.Sprintf("upstream_%d", status)
}
