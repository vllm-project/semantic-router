package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageaccounting"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

type inferenceDispatch struct {
	id                      string
	ordinal                 uint32
	model                   string
	modelID                 string
	modelRevision           int64
	priority                int
	planDigest              string
	pricing                 usageaccounting.Pricing
	startedAt               time.Time
	completedAt             time.Time
	state                   usageaccounting.EvidenceState
	usage                   usageaccounting.ActualUsage
	reason                  string
	planned                 bool
	attempted               bool
	settlementEligible      bool
	attemptEvidenceRequired bool
	dispatchType            string
	terminalReference       backendinvoker.ResponseTerminalReference
	attempts                []usageledger.Attempt
}

type managedRequestDispatch struct {
	mu                    sync.Mutex
	requestID             string
	dispatches            []*inferenceDispatch
	primaryDispatchID     string
	primaryCandidateCount int
	fallback              backendinvoker.FallbackPolicy
	capabilityIssued      bool
	requestDigest         string
	outcomeConsumed       bool
	selectedDispatchID    string
}

type inferenceSettlementPlan struct {
	attemptEvidence accessruntime.AttemptEvidenceSnapshot
	aggregate       usageaccounting.Aggregate
	payload         string
	digest          string
	fenceID         string
}

type inferenceDispatchObserver struct {
	router  *OpenAIRouter
	request *RequestContext
}

func (r *OpenAIRouter) installInferenceDispatchObserver(ctx *RequestContext) {
	if ctx == nil || ctx.ManagedDispatch == nil {
		return
	}
	base := ctx.TraceContext
	if base == nil {
		base = context.Background()
	}
	ctx.TraceContext = looper.WithDispatchObserver(base, &inferenceDispatchObserver{router: r, request: ctx})
}

func (o *inferenceDispatchObserver) Started(ctx context.Context, start looper.DispatchStart) (looper.DispatchAuthorization, error) {
	if o == nil || o.router == nil || o.request == nil {
		return looper.DispatchAuthorization{}, fmt.Errorf("inference dispatch observer is unavailable")
	}
	dispatch, err := o.router.beginInferenceDispatch(ctx, o.request, start.Model)
	if err != nil {
		return looper.DispatchAuthorization{}, err
	}
	grant, err := o.router.issueLooperDispatchGrant(ctx, o.request, dispatch)
	if err != nil {
		return looper.DispatchAuthorization{}, err
	}
	return looper.DispatchAuthorization{
		DispatchID: dispatch.id, Grant: grant, RequestID: o.request.RequestID,
	}, nil
}

func (o *inferenceDispatchObserver) Completed(completionContext context.Context, completion looper.DispatchCompletion) {
	if o == nil || o.router == nil || o.request == nil {
		return
	}
	if !completion.HTTPStarted {
		o.router.completeInferenceDispatch(o.request, completion.DispatchID, usageaccounting.DispatchUsage{State: usageaccounting.EvidenceKnownZero}, "")
		return
	}
	terminalReason := "response_terminal_missing"
	if o.router.ResponseTerminals != nil {
		reference, referenceFound := responseTerminalReference(o.request.ManagedDispatch, completion.DispatchID)
		if referenceFound {
			takeContext := completionContext
			if takeContext == nil {
				takeContext = o.request.TraceContext
			}
			if takeContext == nil {
				takeContext = context.Background()
			}
			takeContext = context.WithoutCancel(takeContext)
			record, found, err := o.router.ResponseTerminals.Take(takeContext, reference)
			if err == nil && found && record.Reference == reference {
				usage := usageFromResponse(responseUsageFromTerminal(record))
				o.router.completeInferenceDispatch(o.request, completion.DispatchID, usage, usage.Reason)
				return
			}
			if err != nil {
				terminalReason = responseTerminalFailureReason(err)
			} else if found {
				terminalReason = "response_terminal_invalid"
			}
		}
	}
	reason := completion.FailureCode
	if reason == "" {
		reason = terminalReason
	}
	o.router.completeInferenceDispatch(o.request, completion.DispatchID, usageaccounting.DispatchUsage{State: usageaccounting.EvidenceUnknown}, reason)
}

func (r *OpenAIRouter) beginPrimaryInferenceDispatch(ctx context.Context, request *RequestContext, model string) error {
	if request == nil || request.LooperRequest {
		return nil
	}
	state := request.ManagedDispatch
	if state == nil {
		return fmt.Errorf("managed dispatch state is missing")
	}
	state.mu.Lock()
	already := state.primaryDispatchID != ""
	state.mu.Unlock()
	if already {
		return nil
	}
	candidates, fallback, err := r.primaryDispatchCandidates(request, model)
	if err != nil {
		return err
	}
	planned := make([]*inferenceDispatch, 0, len(candidates))
	for _, candidate := range candidates {
		dispatch, dispatchErr := r.beginInferenceDispatchCandidate(
			ctx, request, candidate.model, candidate.priority, true, false,
		)
		if dispatchErr != nil {
			return dispatchErr
		}
		planned = append(planned, dispatch)
	}
	if len(planned) == 0 {
		return fmt.Errorf("managed dispatch plan is empty")
	}
	state.mu.Lock()
	state.primaryDispatchID = planned[0].id
	state.primaryCandidateCount = len(planned)
	state.fallback = backendinvoker.FallbackPolicy{On: append(
		[]backendinvoker.FallbackTrigger(nil), fallback.On...,
	)}
	state.mu.Unlock()
	return nil
}

func (r *OpenAIRouter) beginInferenceDispatch(ctx context.Context, request *RequestContext, model string) (*inferenceDispatch, error) {
	return r.beginInferenceDispatchCandidate(ctx, request, model, 0, false, true)
}

func (r *OpenAIRouter) beginInferenceDispatchCandidate(
	ctx context.Context,
	request *RequestContext,
	model string,
	priority int,
	planned bool,
	attempted bool,
) (*inferenceDispatch, error) {
	if request == nil || request.ManagedDispatch == nil {
		return nil, fmt.Errorf("managed dispatch state is unavailable")
	}
	params, ok := r.Config.ModelConfig[model]
	if !ok || params.ResourceID == "" || params.ResourceRevision <= 0 {
		return nil, fmt.Errorf("model %q lacks immutable runtime identity", model)
	}
	generation, generationFound := routingcontext.GenerationFrom(ctx)
	if !generationFound {
		return nil, fmt.Errorf("managed routing generation is unavailable")
	}
	var admission *accessruntime.Admission
	pricing := usageaccounting.Pricing{}
	if r.managedInferenceAccessEnabled() {
		state := request.InferenceAccess
		if state == nil || r.InferenceAccess == nil {
			return nil, fmt.Errorf("inference access runtime is unavailable")
		}
		state.mu.Lock()
		if state.admission == nil || state.admission.Tenant.NamespaceID == "" || state.finalized || state.settlementRun != nil {
			state.mu.Unlock()
			return nil, fmt.Errorf("inference request is not admitted")
		}
		admitted := *state.admission
		admission = &admitted
		state.mu.Unlock()
		var err error
		pricing, err = usageaccounting.CompilePricing(usageaccounting.PricingInput{
			Currency:   admitted.Tenant.BillingCurrency,
			Input:      params.RuntimePricing.InputCostPerMillionTokens,
			Output:     params.RuntimePricing.OutputCostPerMillionTokens,
			CacheRead:  params.RuntimePricing.CacheReadCostPerMillionTokens,
			CacheWrite: params.RuntimePricing.CacheWriteCostPerMillionTokens,
		})
		if err != nil {
			return nil, fmt.Errorf("model pricing: %w", err)
		}
	}
	state := request.ManagedDispatch
	state.mu.Lock()
	defer state.mu.Unlock()
	ordinal := uint32(len(state.dispatches))
	id := "dispatch-" + uuid.NewString()
	digestIdentity := request.RequestID
	if admission != nil {
		digestIdentity = admission.Tenant.AdmissionID
	}
	digest := dispatchPlanDigest(digestIdentity, id, ordinal, params)
	dispatchType := "looper"
	if planned {
		dispatchType = "primary"
	}
	admissionID, admissionDigest := dispatchauthority.RoutingOnlyAdmissionIdentity(generation, request.RequestID)
	if admission != nil {
		admissionID = admission.Tenant.AdmissionID
		admissionDigest = admission.RequestDigest
	}
	terminalReference := backendinvoker.ResponseTerminalReference{
		NamespaceID: generation.NamespaceID, QuotaPartition: generation.QuotaPartition,
		PublicationID: generation.PublicationID, RuntimeEpoch: generation.RuntimeEpoch,
		RoutingRevision: generation.SnapshotRevision, RoutingDigest: generation.RoutingDigest,
		AdmissionID: admissionID, AdmissionDigest: admissionDigest,
		RequestID: request.RequestID, DispatchID: id, DispatchType: dispatchType,
		Ordinal: int(ordinal), Priority: priority, DispatchPlanDigest: digest,
		ModelID: params.ResourceID, ModelRevision: params.ResourceRevision,
	}
	if err := terminalReference.Validate(); err != nil {
		return nil, fmt.Errorf("response terminal reference: %w", err)
	}
	if admission != nil {
		if _, err := r.InferenceAccess.JournalDispatch(ctx, accessruntime.DispatchJournalRequest{
			Admission: *admission, DispatchID: id, Ordinal: ordinal, Digest: digest,
		}); err != nil {
			return nil, err
		}
	}
	dispatch := &inferenceDispatch{
		id: id, ordinal: ordinal, model: model, modelID: params.ResourceID,
		modelRevision: params.ResourceRevision, priority: priority,
		planDigest: digest, pricing: pricing,
		startedAt: time.Now().UTC(), state: usageaccounting.EvidenceUnknown,
		reason: "dispatch_not_terminal", planned: planned, attempted: attempted,
		settlementEligible:      attempted,
		attemptEvidenceRequired: attempted,
		dispatchType:            dispatchType,
		terminalReference:       terminalReference,
	}
	state.dispatches = append(state.dispatches, dispatch)
	return dispatch, nil
}

func responseTerminalReference(
	state *managedRequestDispatch,
	dispatchID string,
) (backendinvoker.ResponseTerminalReference, bool) {
	if state == nil || dispatchID == "" {
		return backendinvoker.ResponseTerminalReference{}, false
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	for _, dispatch := range state.dispatches {
		if dispatch != nil && dispatch.id == dispatchID {
			if err := dispatch.terminalReference.Validate(); err != nil {
				return backendinvoker.ResponseTerminalReference{}, false
			}
			return dispatch.terminalReference, true
		}
	}
	return backendinvoker.ResponseTerminalReference{}, false
}

func (r *OpenAIRouter) issueLooperDispatchGrant(
	ctx context.Context,
	request *RequestContext,
	dispatch *inferenceDispatch,
) (string, error) {
	if r == nil || r.DispatchCapabilities == nil || request == nil ||
		request.ManagedDispatch == nil || dispatch == nil {
		return "", fmt.Errorf("looper dispatch capability runtime is unavailable")
	}
	generation, ok := routingcontext.GenerationFrom(ctx)
	if !ok {
		return "", fmt.Errorf("managed routing generation is unavailable")
	}
	facts := accessruntime.DispatchFacts{
		DispatchID: dispatch.id, Ordinal: dispatch.ordinal,
		DispatchPlanDigest: dispatch.planDigest,
	}
	model := dispatchauthority.ModelIdentity{ID: dispatch.modelID, Revision: dispatch.modelRevision}
	var grant string
	var err error
	if r.managedInferenceAccessEnabled() {
		state := request.InferenceAccess
		if state == nil {
			return "", fmt.Errorf("looper dispatch admission is unavailable")
		}
		state.mu.Lock()
		if state.admission == nil {
			state.mu.Unlock()
			return "", fmt.Errorf("looper dispatch admission is unavailable")
		}
		admission := *state.admission
		state.mu.Unlock()
		grant, err = r.DispatchCapabilities.IssueMeteredGrant(dispatchauthority.GrantIssueRequest{
			Admission: admission, Dispatch: facts, RequestID: request.RequestID, Model: model,
		})
	} else {
		grant, err = r.DispatchCapabilities.IssueRoutingOnlyGrant(ctx, dispatchauthority.RoutingOnlyGrantIssueRequest{
			Generation: generation, Dispatch: facts, RequestID: request.RequestID, Model: model,
		})
	}
	if err != nil {
		return "", err
	}
	markInferenceDispatchAttemptEvidenceRequired(request, dispatch.id)
	return grant, nil
}

func markInferenceDispatchAttemptEvidenceRequired(request *RequestContext, dispatchID string) {
	if request == nil || request.ManagedDispatch == nil || dispatchID == "" {
		return
	}
	state := request.ManagedDispatch
	state.mu.Lock()
	defer state.mu.Unlock()
	for _, dispatch := range state.dispatches {
		if dispatch.id == dispatchID {
			dispatch.attempted = true
			dispatch.settlementEligible = true
			dispatch.attemptEvidenceRequired = true
			return
		}
	}
}

func dispatchPlanDigest(admissionID, dispatchID string, ordinal uint32, params config.ModelParams) string {
	payload := fmt.Sprintf("vllm-sr/dispatch/v1\x00%s\x00%s\x00%d\x00%s\x00%d", admissionID, dispatchID, ordinal, params.ResourceID, params.ResourceRevision)
	digest := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(digest[:])
}

func (r *OpenAIRouter) completeInferenceDispatch(
	request *RequestContext,
	dispatchID string,
	usage usageaccounting.DispatchUsage,
	reason string,
) {
	if request == nil || request.ManagedDispatch == nil {
		return
	}
	state := request.ManagedDispatch
	state.mu.Lock()
	defer state.mu.Unlock()
	for _, dispatch := range state.dispatches {
		if dispatch.id != dispatchID || !dispatch.completedAt.IsZero() {
			continue
		}
		dispatch.completedAt = time.Now().UTC()
		dispatch.state = usage.State
		dispatch.usage = usage.Usage
		dispatch.reason = reason
		return
	}
}

func usageFromResponse(value responseUsageMetrics) usageaccounting.DispatchUsage {
	if value.invalid || !responseUsageHasPricableBreakdown(value) {
		reason := value.invalidReason
		if reason == "" {
			reason = "authoritative_usage_missing"
		}
		return usageaccounting.DispatchUsage{State: usageaccounting.EvidenceUnknown, Reason: reason}
	}
	input, inputErr := quota.ParseQuotaInteger(fmt.Sprintf("%d", value.promptTokens))
	output, outputErr := quota.ParseQuotaInteger(fmt.Sprintf("%d", value.completionTokens))
	cacheRead, readErr := quota.ParseQuotaInteger(fmt.Sprintf("%d", value.cachedPromptTokens))
	cacheWrite, writeErr := quota.ParseQuotaInteger(fmt.Sprintf("%d", value.cacheWriteTokens))
	if inputErr != nil || outputErr != nil || readErr != nil || writeErr != nil {
		return usageaccounting.DispatchUsage{State: usageaccounting.EvidenceUnknown, Reason: "invalid_authoritative_usage"}
	}
	return usageaccounting.DispatchUsage{State: usageaccounting.EvidenceKnownActual, Usage: usageaccounting.ActualUsage{
		InputTotal: input, InputKnown: true, Output: output, OutputKnown: true,
		CacheRead: cacheRead, CacheReadKnown: value.cachedPromptTokensReported,
		CacheWrite: cacheWrite, CacheWriteKnown: value.cacheWriteTokensReported,
	}}
}

func (r *OpenAIRouter) completeAndSettlePrimaryInference(
	ctx *RequestContext,
	usage responseUsageMetrics,
	statusCode int,
) error {
	if !r.managedInferenceAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil || ctx.ManagedDispatch == nil {
		return nil
	}
	state := ctx.ManagedDispatch
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

func (r *OpenAIRouter) settleLooperInference(
	ctx *RequestContext,
	statusCode int,
) error {
	if !r.managedInferenceAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil || ctx.ManagedDispatch == nil {
		return nil
	}
	state := ctx.ManagedDispatch
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
	if !r.managedInferenceAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil || ctx.ManagedDispatch == nil {
		return nil
	}
	state := ctx.ManagedDispatch
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
	if !r.managedInferenceAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil || ctx.ManagedDispatch == nil {
		return nil
	}
	accessState := ctx.InferenceAccess
	accessState.mu.Lock()
	admitted := accessState.admission != nil
	settlementModel := accessState.settlementModel
	accessState.mu.Unlock()
	dispatchState := ctx.ManagedDispatch
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
	return r.settleInference(ctx, statusCode, canonicalUsageReason(reason), &zero)
}

func (r *OpenAIRouter) settleImmediateInference(
	ctx *RequestContext,
	response *ext_proc.ProcessingResponse,
	reason string,
) error {
	if !r.managedInferenceAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil || ctx.ManagedDispatch == nil {
		return nil
	}
	state := ctx.InferenceAccess
	state.mu.Lock()
	if state.admission == nil || state.finalized {
		state.mu.Unlock()
		return nil
	}
	dispatchState := ctx.ManagedDispatch
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
	dispatchState := ctx.ManagedDispatch
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
