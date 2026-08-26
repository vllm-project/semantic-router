package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
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

type requestDispatchState struct {
	mu                    sync.Mutex
	requestID             string
	dispatches            []*inferenceDispatch
	primaryDispatchID     string
	primaryCandidateCount int
	fallback              backendinvoker.FallbackPolicy
	capabilityIssued      bool
	requestDigest         string
	outcomeConsumed       bool
	noDispatchProven      bool
	selectedDispatchID    string
}

type inferenceDispatchObserver struct {
	router  *OpenAIRouter
	request *RequestContext
}

func (r *OpenAIRouter) installInferenceDispatchObserver(ctx *RequestContext) {
	if ctx == nil || ctx.DispatchState == nil {
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
		reference, referenceFound := responseTerminalReference(o.request.DispatchState, completion.DispatchID)
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
	state := request.DispatchState
	if state == nil {
		return fmt.Errorf("request dispatch state is missing")
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
		return fmt.Errorf("request dispatch plan is empty")
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
	if request == nil || request.DispatchState == nil {
		return nil, fmt.Errorf("request dispatch state is unavailable")
	}
	params, ok := r.Config.ModelConfig[model]
	if !ok || params.ResourceID == "" || params.ResourceRevision <= 0 {
		return nil, fmt.Errorf("model %q lacks immutable runtime identity", model)
	}
	generation, generationFound := routingcontext.GenerationFrom(ctx)
	if !generationFound {
		return nil, fmt.Errorf("durable routing generation is unavailable")
	}
	admission, pricing, accessErr := r.inferenceDispatchAdmission(request, params)
	if accessErr != nil {
		return nil, accessErr
	}
	state := request.DispatchState
	state.mu.Lock()
	defer state.mu.Unlock()
	if int64(len(state.dispatches)) > math.MaxUint32 {
		return nil, fmt.Errorf("inference dispatch limit exceeded")
	}
	// #nosec G115 -- the request-local dispatch count is bounded to MaxUint32 above.
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

func (r *OpenAIRouter) inferenceDispatchAdmission(
	request *RequestContext,
	params config.ModelParams,
) (*accessruntime.Admission, usageaccounting.Pricing, error) {
	if !r.nativeAccessEnabled() {
		return nil, usageaccounting.Pricing{}, nil
	}
	state := request.InferenceAccess
	if state == nil || r.InferenceAccess == nil {
		return nil, usageaccounting.Pricing{}, fmt.Errorf("inference access runtime is unavailable")
	}
	state.mu.Lock()
	if state.admission == nil || state.admission.Tenant.NamespaceID == "" || state.finalized || state.settlementRun != nil {
		state.mu.Unlock()
		return nil, usageaccounting.Pricing{}, fmt.Errorf("inference request is not admitted")
	}
	admitted := *state.admission
	state.mu.Unlock()
	pricing, err := usageaccounting.CompilePricing(usageaccounting.PricingInput{
		Currency:   admitted.Tenant.BillingCurrency,
		Input:      params.RuntimePricing.InputCostPerMillionTokens,
		Output:     params.RuntimePricing.OutputCostPerMillionTokens,
		CacheRead:  params.RuntimePricing.CacheReadCostPerMillionTokens,
		CacheWrite: params.RuntimePricing.CacheWriteCostPerMillionTokens,
	})
	if err != nil {
		return nil, usageaccounting.Pricing{}, fmt.Errorf("model pricing: %w", err)
	}
	return &admitted, pricing, nil
}

func responseTerminalReference(
	state *requestDispatchState,
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
		request.DispatchState == nil || dispatch == nil {
		return "", fmt.Errorf("looper dispatch capability runtime is unavailable")
	}
	generation, ok := routingcontext.GenerationFrom(ctx)
	if !ok {
		return "", fmt.Errorf("durable routing generation is unavailable")
	}
	facts := accessruntime.DispatchFacts{
		DispatchID: dispatch.id, Ordinal: dispatch.ordinal,
		DispatchPlanDigest: dispatch.planDigest,
	}
	model := dispatchauthority.ModelIdentity{ID: dispatch.modelID, Revision: dispatch.modelRevision}
	var grant string
	var err error
	if r.nativeAccessEnabled() {
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
	if request == nil || request.DispatchState == nil || dispatchID == "" {
		return
	}
	state := request.DispatchState
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
	if request == nil || request.DispatchState == nil {
		return
	}
	state := request.DispatchState
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
