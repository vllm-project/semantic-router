package extproc

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

func TestStreamingSettlementUsesNeutralAuthoritativeActualAndIsIdempotent(t *testing.T) {
	cfg := inferenceTestConfig(t)
	fake := &fakeInferenceAccess{readEvidence: responseStartedEvidence}
	router := &OpenAIRouter{Config: cfg, InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	if err := router.beginPrimaryInferenceDispatch(ctx.TraceContext, ctx, "internal-model"); err != nil {
		t.Fatal(err)
	}
	applyPrimaryDispatchOutcome(t, ctx, backendinvoker.AttemptResponseStarted)
	usage := responseUsageFromSemanticUsage(authoritativeZeroUsage())
	if usage.invalid || !responseUsageHasPricableBreakdown(usage) {
		t.Fatalf("neutral authoritative zero was not accepted: %+v", usage)
	}
	if err := router.completeAndSettlePrimaryInference(ctx, usage, 200); err != nil {
		t.Fatal(err)
	}
	if err := router.completeAndSettlePrimaryInference(ctx, usage, 200); err != nil {
		t.Fatal(err)
	}
	if len(fake.settlements) != 1 {
		t.Fatalf("settlements = %d, want exactly one", len(fake.settlements))
	}
	event, err := usageledger.DecodeTerminalEvent(fake.settlements[0].Event)
	if err != nil {
		t.Fatal(err)
	}
	if len(event.Dispatches) != 1 || event.Dispatches[0].UsageState != usageledger.UsageKnownActual {
		t.Fatalf("explicit zero event = %+v", event.Dispatches)
	}
	if event.ExternalRequestID != ctx.RequestID {
		t.Fatalf("external request ID = %q, want %q", event.ExternalRequestID, ctx.RequestID)
	}
}

func TestSettlementEvidenceCoversCompleteFallbackJournalWithoutBillingUnattemptedCandidates(t *testing.T) {
	cfg := inferenceTestConfig(t)
	base := cfg.ModelConfig["internal-model"]
	for index, name := range []string{"fallback-one", "fallback-two"} {
		params := base
		params.ResourceID = fmt.Sprintf("mdl_fallback_%d", index+1)
		params.ResourceRevision = int64(index + 2)
		cfg.ModelConfig[name] = params
	}
	pending := true
	fake := &fakeInferenceAccess{}
	fake.readEvidence = func(request accessruntime.AttemptEvidenceRequest) (accessruntime.AttemptEvidenceSnapshot, error) {
		snapshot, err := responseStartedEvidence(request)
		if err != nil {
			return accessruntime.AttemptEvidenceSnapshot{}, err
		}
		for index := 1; index < len(snapshot.Dispatches); index++ {
			snapshot.Dispatches[index] = accessruntime.AttemptEvidenceObservation{
				DispatchID: request.Dispatches[index].DispatchID,
			}
		}
		return snapshot, nil
	}
	fake.settle = func(request accessruntime.SettlementRequest) (quotaruntime.FinalizationResult, error) {
		if len(request.AttemptEvidence.Observations()) != len(fake.journal) {
			return quotaruntime.FinalizationResult{}, fmt.Errorf("settlement dispatch journal differs")
		}
		if request.FenceID != "" {
			return quotaruntime.FinalizationResult{}, fmt.Errorf("known fallback settlement opened a fence")
		}
		pending = false
		return quotaruntime.FinalizationResult{}, nil
	}
	router := &OpenAIRouter{Config: cfg, InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	for index, name := range []string{"internal-model", "fallback-one", "fallback-two"} {
		if _, err := router.beginInferenceDispatchCandidate(
			ctx.TraceContext, ctx, name, index, true, false,
		); err != nil {
			t.Fatal(err)
		}
	}
	ctx.DispatchState.mu.Lock()
	ctx.DispatchState.primaryDispatchID = ctx.DispatchState.dispatches[0].id
	ctx.DispatchState.primaryCandidateCount = len(ctx.DispatchState.dispatches)
	ctx.DispatchState.mu.Unlock()
	applyPrimaryDispatchOutcome(t, ctx, backendinvoker.AttemptResponseStarted)

	usage := responseUsageMetrics{
		promptTokens: 7, promptTokensReported: true,
		completionTokens: 3, completionTokensReported: true,
		totalTokens: 10, totalTokensReported: true,
	}
	if err := router.completeAndSettlePrimaryInference(ctx, usage, 200); err != nil {
		t.Fatal(err)
	}
	if len(fake.evidenceReads) != 1 || len(fake.evidenceReads[0].Dispatches) != 3 {
		t.Fatalf("attempt evidence read = %+v, want the complete three-candidate journal", fake.evidenceReads)
	}
	if len(fake.settlements) != 1 {
		t.Fatalf("settlement evidence = %+v, want all three journaled candidates", fake.settlements)
	}
	observations := fake.settlements[0].AttemptEvidence.Observations()
	if len(observations) != 3 {
		t.Fatalf("settlement evidence = %+v, want all three journaled candidates", fake.settlements)
	}
	if !observations[0].Present || observations[1].Present || observations[2].Present {
		t.Fatalf("fallback execution evidence = %+v, want one attempted and two not executed", observations)
	}
	event, err := usageledger.DecodeTerminalEvent(fake.settlements[0].Event)
	if err != nil {
		t.Fatal(err)
	}
	if len(event.Dispatches) != 1 || event.Dispatches[0].ModelID != base.ResourceID ||
		event.Dispatches[0].UsageState != usageledger.UsageKnownActual ||
		event.EvidenceState != usageledger.EvidenceKnown || event.Fence != nil ||
		fake.settlements[0].FenceID != "" {
		t.Fatalf("terminal usage included an unattempted fallback candidate: %+v", event)
	}
	if pending {
		t.Fatal("known fallback settlement left the admission pending")
	}
}

func TestTerminalExternalRequestIDRejectsUntrustedValues(t *testing.T) {
	requestID := uuid.NewString()
	if terminalExternalRequestID(requestID) != requestID {
		t.Fatal("canonical request ID was discarded")
	}
	for _, value := range []string{"authorization: Bearer secret", " request-1 ", "request-1"} {
		if terminalExternalRequestID(value) != "" {
			t.Fatalf("unsafe external request ID %q was retained", value)
		}
	}
}

func TestSettlementUsesAuthoritativeRetryEvidenceWithoutDoubleCounting(t *testing.T) {
	cfg := inferenceTestConfig(t)
	fake := &fakeInferenceAccess{}
	fake.readEvidence = func(request accessruntime.AttemptEvidenceRequest) (accessruntime.AttemptEvidenceSnapshot, error) {
		dispatch := request.Dispatches[0]
		started := time.Now().UTC().Add(-time.Second).Truncate(time.Millisecond)
		return accessruntime.AttemptEvidenceSnapshot{Dispatches: []accessruntime.AttemptEvidenceObservation{{
			DispatchID: dispatch.DispatchID, Present: true,
			Evidence: quotaruntime.DispatchAttemptEvidence{
				DispatchID: dispatch.DispatchID, DispatchType: "primary", Ordinal: dispatch.Ordinal,
				DispatchPlanDigest: dispatch.DispatchPlanDigest, ModelID: dispatch.ModelID,
				ModelRevision: dispatch.ModelRevision, StartedAt: started,
				Deadline: started.Add(time.Minute), MaxAttempts: 2,
				Attempts: []quotaruntime.AttemptEvidence{
					{
						AttemptID: dispatch.DispatchID + ":1", AttemptNumber: 1,
						BackendID: uuid.NewString(), ProviderID: "test-provider",
						State: quotaruntime.AttemptEvidenceKnownZero, ErrorCode: "transport_error",
						StartedAt: started, CompletedAt: started.Add(10 * time.Millisecond), Finished: true,
					},
					{
						AttemptID: dispatch.DispatchID + ":2", AttemptNumber: 2,
						BackendID: uuid.NewString(), ProviderID: "test-provider",
						State: quotaruntime.AttemptEvidenceResponseStarted, StatusCode: 200,
						StartedAt: started.Add(20 * time.Millisecond), CompletedAt: started.Add(30 * time.Millisecond), Finished: true,
					},
				},
			},
		}}}, nil
	}
	router := &OpenAIRouter{Config: cfg, InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	if err := router.beginPrimaryInferenceDispatch(ctx.TraceContext, ctx, "internal-model"); err != nil {
		t.Fatal(err)
	}
	applyPrimaryDispatchOutcome(t, ctx, backendinvoker.AttemptResponseStarted)
	ctx.DispatchState.mu.Lock()
	dispatchID := ctx.DispatchState.primaryDispatchID
	ctx.DispatchState.mu.Unlock()
	markInferenceDispatchAttemptEvidenceRequired(ctx, dispatchID)
	usage := responseUsageMetrics{
		promptTokens: 7, promptTokensReported: true,
		completionTokens: 3, completionTokensReported: true,
		totalTokens: 10, totalTokensReported: true,
	}
	if err := router.completeAndSettlePrimaryInference(ctx, usage, 200); err != nil {
		t.Fatal(err)
	}
	event, err := usageledger.DecodeTerminalEvent(fake.settlements[0].Event)
	if err != nil {
		t.Fatal(err)
	}
	dispatch := event.Dispatches[0]
	if dispatch.InputTokens != "7" || dispatch.OutputTokens != "3" ||
		dispatch.UsageState != usageledger.UsageKnownActual || len(dispatch.Attempts) != 2 ||
		dispatch.Attempts[0].State != usageledger.UsageKnownZero ||
		dispatch.Attempts[1].State != usageledger.UsageKnownActual {
		t.Fatalf("authoritative retry dispatch = %+v", dispatch)
	}
}

func TestSettlementFailsClosedWhenRequiredAttemptEvidenceIsMissing(t *testing.T) {
	cfg := inferenceTestConfig(t)
	fake := &fakeInferenceAccess{}
	router := &OpenAIRouter{Config: cfg, InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	if err := router.beginPrimaryInferenceDispatch(ctx.TraceContext, ctx, "internal-model"); err != nil {
		t.Fatal(err)
	}
	applyPrimaryDispatchOutcome(t, ctx, backendinvoker.AttemptResponseStarted)
	ctx.DispatchState.mu.Lock()
	dispatchID := ctx.DispatchState.primaryDispatchID
	ctx.DispatchState.mu.Unlock()
	markInferenceDispatchAttemptEvidenceRequired(ctx, dispatchID)
	usage := responseUsageMetrics{
		promptTokens: 7, promptTokensReported: true,
		completionTokens: 3, completionTokensReported: true,
	}
	if err := router.completeAndSettlePrimaryInference(ctx, usage, 200); err != nil {
		t.Fatal(err)
	}
	event, err := usageledger.DecodeTerminalEvent(fake.settlements[0].Event)
	if err != nil {
		t.Fatal(err)
	}
	if event.EvidenceState != usageledger.EvidenceUnknown ||
		event.Dispatches[0].UsageState != usageledger.UsageUnknown ||
		event.Dispatches[0].UnknownReason != "attempt_evidence_missing" {
		t.Fatalf("missing attempt evidence event = %+v", event)
	}
}

func TestSettlementRebuildsAfterAttemptEvidenceCASConflict(t *testing.T) {
	cfg := inferenceTestConfig(t)
	fake := &fakeInferenceAccess{}
	settleCalls := 0
	fake.settle = func(accessruntime.SettlementRequest) (quotaruntime.FinalizationResult, error) {
		settleCalls++
		if settleCalls == 1 {
			return quotaruntime.FinalizationResult{}, quotaruntime.ErrEvidenceChanged
		}
		return quotaruntime.FinalizationResult{}, nil
	}
	router := &OpenAIRouter{Config: cfg, InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	if err := router.beginPrimaryInferenceDispatch(ctx.TraceContext, ctx, "internal-model"); err != nil {
		t.Fatal(err)
	}
	applyPrimaryDispatchOutcome(t, ctx, backendinvoker.AttemptResponseStarted)
	usage := responseUsageMetrics{
		promptTokens: 1, promptTokensReported: true,
		completionTokens: 1, completionTokensReported: true,
	}
	if err := router.completeAndSettlePrimaryInference(ctx, usage, 200); err != nil {
		t.Fatal(err)
	}
	if len(fake.evidenceReads) != 2 || len(fake.settlements) != 2 {
		t.Fatalf("evidence reads/settlements = %d/%d, want 2/2", len(fake.evidenceReads), len(fake.settlements))
	}
}

func TestProvenPreInferenceFailureSettlesKnownZero(t *testing.T) {
	cfg := inferenceTestConfig(t)
	fake := &fakeInferenceAccess{readEvidence: func(request accessruntime.AttemptEvidenceRequest) (accessruntime.AttemptEvidenceSnapshot, error) {
		dispatch := request.Dispatches[0]
		started := time.Now().UTC().Add(-time.Second).Truncate(time.Millisecond)
		return accessruntime.AttemptEvidenceSnapshot{Dispatches: []accessruntime.AttemptEvidenceObservation{{
			DispatchID: dispatch.DispatchID, Present: true,
			Evidence: quotaruntime.DispatchAttemptEvidence{
				DispatchID: dispatch.DispatchID, DispatchType: "primary", Ordinal: dispatch.Ordinal,
				DispatchPlanDigest: dispatch.DispatchPlanDigest, ModelID: dispatch.ModelID,
				ModelRevision: dispatch.ModelRevision, StartedAt: started,
				Deadline: started.Add(time.Minute), MaxAttempts: 1,
				Attempts: []quotaruntime.AttemptEvidence{{
					AttemptID: dispatch.DispatchID + ":1", AttemptNumber: 1,
					BackendID: uuid.NewString(), ProviderID: "test-provider",
					State: quotaruntime.AttemptEvidenceKnownZero, ErrorCode: "connect_refused",
					StartedAt: started, CompletedAt: started.Add(10 * time.Millisecond), Finished: true,
				}},
			},
		}}}, nil
	}}
	router := &OpenAIRouter{Config: cfg, InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	if err := router.beginPrimaryInferenceDispatch(ctx.TraceContext, ctx, "internal-model"); err != nil {
		t.Fatal(err)
	}
	applyPrimaryDispatchOutcome(t, ctx, backendinvoker.AttemptKnownZero)
	if err := router.settleUnknownInference(ctx, 502, "transport_error"); err != nil {
		t.Fatal(err)
	}
	event, err := usageledger.DecodeTerminalEvent(fake.settlements[0].Event)
	if err != nil {
		t.Fatal(err)
	}
	if event.EvidenceState != usageledger.EvidenceKnown ||
		event.Dispatches[0].UsageState != usageledger.UsageKnownZero ||
		event.Dispatches[0].Attempts[0].State != usageledger.UsageKnownZero {
		t.Fatalf("known-zero terminal event = %+v", event)
	}
}

func TestDisconnectSettlesUnknownAndDoesNotRetryFinalization(t *testing.T) {
	cfg := inferenceTestConfig(t)
	fake := &fakeInferenceAccess{}
	router := &OpenAIRouter{Config: cfg, InferenceAccess: fake}
	ctx := admittedInferenceTestContext("internal-model")
	if err := router.beginPrimaryInferenceDispatch(ctx.TraceContext, ctx, "internal-model"); err != nil {
		t.Fatal(err)
	}
	markPrimaryCapabilityIssued(ctx)
	if err := router.settleUnknownInference(ctx, 502, "client_disconnected"); err != nil {
		t.Fatal(err)
	}
	if err := router.settleUnknownInference(ctx, 502, "client_disconnected"); err != nil {
		t.Fatal(err)
	}
	if len(fake.settlements) != 1 {
		t.Fatalf("disconnect settlements = %d, want one", len(fake.settlements))
	}
	event, err := usageledger.DecodeTerminalEvent(fake.settlements[0].Event)
	if err != nil {
		t.Fatal(err)
	}
	if event.EvidenceState != usageledger.EvidenceUnknown || event.Dispatches[0].UsageState != usageledger.UsageUnknown {
		t.Fatalf("disconnect event = %+v", event)
	}
}

func applyPrimaryDispatchOutcome(t *testing.T, ctx *RequestContext, state backendinvoker.AttemptState) {
	t.Helper()
	markPrimaryCapabilityIssued(ctx)
	dispatchState := ctx.DispatchState
	dispatchState.mu.Lock()
	dispatch := dispatchState.dispatches[0]
	requestDigest := dispatchState.requestDigest
	dispatchState.mu.Unlock()
	ctx.InferenceAccess.mu.Lock()
	admission := *ctx.InferenceAccess.admission
	ctx.InferenceAccess.mu.Unlock()
	outcome := backendinvoker.DispatchOutcome{
		AdmissionID: admission.Tenant.AdmissionID, AdmissionDigest: admission.RequestDigest,
		RequestID: ctx.RequestID, RequestDigest: requestDigest,
		Attempted: []backendinvoker.DispatchOutcomeCandidate{{
			DispatchID: dispatch.id, DispatchType: dispatch.dispatchType,
			Ordinal: int(dispatch.ordinal), DispatchPlanDigest: dispatch.planDigest,
			ModelID: dispatch.modelID, ModelRevision: dispatch.modelRevision,
			Priority: dispatch.priority, State: state, AttemptCount: 1,
		}},
	}
	switch state {
	case backendinvoker.AttemptKnownZero:
		outcome.Attempted[0].FallbackTrigger = backendinvoker.FallbackUnavailable
	case backendinvoker.AttemptResponseStarted:
		outcome.SelectedDispatchID = dispatch.id
	}
	if err := applyDispatchOutcome(ctx, outcome); err != nil {
		t.Fatal(err)
	}
}

func markPrimaryCapabilityIssued(ctx *RequestContext) {
	ctx.DispatchState.mu.Lock()
	ctx.DispatchState.capabilityIssued = true
	ctx.DispatchState.requestDigest = strings.Repeat("a", 64)
	ctx.DispatchState.mu.Unlock()
}

func admittedInferenceTestContext(model string) *RequestContext {
	admissionID := uuid.NewString()
	tenant := inferenceTestTenant(admissionID)
	return &RequestContext{
		Headers:      map[string]string{":path": "/v1/chat/completions"},
		RequestID:    admissionID,
		RequestModel: model, StartTime: time.Now().UTC(), TraceContext: inferenceTestTraceContext(tenant),
		DispatchState: &requestDispatchState{requestID: admissionID},
		InferenceAccess: &inferenceRequestAccess{
			target: accessruntime.Target{
				ResourceType: accesscontrol.GrantResourceModel,
				ResourceID:   "mdl_internal_model",
				Permission:   accesscontrol.GrantPermissionInvoke,
			},
			admission: &accessruntime.Admission{
				Result: quotaruntime.AdmissionResult{Disposition: quotaruntime.AdmissionAllowed},
				Tenant: tenant,
				Target: accessruntime.Target{
					ResourceType: accesscontrol.GrantResourceModel,
					ResourceID:   "mdl_internal_model",
					Permission:   accesscontrol.GrantPermissionInvoke,
				},
				RequestDigest: strings.Repeat("d", 64), PreparedAt: time.Now().UTC(),
			},
		},
	}
}

func inferenceTestTraceContext(tenant accessruntime.TenantContext) context.Context {
	generation, err := generationForTenant(tenant)
	if err != nil {
		panic(err)
	}
	traceContext, err := routingcontext.WithGeneration(context.Background(), generation)
	if err != nil {
		panic(err)
	}
	return traceContext
}
