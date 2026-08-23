package extproc

import (
	"context"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageaccounting"
)

func TestBackendDispatchOutcomeSelectsOnlyAuthenticatedAttemptedPrefix(t *testing.T) {
	ctx := dispatchOutcomeTestContext(t)
	outcome := dispatchOutcomeForTest(ctx, 2)
	outcome.Attempted[0].State = backendinvoker.AttemptKnownZero
	outcome.Attempted[0].FallbackTrigger = backendinvoker.FallbackUnavailable
	outcome.Attempted[1].State = backendinvoker.AttemptResponseStarted
	outcome.SelectedDispatchID = outcome.Attempted[1].DispatchID
	runtime := &capturingDispatchCapabilityRuntime{outcome: outcome}
	router := &OpenAIRouter{DispatchCapabilities: runtime}
	request := dispatchOutcomeResponse("signed-outcome")

	mutation, err := router.consumeBackendDispatchOutcome(request, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.verifiedToken != "signed-outcome" ||
		runtime.verificationRequest.RequestID != ctx.RequestID {
		t.Fatalf("verification = token %q, request %+v", runtime.verifiedToken, runtime.verificationRequest)
	}
	if !containsHeaderName(mutation.GetRemoveHeaders(), backendinvoker.DispatchOutcomeHeader) {
		t.Fatalf("outcome removal mutation = %v", mutation.GetRemoveHeaders())
	}
	if headers := responseHeaderMap(request).GetHeaders(); len(headers) != 1 || headers[0].GetKey() != "content-type" {
		t.Fatalf("private outcome survived response headers: %+v", headers)
	}

	state := ctx.ManagedDispatch
	state.mu.Lock()
	defer state.mu.Unlock()
	if !state.outcomeConsumed || state.selectedDispatchID != "dispatch-1" ||
		ctx.VSRSelectedModel != "model-1" {
		t.Fatalf("selected state = consumed %v, dispatch %q, model %q", state.outcomeConsumed, state.selectedDispatchID, ctx.VSRSelectedModel)
	}
	if !state.dispatches[0].attempted || !state.dispatches[0].settlementEligible ||
		state.dispatches[0].state != usageaccounting.EvidenceKnownZero {
		t.Fatalf("first attempted dispatch = %+v", state.dispatches[0])
	}
	if !state.dispatches[1].attempted || !state.dispatches[1].settlementEligible ||
		state.dispatches[1].state != usageaccounting.EvidenceUnknown {
		t.Fatalf("selected attempted dispatch = %+v", state.dispatches[1])
	}
	if state.dispatches[2].attempted || state.dispatches[2].settlementEligible {
		t.Fatalf("unattempted planned dispatch became usage: %+v", state.dispatches[2])
	}
}

func TestBackendDispatchOutcomeRejectsMismatchedAttemptEvidence(t *testing.T) {
	tests := map[string]func(*backendinvoker.DispatchOutcome){
		"reordered candidate": func(outcome *backendinvoker.DispatchOutcome) {
			outcome.Attempted[0], outcome.Attempted[1] = outcome.Attempted[1], outcome.Attempted[0]
		},
		"request digest": func(outcome *backendinvoker.DispatchOutcome) {
			outcome.RequestDigest = strings.Repeat("f", 64)
		},
		"admission": func(outcome *backendinvoker.DispatchOutcome) {
			outcome.AdmissionID = "different-admission"
		},
		"caller-selected candidate": func(outcome *backendinvoker.DispatchOutcome) {
			outcome.SelectedDispatchID = outcome.Attempted[0].DispatchID
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := dispatchOutcomeTestContext(t)
			outcome := dispatchOutcomeForTest(ctx, 2)
			outcome.Attempted[0].State = backendinvoker.AttemptKnownZero
			outcome.Attempted[0].FallbackTrigger = backendinvoker.FallbackUnavailable
			outcome.Attempted[1].State = backendinvoker.AttemptResponseStarted
			outcome.SelectedDispatchID = outcome.Attempted[1].DispatchID
			mutate(&outcome)
			if err := applyDispatchOutcome(ctx, outcome); err == nil {
				t.Fatal("mismatched dispatch outcome was accepted")
			}
			ctx.ManagedDispatch.mu.Lock()
			consumed := ctx.ManagedDispatch.outcomeConsumed
			ctx.ManagedDispatch.mu.Unlock()
			if consumed {
				t.Fatal("rejected outcome changed request settlement state")
			}
		})
	}
}

func TestBackendDispatchOutcomeIsRequiredExactlyOnce(t *testing.T) {
	for name, request := range map[string]*ext_proc.ProcessingRequest_ResponseHeaders{
		"missing":   dispatchOutcomeResponse(),
		"duplicate": dispatchOutcomeResponse("one", "two"),
		"empty":     dispatchOutcomeResponse(""),
	} {
		t.Run(name, func(t *testing.T) {
			ctx := dispatchOutcomeTestContext(t)
			router := &OpenAIRouter{DispatchCapabilities: &capturingDispatchCapabilityRuntime{}}
			mutation, err := router.consumeBackendDispatchOutcome(request, ctx)
			if err == nil {
				t.Fatal("invalid outcome header was accepted")
			}
			if !containsHeaderName(mutation.GetRemoveHeaders(), backendinvoker.DispatchOutcomeHeader) {
				t.Fatalf("private outcome removal = %v", mutation.GetRemoveHeaders())
			}
			for _, header := range responseHeaderMap(request).GetHeaders() {
				if strings.EqualFold(header.GetKey(), backendinvoker.DispatchOutcomeHeader) {
					t.Fatal("invalid private outcome remained in provider headers")
				}
			}
		})
	}
}

func TestRoutingOnlyManagedRequestUsesPublishedFallbackAssignment(t *testing.T) {
	decisionID := "decision"
	rule := config.EntrypointRule{
		ID: "rule", Name: "default",
		Action: config.EntrypointRuleAction{Assignments: map[string]config.RoutingAssignmentSet{
			decisionID: {
				Models: []config.RoutingModelAssignment{
					{ModelName: "active-a", Priority: 0},
					{ModelName: "active-b", Priority: 0},
					{ModelName: "fallback-z", Priority: 1},
					{ModelName: "fallback-a", Priority: 1},
					{ModelName: "last", Priority: 2},
				},
				Fallback: &config.RoutingFallbackPolicy{
					Strategy: "priority", On: []string{"unavailable", "timeout"},
				},
			},
		}},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{Entrypoints: []config.EntrypointMapping{{
		ID: "entrypoint", Name: "virtual", ModelNames: []string{"virtual/model"},
		Rules: []config.EntrypointRule{rule},
	}}}}
	ctx := &RequestContext{
		RequestModel: "virtual/model", Headers: map[string]string{":path": "/v1/chat/completions"},
		VSRSelectedDecision: &config.Decision{ID: decisionID},
	}
	candidates, fallback, err := router.primaryDispatchCandidates(ctx, "active-b")
	if err != nil {
		t.Fatal(err)
	}
	want := []primaryDispatchCandidate{
		{model: "active-b", priority: 0},
		{model: "fallback-a", priority: 1},
		{model: "fallback-z", priority: 1},
		{model: "last", priority: 2},
	}
	if len(candidates) != len(want) {
		t.Fatalf("candidates = %+v, want %+v", candidates, want)
	}
	for index := range want {
		if candidates[index] != want[index] {
			t.Fatalf("candidate %d = %+v, want %+v", index, candidates[index], want[index])
		}
	}
	if len(fallback.On) != 2 || fallback.On[0] != backendinvoker.FallbackUnavailable ||
		fallback.On[1] != backendinvoker.FallbackTimeout {
		t.Fatalf("fallback policy = %+v", fallback)
	}
	if _, _, err := router.primaryDispatchCandidates(ctx, "not-assigned"); err == nil {
		t.Fatal("selected Model outside the active tier was accepted")
	}
}

func dispatchOutcomeTestContext(t *testing.T) *RequestContext {
	t.Helper()
	trace, err := routingcontext.WithGeneration(context.Background(), routingcontext.Generation{
		NamespaceID: "namespace", QuotaPartition: "partition", PublicationID: "publication",
		RuntimeEpoch: 2, SnapshotRevision: 7, RoutingDigest: strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	requestDigest := strings.Repeat("c", 64)
	admissionDigest := strings.Repeat("b", 64)
	dispatches := make([]*inferenceDispatch, 0, 3)
	for index, digestCharacter := range []string{"d", "e", "f"} {
		dispatches = append(dispatches, &inferenceDispatch{
			id: "dispatch-" + string(rune('0'+index)), ordinal: uint32(index),
			model: "model-" + string(rune('0'+index)), modelID: "model-id-" + string(rune('0'+index)),
			modelRevision: int64(index + 1), priority: index,
			planDigest: strings.Repeat(digestCharacter, 64), planned: true,
			dispatchType: "primary",
		})
	}
	return &RequestContext{
		RequestID: "request", TraceContext: trace, VSRSelectedModel: "model-0",
		InferenceAccess: &inferenceRequestAccess{admission: &accessruntime.Admission{
			Tenant:        accessruntime.TenantContext{AdmissionID: "admission"},
			RequestDigest: admissionDigest,
		}},
		ManagedDispatch: &managedRequestDispatch{
			requestID: "request", dispatches: dispatches, primaryDispatchID: "dispatch-0",
			primaryCandidateCount: len(dispatches), capabilityIssued: true,
			requestDigest: requestDigest,
		},
	}
}

func dispatchOutcomeForTest(ctx *RequestContext, attempted int) backendinvoker.DispatchOutcome {
	ctx.ManagedDispatch.mu.Lock()
	defer ctx.ManagedDispatch.mu.Unlock()
	result := backendinvoker.DispatchOutcome{
		AdmissionID:     ctx.InferenceAccess.admission.Tenant.AdmissionID,
		AdmissionDigest: ctx.InferenceAccess.admission.RequestDigest,
		RequestID:       ctx.RequestID, RequestDigest: ctx.ManagedDispatch.requestDigest,
		Attempted: make([]backendinvoker.DispatchOutcomeCandidate, 0, attempted),
	}
	for _, dispatch := range ctx.ManagedDispatch.dispatches[:attempted] {
		result.Attempted = append(result.Attempted, backendinvoker.DispatchOutcomeCandidate{
			DispatchID: dispatch.id, DispatchType: dispatch.dispatchType,
			Ordinal: int(dispatch.ordinal), DispatchPlanDigest: dispatch.planDigest,
			ModelID: dispatch.modelID, ModelRevision: dispatch.modelRevision,
			Priority: dispatch.priority, State: backendinvoker.AttemptKnownZero,
			FallbackTrigger: backendinvoker.FallbackUnavailable, AttemptCount: 1,
		})
	}
	return result
}

func dispatchOutcomeResponse(values ...string) *ext_proc.ProcessingRequest_ResponseHeaders {
	headers := []*core.HeaderValue{{Key: "content-type", Value: "application/json"}}
	for _, value := range values {
		headers = append(headers, &core.HeaderValue{
			Key: backendinvoker.DispatchOutcomeHeader, Value: value,
		})
	}
	return &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: headers},
	}}
}

func containsHeaderName(values []string, expected string) bool {
	for _, value := range values {
		if strings.EqualFold(value, expected) {
			return true
		}
	}
	return false
}
