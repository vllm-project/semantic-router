package extproc

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

type fakeInferenceAccess struct {
	authentications []accessruntime.AuthenticationRequest
	authorizations  []accessruntime.AuthorizationRequest
	authenticate    func(accessruntime.AuthenticationRequest) (accessruntime.Authentication, error)
	authorize       func(accessruntime.AuthorizationRequest) (accessruntime.Authorization, error)
	catalogCalls    int
	catalog         accessruntime.CatalogDiscovery
	admissions      []accessruntime.AdmissionRequest
	heartbeatMu     sync.Mutex
	heartbeats      []accessruntime.Admission
	heartbeat       func(accessruntime.Admission) (quotaruntime.AdmissionHeartbeatResult, error)
	journal         []accessruntime.DispatchJournalRequest
	evidenceReads   []accessruntime.AttemptEvidenceRequest
	readEvidence    func(accessruntime.AttemptEvidenceRequest) (accessruntime.AttemptEvidenceSnapshot, error)
	settlements     []accessruntime.SettlementRequest
	settle          func(accessruntime.SettlementRequest) (quotaruntime.FinalizationResult, error)
}

func (f *fakeInferenceAccess) Authenticate(_ context.Context, request accessruntime.AuthenticationRequest) (accessruntime.Authentication, error) {
	f.authentications = append(f.authentications, request)
	if f.authenticate != nil {
		return f.authenticate(request)
	}
	return accessruntime.Authentication{
		Result: quotaruntime.AccessCheckResult{Disposition: quotaruntime.AdmissionAllowed},
		Tenant: inferenceTestTenant(""),
	}, nil
}

func (f *fakeInferenceAccess) Authorize(_ context.Context, request accessruntime.AuthorizationRequest) (accessruntime.Authorization, error) {
	f.authorizations = append(f.authorizations, request)
	if f.authorize != nil {
		return f.authorize(request)
	}
	return accessruntime.Authorization{
		Result: quotaruntime.AccessCheckResult{Disposition: quotaruntime.AdmissionAllowed},
		Tenant: inferenceTestTenant(""), Target: request.Target,
	}, nil
}

func (f *fakeInferenceAccess) Discover(context.Context, accessruntime.DiscoveryRequest) (accessruntime.Discovery, error) {
	panic("single-resource discovery must not be used by /v1/models")
}

func (f *fakeInferenceAccess) DiscoverCatalog(_ context.Context, _ accessruntime.CatalogDiscoveryRequest) (accessruntime.CatalogDiscovery, error) {
	f.catalogCalls++
	return f.catalog, nil
}

func (f *fakeInferenceAccess) Admit(_ context.Context, request accessruntime.AdmissionRequest) (accessruntime.Admission, error) {
	f.admissions = append(f.admissions, request)
	return accessruntime.Admission{
		Result: quotaruntime.AdmissionResult{Disposition: quotaruntime.AdmissionAllowed},
		Tenant: inferenceTestTenant(request.AdmissionID), Target: request.Target,
		RequestDigest: request.RequestDigest, PreparedAt: time.Now().UTC(),
	}, nil
}

func (f *fakeInferenceAccess) Heartbeat(_ context.Context, admission accessruntime.Admission) (quotaruntime.AdmissionHeartbeatResult, error) {
	f.heartbeatMu.Lock()
	f.heartbeats = append(f.heartbeats, admission)
	f.heartbeatMu.Unlock()
	if f.heartbeat != nil {
		return f.heartbeat(admission)
	}
	return quotaruntime.AdmissionHeartbeatResult{}, nil
}

func (f *fakeInferenceAccess) heartbeatCount() int {
	f.heartbeatMu.Lock()
	defer f.heartbeatMu.Unlock()
	return len(f.heartbeats)
}

func (f *fakeInferenceAccess) JournalDispatch(_ context.Context, request accessruntime.DispatchJournalRequest) (quotaruntime.MutationResult, error) {
	f.journal = append(f.journal, request)
	return quotaruntime.MutationResult{}, nil
}

func (f *fakeInferenceAccess) ReadAttemptEvidence(_ context.Context, request accessruntime.AttemptEvidenceRequest) (accessruntime.AttemptEvidenceSnapshot, error) {
	f.evidenceReads = append(f.evidenceReads, request)
	if f.readEvidence != nil {
		return f.readEvidence(request)
	}
	observations := make([]accessruntime.AttemptEvidenceObservation, 0, len(request.Dispatches))
	for _, dispatch := range request.Dispatches {
		observations = append(observations, accessruntime.AttemptEvidenceObservation{DispatchID: dispatch.DispatchID})
	}
	return accessruntime.AttemptEvidenceSnapshot{Dispatches: observations}, nil
}

func (f *fakeInferenceAccess) Settle(_ context.Context, request accessruntime.SettlementRequest) (quotaruntime.FinalizationResult, error) {
	f.settlements = append(f.settlements, request)
	if f.settle != nil {
		return f.settle(request)
	}
	return quotaruntime.FinalizationResult{}, nil
}

func responseStartedEvidence(request accessruntime.AttemptEvidenceRequest) (accessruntime.AttemptEvidenceSnapshot, error) {
	started := time.Now().UTC().Add(-time.Second).Truncate(time.Millisecond)
	observations := make([]accessruntime.AttemptEvidenceObservation, 0, len(request.Dispatches))
	for _, dispatch := range request.Dispatches {
		observations = append(observations, accessruntime.AttemptEvidenceObservation{
			DispatchID: dispatch.DispatchID, Present: true,
			Evidence: quotaruntime.DispatchAttemptEvidence{
				DispatchID: dispatch.DispatchID, DispatchType: inferenceDispatchType(int(dispatch.Ordinal)),
				Ordinal: dispatch.Ordinal, DispatchPlanDigest: dispatch.DispatchPlanDigest,
				ModelID: dispatch.ModelID, ModelRevision: dispatch.ModelRevision,
				StartedAt: started, Deadline: started.Add(time.Minute), MaxAttempts: 1,
				Attempts: []quotaruntime.AttemptEvidence{{
					AttemptID: dispatch.DispatchID + ":1", AttemptNumber: 1,
					BackendID: uuid.NewString(), ProviderID: "test-provider",
					State: quotaruntime.AttemptEvidenceResponseStarted, StatusCode: 200,
					StartedAt: started, CompletedAt: started.Add(10 * time.Millisecond), Finished: true,
				}},
			},
		})
	}
	return accessruntime.AttemptEvidenceSnapshot{Dispatches: observations}, nil
}

func inferenceTestTenant(admissionID string) accessruntime.TenantContext {
	return accessruntime.TenantContext{
		AdmissionID: admissionID, NamespaceID: uuid.MustParse("00000000-0000-0000-0000-000000000101").String(),
		QuotaPartition: "partition", APIKeyID: uuid.MustParse("00000000-0000-0000-0000-000000000102").String(),
		UserID:         uuid.MustParse("00000000-0000-0000-0000-000000000103").String(),
		PolicyRevision: 1, PolicyDigest: "digest", BillingCurrency: "USD",
		PublicationID: "publication", RuntimeEpoch: 1, RoutingRevision: 1,
		RoutingDigest: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
	}
}

func TestNativeAccessInferenceSessionFailsClosedAndNeverRetainsCredential(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{Access: config.AccessServiceConfig{Enabled: true}}}
	missing := &RequestContext{Headers: map[string]string{":path": "/v1/chat/completions"}}
	if status := inferenceAccessDisposition(router.bindInferenceAuthentication(missing)); status != 401 {
		t.Fatalf("missing credential status = %d, want 401", status)
	}

	authentication := accessruntime.Authentication{Tenant: inferenceTestTenant("")}
	ctx := &RequestContext{
		Headers:      map[string]string{"Authorization": "must-not-survive"},
		TraceContext: withInferenceAuthentication(context.Background(), authentication),
	}
	if response := router.bindInferenceAuthentication(ctx); response != nil {
		t.Fatalf("valid session rejected: %+v", response)
	}
	if ctx.InferenceAccess == nil || headerValueCI(ctx, "authorization") != "" {
		t.Fatalf("session was not bound or authorization survived: %+v", ctx)
	}
	mutation := buildIdentityEncodingRequestMutation(true)
	if !slices.Contains(mutation.RemoveHeaders, "authorization") {
		t.Fatalf("authenticated upstream mutation does not remove authorization: %v", mutation.RemoveHeaders)
	}
}

func TestNativeAccessModelsRejectsAnonymousRequestsBeforeCatalogDiscovery(t *testing.T) {
	fake := &fakeInferenceAccess{}
	router := &OpenAIRouter{
		Config:          &config.RouterConfig{Access: config.AccessServiceConfig{Enabled: true}},
		InferenceAccess: fake,
	}
	ctx := &RequestContext{Headers: make(map[string]string)}

	response, err := router.handleRequestHeaders(newRequestHeaders("GET", "/v1/models"), ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	if status := inferenceAccessDisposition(response); status != 401 {
		t.Fatalf("anonymous native-access compositionls status = %d, want 401", status)
	}
	if fake.catalogCalls != 0 {
		t.Fatalf("anonymous native-access compositionls catalog calls = %d, want 0", fake.catalogCalls)
	}
}

func TestFileAuthorityDoesNotInterpretCallerAuthorization(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{Headers: map[string]string{"Authorization": "Bearer provider-owned-key"}}
	if response := router.bindInferenceAuthentication(ctx); response != nil {
		t.Fatalf("file-authority authorization was interpreted: %+v", response)
	}
	if got := headerValueCI(ctx, "authorization"); got != "Bearer provider-owned-key" {
		t.Fatalf("file-authority authorization = %q", got)
	}
	if slices.Contains(buildIdentityEncodingRequestMutation(false).RemoveHeaders, "authorization") {
		t.Fatal("file-authority mutation removes caller authorization")
	}
}

func TestEntrypointGrantAuthorizesPublishedInternalModels(t *testing.T) {
	cfg := inferenceTestConfig(t)
	fake := &fakeInferenceAccess{}
	fake.authorize = func(request accessruntime.AuthorizationRequest) (accessruntime.Authorization, error) {
		result := quotaruntime.AccessCheckResult{Disposition: quotaruntime.AdmissionForbidden}
		if request.Target.ResourceType == accesscontrol.GrantResourceEntrypoint {
			result.Disposition = quotaruntime.AdmissionAllowed
		}
		tenant := inferenceTestTenant("")
		tenant.RoutingClaims = map[string]routingsnapshot.ClaimValue{"tier": {Kind: "string", String: "premium"}}
		return accessruntime.Authorization{Result: result, Tenant: tenant, Target: request.Target}, nil
	}
	router := &OpenAIRouter{Config: cfg, InferenceAccess: fake}
	tenant := inferenceTestTenant("")
	ctx := &RequestContext{
		Headers:      map[string]string{":path": "/v1/chat/completions"},
		TraceContext: inferenceTestTraceContext(tenant),
		RequestID:    "entrypoint-test",
		RequestModel: "virtual/premium", InferenceAccess: &inferenceRequestAccess{
			tenant: tenant,
		},
		DispatchState: &requestDispatchState{requestID: "entrypoint-test"},
	}
	if response := router.authorizeInferenceTarget(context.Background(), ctx, ctx.RequestModel); response != nil {
		t.Fatalf("entrypoint authorization failed: status=%d", inferenceAccessDisposition(response))
	}
	recipe := ctx.Routing.SelectedRecipe()
	if recipe == nil || len(recipe.Profile.Decisions) != 1 {
		t.Fatal("entrypoint did not resolve its published recipe")
	}
	ctx.VSRSelectedDecision = &recipe.Profile.Decisions[0]
	if response := router.admitInferenceRequest(context.Background(), ctx, "internal-model"); response != nil {
		t.Fatalf("entrypoint admission required a direct model grant: status=%d", inferenceAccessDisposition(response))
	}
	if len(fake.authorizations) != 1 || fake.authorizations[0].Target.ResourceType != accesscontrol.GrantResourceEntrypoint {
		t.Fatalf("unexpected model access checks: %+v", fake.authorizations)
	}
	if len(fake.admissions) != 1 || fake.admissions[0].Target.ResourceType != accesscontrol.GrantResourceEntrypoint {
		t.Fatalf("unexpected admission: %+v", fake.admissions)
	}
	if err := router.settleNoBackendInference(ctx, 500, "pre_dispatch_failure"); err != nil {
		t.Fatal(err)
	}
	if len(fake.settlements) != 1 {
		t.Fatalf("pre-dispatch settlements = %d, want one", len(fake.settlements))
	}
	event, err := usageledger.DecodeTerminalEvent(fake.settlements[0].Event)
	if err != nil {
		t.Fatal(err)
	}
	if len(event.Dispatches) != 1 || event.Dispatches[0].ModelName != "internal-model" ||
		event.Dispatches[0].UsageState != usageledger.UsageKnownZero {
		t.Fatalf("pre-dispatch event = %+v", event.Dispatches)
	}
}

func TestDirectModelDenialIsNondisclosing(t *testing.T) {
	cfg := inferenceTestConfig(t)
	fake := &fakeInferenceAccess{authorize: func(request accessruntime.AuthorizationRequest) (accessruntime.Authorization, error) {
		return accessruntime.Authorization{
			Result: quotaruntime.AccessCheckResult{Disposition: quotaruntime.AdmissionForbidden},
			Target: request.Target,
		}, nil
	}}
	router := &OpenAIRouter{Config: cfg, InferenceAccess: fake}
	ctx := &RequestContext{RequestModel: "internal-model", InferenceAccess: &inferenceRequestAccess{
		tenant: inferenceTestTenant(""),
	}}
	denied := router.authorizeInferenceTarget(context.Background(), ctx, ctx.RequestModel)
	if status := inferenceAccessDisposition(denied); status != 404 {
		t.Fatalf("direct model denial status = %d, want 404", status)
	}

	missingContext := &RequestContext{
		RequestModel:    "model-that-does-not-exist",
		InferenceAccess: &inferenceRequestAccess{tenant: inferenceTestTenant("")},
	}
	missing := router.authorizeInferenceTarget(context.Background(), missingContext, missingContext.RequestModel)
	if status := inferenceAccessDisposition(missing); status != 404 {
		t.Fatalf("missing model status = %d, want 404", status)
	}
	if string(denied.GetImmediateResponse().Body) != string(missing.GetImmediateResponse().Body) {
		t.Fatalf("forbidden response disclosed model existence: denied=%q missing=%q",
			denied.GetImmediateResponse().Body, missing.GetImmediateResponse().Body)
	}
}

func TestNativeAccessModelsUsesOneCatalogSnapshotAndRoutingClaims(t *testing.T) {
	cfg := inferenceTestConfig(t)
	tenant := inferenceTestTenant("")
	tenant.RoutingClaims = map[string]routingsnapshot.ClaimValue{
		"tier": {Kind: "string", String: "free"},
	}
	fake := &fakeInferenceAccess{catalog: accessruntime.CatalogDiscovery{
		Result: quotaruntime.AccessCheckResult{Disposition: quotaruntime.AdmissionAllowed},
		Tenant: tenant,
		Resources: map[accesscontrol.GrantResourceType][]string{
			accesscontrol.GrantResourceEntrypoint: {cfg.Entrypoints[0].ID},
			accesscontrol.GrantResourceModel:      {cfg.ModelConfig["internal-model"].ResourceID},
		},
	}}
	router := &OpenAIRouter{Config: cfg, InferenceAccess: fake}
	ctx := &RequestContext{
		Headers:         map[string]string{":path": "/v1/models?for_path=%2Fv1%2Fchat%2Fcompletions"},
		InferenceAccess: &inferenceRequestAccess{tenant: inferenceTestTenant("")},
	}
	response, handleAuthorizedModelsRequestErr := router.handleAuthorizedModelsRequest(context.Background(), ctx)
	if handleAuthorizedModelsRequestErr != nil {
		t.Fatal(handleAuthorizedModelsRequestErr)
	}
	if fake.catalogCalls != 1 {
		t.Fatalf("catalog calls=%d", fake.catalogCalls)
	}
	var catalog publicmodels.OpenAIModelList
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &catalog); err != nil {
		t.Fatal(err)
	}
	if len(catalog.Data) != 1 || catalog.Data[0].ID != "internal-model" {
		t.Fatalf("claim-ineligible entrypoint leaked into catalog: %+v", catalog.Data)
	}

	fake.catalog.Tenant.RoutingClaims["tier"] = routingsnapshot.ClaimValue{Kind: "string", String: "premium"}
	response, handleAuthorizedModelsRequestErr = router.handleAuthorizedModelsRequest(context.Background(), ctx)
	if handleAuthorizedModelsRequestErr != nil {
		t.Fatal(handleAuthorizedModelsRequestErr)
	}
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &catalog); err != nil {
		t.Fatal(err)
	}
	ids := make([]string, 0, len(catalog.Data))
	for _, model := range catalog.Data {
		ids = append(ids, model.ID)
	}
	if !slices.Contains(ids, "virtual/premium") || !slices.Contains(ids, "internal-model") {
		t.Fatalf("eligible catalog = %v", ids)
	}
}

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
				ResourceID:   "00000000-0000-0000-0000-000000000201",
				Permission:   accesscontrol.GrantPermissionInvoke,
			},
			admission: &accessruntime.Admission{
				Result: quotaruntime.AdmissionResult{Disposition: quotaruntime.AdmissionAllowed},
				Tenant: tenant,
				Target: accessruntime.Target{
					ResourceType: accesscontrol.GrantResourceModel,
					ResourceID:   "00000000-0000-0000-0000-000000000201",
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

func inferenceTestConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	const (
		recipeID     = "recipe-test"
		decisionID   = "decision-test"
		modelID      = "00000000-0000-0000-0000-000000000201"
		entrypointID = "00000000-0000-0000-0000-000000000202"
	)
	one := "1"
	cfg := &config.RouterConfig{
		AccessStore:   &config.AccessStoreConfig{Type: config.AccessStoreTypePostgres},
		Access:        config.AccessServiceConfig{Enabled: true},
		RouterOptions: config.RouterOptions{IncludeConfigModelsInList: true},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"internal-model": {
				ResourceID: modelID, ResourceRevision: 1,
				RuntimePricing: config.ModelRuntimePricing{
					InputCostPerMillionTokens: &one, OutputCostPerMillionTokens: &one,
					CacheReadCostPerMillionTokens: &one, CacheWriteCostPerMillionTokens: &one,
				},
			},
		}},
		Recipes: []config.RoutingRecipe{{
			ID: recipeID, Revision: 1, Name: "test-recipe",
			Profile: config.RoutingProfile{Decisions: []config.Decision{{
				ID: decisionID, Name: "test-decision",
				ModelRefs: []config.ModelRef{{Model: "internal-model"}},
			}}},
		}},
		Entrypoints: []config.EntrypointMapping{{
			ID: entrypointID, Revision: 1, Name: "premium", ModelNames: []string{"virtual/premium"},
			Rules: []config.EntrypointRule{{
				ID: "rule-premium", Name: "premium",
				Matches: []config.EntrypointMatch{{Claim: &config.EntrypointClaimMatch{
					Name: "tier", Value: config.EntrypointClaimValue{Kind: "string", String: "premium"},
				}}},
				Action: config.EntrypointRuleAction{
					RecipeID: recipeID, RecipeRevision: 1, Recipe: "test-recipe",
					Assignments: map[string]config.RoutingAssignmentSet{
						decisionID: {Models: []config.RoutingModelAssignment{{ModelID: modelID, ModelRevision: 1, ModelName: "internal-model", Weight: "1"}}},
					},
				},
			}},
		}},
	}
	if err := cfg.PrepareEntrypointRecipes(); err != nil {
		t.Fatal(err)
	}
	return cfg
}
