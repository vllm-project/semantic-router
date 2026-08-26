package accessruntime

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type fakeDelegationBarriers struct {
	state  managementauth.DelegationBarrierState
	err    error
	checks []managementauth.DelegationBarrierCheck
}

func (barriers *fakeDelegationBarriers) CheckDelegation(
	_ context.Context,
	check managementauth.DelegationBarrierCheck,
) (managementauth.DelegationBarrierState, error) {
	barriers.checks = append(barriers.checks, check)
	return barriers.state, barriers.err
}

type fakeReader struct {
	location    CredentialLocation
	credential  accessprojection.CredentialProjection
	active      ActivePolicy
	policy      accessprojection.Projection
	err         error
	activeReads int
	policyReads int
}

func (f *fakeReader) LocateCredential(context.Context, accesscredential.Kind, string) (CredentialLocation, error) {
	return f.location, f.err
}

func (f *fakeReader) ReadCredential(context.Context, CredentialLocation, accesscredential.Kind, string) (accessprojection.CredentialProjection, error) {
	return f.credential, f.err
}

func (f *fakeReader) ReadActivePolicy(context.Context, CredentialLocation, string) (ActivePolicy, error) {
	f.activeReads++
	return f.active, f.err
}

func (f *fakeReader) ReadPolicy(context.Context, CredentialLocation, ActivePolicy) (accessprojection.Projection, error) {
	f.policyReads++
	return f.policy, f.err
}

type fakeEngine struct {
	request             *quotaruntime.AdmissionRequest
	heartbeatRequest    *quotaruntime.AdmissionHeartbeatRequest
	heartbeatResult     quotaruntime.AdmissionHeartbeatResult
	accessRequest       *quotaruntime.AccessCheckRequest
	dispatchRequest     *quotaruntime.DispatchJournalRequest
	evidenceRequests    []quotaruntime.ReadAttemptEvidenceRequest
	finalizationRequest *quotaruntime.FinalizationRequest
	result              quotaruntime.AdmissionResult
	err                 error
	onCheck             func()
	check               func(quotaruntime.AccessCheckRequest) (quotaruntime.AccessCheckResult, error)
	readEvidence        func(quotaruntime.ReadAttemptEvidenceRequest) (quotaruntime.ReadAttemptEvidenceResult, error)
}

func (f *fakeEngine) CheckAccess(_ context.Context, request quotaruntime.AccessCheckRequest) (quotaruntime.AccessCheckResult, error) {
	f.accessRequest = &request
	if f.onCheck != nil {
		f.onCheck()
	}
	if f.check != nil {
		return f.check(request)
	}
	return quotaruntime.AccessCheckResult{Disposition: f.result.Disposition, Reason: f.result.BlockingReason}, f.err
}

func (f *fakeEngine) Admit(_ context.Context, request quotaruntime.AdmissionRequest) (quotaruntime.AdmissionResult, error) {
	f.request = &request
	return f.result, f.err
}

func (f *fakeEngine) Heartbeat(_ context.Context, request quotaruntime.AdmissionHeartbeatRequest) (quotaruntime.AdmissionHeartbeatResult, error) {
	f.heartbeatRequest = &request
	return f.heartbeatResult, f.err
}

func (f *fakeEngine) JournalDispatch(_ context.Context, request quotaruntime.DispatchJournalRequest) (quotaruntime.MutationResult, error) {
	f.dispatchRequest = &request
	return quotaruntime.MutationResult{}, f.err
}

func (f *fakeEngine) ReadAttemptEvidence(_ context.Context, request quotaruntime.ReadAttemptEvidenceRequest) (quotaruntime.ReadAttemptEvidenceResult, error) {
	f.evidenceRequests = append(f.evidenceRequests, request)
	if f.readEvidence != nil {
		return f.readEvidence(request)
	}
	return quotaruntime.ReadAttemptEvidenceResult{}, f.err
}

func (f *fakeEngine) Finalize(_ context.Context, request quotaruntime.FinalizationRequest) (quotaruntime.FinalizationResult, error) {
	f.finalizationRequest = &request
	return quotaruntime.FinalizationResult{}, f.err
}

func TestAdmitAuthenticatesAuthorizesAndPinsAtomicPlan(t *testing.T) {
	runtime, issued, reader, engine := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	if engine.accessRequest == nil || countPreconditionField(engine.accessRequest.Preconditions, "secret_hmac") != 1 {
		t.Fatalf("authentication did not atomically pin the verified credential: %+v", engine.accessRequest)
	}
	admission, err := runtime.Admit(context.Background(), AdmissionRequest{
		Session:     authentication.Session,
		Target:      Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-chat", Permission: accesscontrol.GrantPermissionInvoke},
		AdmissionID: "admission-1", RequestDigest: "request-digest", LeaseDuration: time.Minute,
	})
	if err != nil {
		t.Fatalf("Admit() error = %v", err)
	}
	if !admission.Result.Allowed() || admission.Tenant.AdmissionID == "" {
		t.Fatalf("Admit() = %+v", admission)
	}
	if admission.Tenant.APIKeyID != reader.policy.KeyID || admission.Tenant.PolicyDigest != reader.policy.Digest {
		t.Fatalf("tenant context = %+v", admission.Tenant)
	}
	if engine.request == nil || len(engine.request.Preconditions) < 10 {
		t.Fatalf("atomic request did not pin access state: %+v", engine.request)
	}
	if len(engine.request.Rules) != 1 || engine.request.Rules[0].BindingID != "rate-binding-user" {
		t.Fatalf("quota rules = %+v", engine.request.Rules)
	}
	if countPreconditionField(engine.request.Preconditions, "secret_hmac") != 0 {
		t.Fatalf("session retained a credential verifier: %+v", engine.request.Preconditions)
	}
}

func TestDelegationPinsAuthorityAndRechecksSharedLifecycleBarrier(t *testing.T) {
	runtime, _, reader, engine := testRuntime(t)
	issued, err := runtime.peppers[accesscredential.KindDelegation].Issue(
		accesscredential.KindDelegation,
		"delegation001",
	)
	if err != nil {
		t.Fatal(err)
	}
	expires := time.Now().UTC().Add(time.Hour)
	reader.credential = accessprojection.CredentialProjection{
		Kind: string(accesscredential.KindDelegation), KID: issued.Digest.PublicID, KeyID: reader.policy.KeyID,
		SecretHMAC: issued.Digest.HMAC, PepperVersion: issued.Digest.PepperVersion,
		Status: string(accesscontrol.CredentialStatusActive), NotBefore: time.Now().UTC().Add(-time.Minute), ExpiresAt: &expires,
		ManagementSessionID: "management-session-1", PrincipalID: "principal-1",
		DelegationEpoch: reader.policy.DelegationEpoch, UserID: reader.policy.UserID,
		TeamID: reader.policy.TeamID, Audience: "vllm-sr-inference",
	}
	barriers := runtime.delegationBarriers.(*fakeDelegationBarriers)
	authentication, err := runtime.Authenticate(context.Background(), AuthenticationRequest{Credential: issued.Plaintext})
	if err != nil || !authentication.Result.Allowed() {
		t.Fatalf("delegated Authenticate() = %+v, %v", authentication, err)
	}
	if len(barriers.checks) != 1 || barriers.checks[0].SessionID != reader.credential.ManagementSessionID ||
		barriers.checks[0].PrincipalID != reader.credential.PrincipalID {
		t.Fatalf("delegation barrier checks = %+v", barriers.checks)
	}
	for field, expected := range map[string]int{
		"management_session_id": 1, "principal_id": 1, "delegation_epoch": 2,
		"user_id": 1, "team_id": 0, "audience": 1,
	} {
		if countPreconditionField(engine.accessRequest.Preconditions, field) != expected {
			t.Fatalf("delegated authority field %q not pinned %d times: %+v", field, expected, engine.accessRequest.Preconditions)
		}
	}
	for index, precondition := range engine.accessRequest.Preconditions {
		if err := precondition.Validate(); err != nil {
			t.Fatalf("user-owned delegation precondition %d is invalid: %+v: %v", index, precondition, err)
		}
	}
	barriers.state.SessionDenied = true
	admission, err := runtime.Admit(context.Background(), AdmissionRequest{
		Session:     authentication.Session,
		Target:      Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-chat", Permission: accesscontrol.GrantPermissionInvoke},
		AdmissionID: "delegated-admission", RequestDigest: "delegated-request", LeaseDuration: time.Minute,
	})
	if err != nil || admission.Result.Disposition != quotaruntime.AdmissionForbidden || admission.Result.BlockingReason != "management_session_denied" {
		t.Fatalf("revoked delegated Admit() = %+v, %v", admission, err)
	}
}

func TestTeamOwnedDelegationPinsNonEmptyTeamPrecondition(t *testing.T) {
	_, _, reader, _ := testRuntime(t)
	credential := reader.credential
	credential.Kind = string(accesscredential.KindDelegation)
	credential.ManagementSessionID = "management-session-1"
	credential.PrincipalID = "principal-1"
	credential.DelegationEpoch = reader.policy.DelegationEpoch
	credential.UserID = reader.policy.UserID
	credential.TeamID = "team-1"
	credential.Audience = "vllm-sr-inference"
	projection := reader.policy
	projection.TeamID = credential.TeamID

	preconditions, err := compilePreconditions(
		"test:access", reader.location, accesscredential.KindDelegation,
		"delegation001", credential, reader.active, projection,
	)
	if err != nil {
		t.Fatal(err)
	}
	if countPreconditionField(preconditions, "team_id") != 1 {
		t.Fatalf("team-owned delegation did not pin Team identity: %+v", preconditions)
	}
	for index, precondition := range preconditions {
		if err := precondition.Validate(); err != nil {
			t.Fatalf("team-owned delegation precondition %d is invalid: %+v: %v", index, precondition, err)
		}
	}
}

func TestAuthenticateRejectsInvalidSecretBeforeAtomicGuard(t *testing.T) {
	runtime, issued, _, engine := testRuntime(t)
	replacement := "A"
	if issued.Plaintext[len(issued.Plaintext)-1:] == replacement {
		replacement = "B"
	}
	credential := issued.Plaintext[:len(issued.Plaintext)-1] + replacement
	authentication, err := runtime.Authenticate(context.Background(), AuthenticationRequest{Credential: credential})
	if err != nil {
		t.Fatalf("invalid credential returned internal error: %v", err)
	}
	if authentication.Result.Disposition != quotaruntime.AdmissionUnauthenticated ||
		authentication.Session.state != nil || engine.accessRequest != nil || engine.request != nil {
		t.Fatalf("invalid credential authentication = %+v, access request = %+v, admission request = %+v", authentication, engine.accessRequest, engine.request)
	}
}

func TestAdmitUsesNondisclosingForbiddenDecision(t *testing.T) {
	runtime, issued, _, engine := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	admission, err := runtime.Admit(context.Background(), AdmissionRequest{
		Session:     authentication.Session,
		Target:      Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-private", Permission: accesscontrol.GrantPermissionInvoke},
		AdmissionID: "admission-3", RequestDigest: "request-digest", LeaseDuration: time.Minute,
	})
	if err != nil {
		t.Fatalf("forbidden target returned internal error: %v", err)
	}
	if admission.Result.Disposition != quotaruntime.AdmissionForbidden || admission.Result.BlockingReason != "resource_not_found" || engine.request != nil {
		t.Fatalf("forbidden admission = %+v, engine request = %+v", admission, engine.request)
	}
}

func TestAdmitFailsClosedOnProjectionDigestMismatch(t *testing.T) {
	runtime, issued, reader, engine := testRuntime(t)
	reader.policy.Grants[0].ResourceID = "tampered"
	authentication, err := runtime.Authenticate(context.Background(), AuthenticationRequest{Credential: issued.Plaintext})
	if err == nil || authentication.Result.Disposition != quotaruntime.AdmissionUnavailable ||
		authentication.Session.state != nil || engine.accessRequest != nil || engine.request != nil {
		t.Fatalf("tampered projection authentication = %+v, error = %v, access request = %+v, admission request = %+v", authentication, err, engine.accessRequest, engine.request)
	}
}

func TestAuthorizeAndDiscoverReuseAtomicGuardWithoutQuotaMutation(t *testing.T) {
	runtime, issued, _, engine := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	checked, err := runtime.Authorize(context.Background(), AuthorizationRequest{
		Session: authentication.Session,
		Target:  Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-chat", Permission: accesscontrol.GrantPermissionInvoke},
	})
	if err != nil || !checked.Result.Allowed() || checked.Tenant.APIKeyID == "" {
		t.Fatalf("Authorize() = %+v, %v", checked, err)
	}
	if engine.accessRequest == nil || engine.request != nil {
		t.Fatalf("read-only check requests = access:%+v admission:%+v", engine.accessRequest, engine.request)
	}
	if countPreconditionReason(engine.accessRequest.Preconditions, "resource_denied") != 1 {
		t.Fatalf("target resource barrier was not pinned: %+v", engine.accessRequest.Preconditions)
	}
	discovered, err := runtime.Discover(context.Background(), DiscoveryRequest{
		Session: authentication.Session, ResourceType: accesscontrol.GrantResourceEntrypoint,
		Permission: accesscontrol.GrantPermissionInvoke,
	})
	if err != nil || len(discovered.ResourceIDs) != 1 || discovered.ResourceIDs[0] != "entry-chat" {
		t.Fatalf("Discover() = %+v, %v", discovered, err)
	}
	if engine.accessRequest == nil || countPreconditionReason(engine.accessRequest.Preconditions, "resource_denied") != 1 {
		t.Fatalf("discovery resource barriers = %+v", engine.accessRequest)
	}
}

func TestDiscoverCatalogPinsOneProjectionAcrossResourceClasses(t *testing.T) {
	runtime, issued, reader, engine := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	engine.onCheck = func() {
		// Simulate the active policy changing after the atomic guard. The
		// response must remain wholly derived from the already pinned snapshot.
		reader.policy = accessprojection.Projection{}
		reader.active = ActivePolicy{KeyID: "key-1", Revision: 99, Digest: "revoked", PublicationID: "publication-1"}
	}
	discovered, err := runtime.DiscoverCatalog(context.Background(), CatalogDiscoveryRequest{
		Session: authentication.Session,
		Queries: []DiscoveryQuery{
			{ResourceType: accesscontrol.GrantResourceEntrypoint, Permission: accesscontrol.GrantPermissionDiscover},
			{ResourceType: accesscontrol.GrantResourceModel, Permission: accesscontrol.GrantPermissionDiscover},
		},
	})
	if err != nil || !discovered.Result.Allowed() || discovered.Tenant.APIKeyID == "" {
		t.Fatalf("DiscoverCatalog() = %+v, %v", discovered, err)
	}
	if reader.activeReads != 1 || reader.policyReads != 1 {
		t.Fatalf("catalog discovery read active/policy %d/%d times, want 1/1", reader.activeReads, reader.policyReads)
	}
	if discovered.Tenant.PolicyRevision != 7 {
		t.Fatalf("policy revision = %d, want pinned revision 7", discovered.Tenant.PolicyRevision)
	}
	entrypoints := discovered.Resources[accesscontrol.GrantResourceEntrypoint]
	models := discovered.Resources[accesscontrol.GrantResourceModel]
	if len(entrypoints) != 1 || entrypoints[0] != "entry-chat" || len(models) != 1 || models[0] != "model-chat" {
		t.Fatalf("mixed or incomplete catalog: entrypoints=%v models=%v", entrypoints, models)
	}
	if engine.accessRequest == nil || countPreconditionReason(engine.accessRequest.Preconditions, "resource_denied") != 2 {
		t.Fatalf("catalog did not atomically guard every disclosed routing resource: %+v", engine.accessRequest)
	}
}

func TestSessionRerunsPinnedPolicyAndPublicationGuards(t *testing.T) {
	runtime, issued, reader, engine := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	engine.check = func(request quotaruntime.AccessCheckRequest) (quotaruntime.AccessCheckResult, error) {
		if countPreconditionReason(request.Preconditions, "policy_changed") < 3 ||
			countPreconditionReason(request.Preconditions, "routing_publication_changed") != 1 ||
			countPreconditionReason(request.Preconditions, "routing_revision_changed") != 1 {
			t.Fatalf("session operation omitted atomic policy/publication guards: %+v", request.Preconditions)
		}
		return quotaruntime.AccessCheckResult{
			Disposition: quotaruntime.AdmissionForbidden,
			Reason:      "policy_changed",
		}, nil
	}
	reader.active = ActivePolicy{KeyID: "key-1", Revision: 99, Digest: "changed"}

	authorized, err := runtime.Authorize(context.Background(), AuthorizationRequest{
		Session: authentication.Session,
		Target:  Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-chat", Permission: accesscontrol.GrantPermissionInvoke},
	})
	if err != nil || authorized.Result.Disposition != quotaruntime.AdmissionForbidden ||
		authorized.Result.Reason != "policy_changed" {
		t.Fatalf("Authorize() after policy switch = %+v, %v", authorized, err)
	}
	if reader.activeReads != 1 || reader.policyReads != 1 {
		t.Fatalf("session unexpectedly reread non-atomic policy state: active=%d policy=%d", reader.activeReads, reader.policyReads)
	}
}

func TestSessionIsBoundToCreatingRuntime(t *testing.T) {
	runtime, issued, _, _ := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	other, _, _, otherEngine := testRuntime(t)

	authorized, err := other.Authorize(context.Background(), AuthorizationRequest{
		Session: authentication.Session,
		Target:  Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-chat", Permission: accesscontrol.GrantPermissionInvoke},
	})
	if !errors.Is(err, ErrInvalidSession) || authorized.Result.Disposition != quotaruntime.AdmissionUnavailable {
		t.Fatalf("foreign session authorization = %+v, %v", authorized, err)
	}
	if otherEngine.accessRequest != nil || otherEngine.request != nil {
		t.Fatalf("foreign session reached atomic runtime: access=%+v admission=%+v", otherEngine.accessRequest, otherEngine.request)
	}
}

func TestTenantContextResultsAreDefensiveCopies(t *testing.T) {
	runtime, issued, _, _ := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	authentication.Tenant.RoutingClaims["tier"] = routingsnapshot.ClaimValue{Kind: "string", String: "mutated"}

	authorized, err := runtime.Authorize(context.Background(), AuthorizationRequest{
		Session: authentication.Session,
		Target:  Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-chat", Permission: accesscontrol.GrantPermissionInvoke},
	})
	if err != nil || !authorized.Result.Allowed() {
		t.Fatalf("Authorize() = %+v, %v", authorized, err)
	}
	if got := authorized.Tenant.RoutingClaims["tier"].String; got != "free" {
		t.Fatalf("session tenant claim = %q, want immutable value free", got)
	}
}

func TestJournalRejectsModifiedAdmissionIdentity(t *testing.T) {
	runtime, issued, _, engine := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	admission, err := runtime.Admit(context.Background(), AdmissionRequest{
		Session:     authentication.Session,
		Target:      Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-chat", Permission: accesscontrol.GrantPermissionInvoke},
		AdmissionID: "admission-tamper", RequestDigest: "request-digest", LeaseDuration: time.Minute,
	})
	if err != nil || !admission.Result.Allowed() {
		t.Fatalf("Admit() = %+v, %v", admission, err)
	}
	admission.Tenant.QuotaPartition = "other-partition"
	if _, err := runtime.JournalDispatch(context.Background(), DispatchJournalRequest{
		Admission: admission, DispatchID: "dispatch-1", Digest: "dispatch-digest",
	}); err == nil {
		t.Fatal("JournalDispatch() accepted a modified admission")
	}
	if engine.dispatchRequest != nil {
		t.Fatalf("modified admission reached atomic runtime: %+v", engine.dispatchRequest)
	}
}

func TestPrepareDispatchReturnsOpaquePinnedAuthorityFacts(t *testing.T) {
	runtime, issued, _, _ := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	admission, err := runtime.Admit(context.Background(), AdmissionRequest{
		Session:     authentication.Session,
		Target:      Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-chat", Permission: accesscontrol.GrantPermissionInvoke},
		AdmissionID: "admission-prepared", RequestDigest: strings.Repeat("b", 64), LeaseDuration: time.Minute,
	})
	if err != nil || !admission.Result.Allowed() {
		t.Fatalf("Admit() = %+v, %v", admission, err)
	}
	digest := strings.Repeat("a", 64)
	prepared, err := runtime.PrepareDispatch(admission, DispatchFacts{
		DispatchID: "dispatch-1", Ordinal: 2, DispatchPlanDigest: digest,
	})
	if err != nil {
		t.Fatalf("PrepareDispatch() error = %v", err)
	}
	if prepared.NamespaceID() != admission.Tenant.NamespaceID ||
		prepared.QuotaPartition() != admission.Tenant.QuotaPartition ||
		prepared.RoutingRevision() != admission.Tenant.RoutingRevision ||
		prepared.AdmissionID() != admission.Tenant.AdmissionID ||
		prepared.AdmissionDigest() != admission.RequestDigest ||
		prepared.DispatchID() != "dispatch-1" || prepared.Ordinal() != 2 ||
		prepared.DispatchPlanDigest() != digest {
		t.Fatalf("prepared dispatch = namespace=%q partition=%q admission=%q digest=%q dispatch=%q ordinal=%d plan=%q",
			prepared.NamespaceID(), prepared.QuotaPartition(), prepared.AdmissionID(), prepared.AdmissionDigest(),
			prepared.DispatchID(), prepared.Ordinal(), prepared.DispatchPlanDigest())
	}
	var zero PreparedDispatch
	if zero.NamespaceID() != "" || zero.AdmissionID() != "" || zero.Ordinal() != 0 {
		t.Fatal("zero PreparedDispatch exposed authority facts")
	}

	modified := admission
	modified.Tenant.AdmissionID = "other"
	if _, err := runtime.PrepareDispatch(modified, DispatchFacts{
		DispatchID: "dispatch-1", DispatchPlanDigest: digest,
	}); err == nil {
		t.Fatal("PrepareDispatch() accepted a modified admission")
	}
	if _, err := runtime.PrepareDispatch(admission, DispatchFacts{
		DispatchID: "dispatch-1", DispatchPlanDigest: "not-a-digest",
	}); err == nil {
		t.Fatal("PrepareDispatch() accepted an invalid plan digest")
	}
}

func TestReadAttemptEvidencePinsOneStableAdmissionRevision(t *testing.T) {
	runtime, issued, _, engine := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	admission, err := runtime.Admit(context.Background(), AdmissionRequest{
		Session: authentication.Session,
		Target: Target{
			ResourceType: accesscontrol.GrantResourceEntrypoint,
			ResourceID:   "entry-chat", Permission: accesscontrol.GrantPermissionInvoke,
		},
		AdmissionID: "admission-evidence", RequestDigest: strings.Repeat("b", 64),
		LeaseDuration: time.Minute,
	})
	if err != nil || !admission.Result.Allowed() {
		t.Fatalf("Admit() = %+v, %v", admission, err)
	}
	engine.readEvidence = func(request quotaruntime.ReadAttemptEvidenceRequest) (quotaruntime.ReadAttemptEvidenceResult, error) {
		return quotaruntime.ReadAttemptEvidenceResult{Revision: 7}, nil
	}
	dispatches := []AttemptEvidenceDispatch{
		{DispatchID: "dispatch-0", Ordinal: 0, DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-0", ModelRevision: 1},
		{DispatchID: "dispatch-1", Ordinal: 1, DispatchPlanDigest: strings.Repeat("c", 64), ModelID: "model-1", ModelRevision: 2},
	}
	snapshot, err := runtime.ReadAttemptEvidence(context.Background(), AttemptEvidenceRequest{
		Admission: admission, Dispatches: dispatches,
	})
	if err != nil || len(snapshot.Observations()) != 2 || len(engine.evidenceRequests) != 2 {
		t.Fatalf("ReadAttemptEvidence() = %+v, %v; requests=%d", snapshot, err, len(engine.evidenceRequests))
	}
	for index, request := range engine.evidenceRequests {
		// #nosec G115 -- this fixture contains two evidence requests.
		if request.AdmissionID != admission.Tenant.AdmissionID ||
			request.AdmissionDigest != admission.RequestDigest || request.Ordinal != uint32(index) {
			t.Fatalf("evidence request %d = %+v", index, request)
		}
	}
	if _, err := runtime.Settle(context.Background(), SettlementRequest{
		Admission: admission, AttemptEvidence: snapshot,
		FinalizationDigest: "final-evidence", Event: `{"admissionId":"admission-evidence"}`,
		EventEvidenceState: "known",
	}); err != nil {
		t.Fatalf("Settle() with genuine evidence snapshot error = %v", err)
	}
	if engine.finalizationRequest == nil || engine.finalizationRequest.DispatchCount != 2 ||
		engine.finalizationRequest.EvidenceRevision != 7 {
		t.Fatalf("finalization attempt CAS = %+v", engine.finalizationRequest)
	}
	if _, err := runtime.Settle(context.Background(), SettlementRequest{
		Admission:          admission,
		AttemptEvidence:    AttemptEvidenceSnapshot{Dispatches: snapshot.Observations()},
		FinalizationDigest: "forged-evidence", Event: `{"admissionId":"admission-evidence"}`,
		EventEvidenceState: "known",
	}); err == nil {
		t.Fatal("Settle() accepted a caller-constructed attempt evidence snapshot")
	}

	engine.evidenceRequests = nil
	engine.readEvidence = func(request quotaruntime.ReadAttemptEvidenceRequest) (quotaruntime.ReadAttemptEvidenceResult, error) {
		return quotaruntime.ReadAttemptEvidenceResult{Revision: uint64(request.Ordinal + 1)}, nil
	}
	if _, err := runtime.ReadAttemptEvidence(context.Background(), AttemptEvidenceRequest{
		Admission: admission, Dispatches: dispatches,
	}); !errors.Is(err, quotaruntime.ErrEvidenceChanged) {
		t.Fatalf("changing attempt revision error = %v, want %v", err, quotaruntime.ErrEvidenceChanged)
	}
}

func countPreconditionReason(preconditions []quotaruntime.AdmissionPrecondition, reason string) int {
	count := 0
	for _, precondition := range preconditions {
		if precondition.Reason == reason {
			count++
		}
	}
	return count
}

func countPreconditionField(preconditions []quotaruntime.AdmissionPrecondition, field string) int {
	count := 0
	for _, precondition := range preconditions {
		if precondition.Field == field {
			count++
		}
	}
	return count
}

func authenticate(t *testing.T, runtime *Runtime, issued accesscredential.Issued) Authentication {
	t.Helper()
	authentication, err := runtime.Authenticate(context.Background(), AuthenticationRequest{Credential: issued.Plaintext})
	if err != nil || !authentication.Result.Allowed() || authentication.Session.state == nil || authentication.Tenant.APIKeyID == "" {
		t.Fatalf("Authenticate() = %+v, %v", authentication, err)
	}
	return authentication
}

func testRuntime(t *testing.T) (*Runtime, accesscredential.Issued, *fakeReader, *fakeEngine) {
	t.Helper()
	pepper := []byte("0123456789abcdef0123456789abcdef")
	keyring := accesscredential.PepperKeyring{ActiveVersion: "pepper-1", Keys: map[string][]byte{"pepper-1": pepper}}
	issued, err := keyring.Issue(accesscredential.KindAPIKey, "publicid0001")
	if err != nil {
		t.Fatal(err)
	}
	projection := testProjection(t)
	now := time.Now().UTC().Add(-time.Minute)
	reader := &fakeReader{
		location: CredentialLocation{
			NamespaceID: projection.NamespaceID, QuotaPartition: projection.QuotaPartition,
			PublicationID: "publication-1", RuntimeEpoch: 1, RoutingRevision: 1,
			RoutingDocumentDigest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		},
		credential: accessprojection.CredentialProjection{
			Kind: string(accesscredential.KindAPIKey), KID: issued.Digest.PublicID, KeyID: projection.KeyID,
			SecretHMAC: issued.Digest.HMAC, PepperVersion: issued.Digest.PepperVersion,
			Status: string(accesscontrol.CredentialStatusActive), NotBefore: now,
		},
		active: ActivePolicy{
			KeyID: projection.KeyID, Revision: projection.Revision, Digest: projection.Digest,
			PublicationID: "publication-1", RuntimeEpoch: 1, RoutingRevision: 1,
			RoutingDocumentDigest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		},
		policy: projection,
	}
	engine := &fakeEngine{result: quotaruntime.AdmissionResult{
		Disposition: quotaruntime.AdmissionAllowed, PlanDigest: strings.Repeat("f", 64),
	}}
	runtime, err := New(RuntimeOptions{
		Reader: reader, Engine: engine, APIKeyPeppers: keyring, DelegationPeppers: keyring,
		DelegationAudience: "vllm-sr-inference",
		DelegationBarriers: &fakeDelegationBarriers{state: managementauth.DelegationBarrierState{Ready: true}},
		KeyPrefix:          "test:access",
	})
	if err != nil {
		t.Fatal(err)
	}
	return runtime, issued, reader, engine
}

func testProjection(t *testing.T) accessprojection.Projection {
	return testProjectionAtRevision(t, 7)
}

func testProjectionAtRevision(t *testing.T, revision uint64) accessprojection.Projection {
	t.Helper()
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	namespace := accesscontrol.Namespace{
		ID: "namespace-1", Name: "default", QuotaPartitionID: "partition-1", BillingCurrency: "USD",
		Status: accesscontrol.NamespaceStatusActive, Revision: 1, RuntimeEpoch: 1, CreatedAt: now, UpdatedAt: now,
	}
	user := accesscontrol.User{
		NamespaceID: namespace.ID, ID: "user-1", Email: "user@example.com", DisplayName: "User",
		Status: accesscontrol.UserStatusActive, CreatedAt: now, UpdatedAt: now,
	}
	key := accesscontrol.APIKey{
		NamespaceID: namespace.ID, ID: "key-1", Name: "User key", Owner: user.SubjectRef(),
		Status: accesscontrol.APIKeyStatusActive, PolicyEpoch: 1, DelegationEpoch: 1, Revision: 1,
		CreatedAt: now, UpdatedAt: now,
	}
	keyExpiry := now.AddDate(10, 0, 0)
	key.ExpiresAt = &keyExpiry
	accessPolicy := accesscontrol.AccessPolicy{
		NamespaceID: namespace.ID, ID: "access-user", DisplayName: "Chat access", Status: accesscontrol.PolicyStatusActive,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
		Grants: []accesscontrol.AccessPolicyGrant{
			{
				PolicyID: "access-user", Resource: accesscontrol.GrantResource{Type: accesscontrol.GrantResourceEntrypoint, ID: "entry-chat"},
				Permission: accesscontrol.GrantPermissionInvoke, Effect: accesscontrol.GrantEffectAllow,
			},
			{
				PolicyID: "access-user", Resource: accesscontrol.GrantResource{Type: accesscontrol.GrantResourceEntrypoint, ID: "entry-chat"},
				Permission: accesscontrol.GrantPermissionDiscover, Effect: accesscontrol.GrantEffectAllow,
			},
			{
				PolicyID: "access-user", Resource: accesscontrol.GrantResource{Type: accesscontrol.GrantResourceModel, ID: "model-chat"},
				Permission: accesscontrol.GrantPermissionDiscover, Effect: accesscontrol.GrantEffectAllow,
			},
		},
	}
	ratePolicy := accesscontrol.RateLimitPolicy{
		NamespaceID: namespace.ID, ID: "rate-user", DisplayName: "User budget", Status: accesscontrol.PolicyStatusActive,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
		Rules: []accesscontrol.RateLimitRule{{
			ID: "rule-rpm", PolicyID: "rate-user", Metric: accesscontrol.RateMetricRequests,
			Algorithm: accesscontrol.RateAlgorithmSlidingLog, Limit: "12", Window: time.Minute,
			Accounting: accesscontrol.RateAccountingRequest, Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
	}
	projection, err := accessprojection.Compile(accessprojection.Candidate{
		Revision: revision, Namespace: namespace, Key: key,
		Relationships: accesscontrol.APIKeyRelationships{OwnerUser: &user},
		RoutingClaims: map[string]routingsnapshot.ClaimValue{
			"tier": {Kind: "string", String: "free"},
		},
		UserAccessBindings: []accesscontrol.AccessPolicyBinding{{
			ID: "access-binding-user", NamespaceID: namespace.ID, Subject: user.SubjectRef(), PolicyID: accessPolicy.ID,
			Status: accesscontrol.BindingStatusActive, Revision: 1,
		}},
		AccessPolicies: map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy{accessPolicy.ID: accessPolicy},
		UserRateBindings: []accesscontrol.RateLimitBinding{{
			ID: "rate-binding-user", NamespaceID: namespace.ID, Subject: user.SubjectRef(), PolicyID: ratePolicy.ID,
			Mode: accesscontrol.RateBindingAllocation, QuotaPartitionID: namespace.QuotaPartitionID,
			Status: accesscontrol.BindingStatusActive, Revision: 1,
		}},
		RatePolicies: map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy{ratePolicy.ID: ratePolicy},
	}, accessprojection.CompileOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return projection
}
