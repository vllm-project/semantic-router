package accessruntime

import (
	"context"
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageaccounting"
)

var (
	ErrProjectionNotFound = errors.New("access runtime projection not found")
	ErrRuntimeUnavailable = errors.New("access runtime unavailable")
	ErrRuntimeCorrupt     = errors.New("access runtime projection is corrupt")
	ErrInvalidSession     = errors.New("access runtime session is invalid")
)

// ActivePolicy is the only policy pointer honored by inference. A staged
// document is inert until the projector atomically advances this value.
type ActivePolicy struct {
	KeyID               string
	Revision            uint64
	Digest              string
	PublicationID       string
	RuntimeEpoch        uint64
	RoutingRevision     int64
	RoutingSnapshotHash string
}

// AppliedPolicy is a credential-free Management view of the one immutable
// projection currently honored by inference for a logical API key.
type AppliedPolicy struct {
	Active     ActivePolicy
	Projection accessprojection.Projection
}

// AppliedPolicyReader resolves a logical key directly inside its namespace
// partition. It never accepts or verifies raw credential material.
type AppliedPolicyReader interface {
	ReadAppliedPolicy(context.Context, string, string, string) (AppliedPolicy, error)
}

// CredentialLocation is the immutable namespace publication selected by the
// global public-kid directory. The directory is only a locator; every field is
// pinned again against partition-local publication gates during admission.
type CredentialLocation struct {
	NamespaceID         string
	QuotaPartition      string
	PublicationID       string
	RuntimeEpoch        uint64
	RoutingRevision     int64
	RoutingSnapshotHash string
}

// ProjectionReader is the narrow read seam implemented by Valkey/Redis. The
// global directory only locates a partition; all remaining records share that
// partition's hash tag and are pinned again by the atomic guard.
type ProjectionReader interface {
	LocateCredential(context.Context, accesscredential.Kind, string) (CredentialLocation, error)
	ReadCredential(context.Context, CredentialLocation, accesscredential.Kind, string) (accessprojection.CredentialProjection, error)
	ReadActivePolicy(context.Context, CredentialLocation, string) (ActivePolicy, error)
	ReadPolicy(context.Context, CredentialLocation, ActivePolicy) (accessprojection.Projection, error)
}

// AtomicEngine is the only request-time quota-store seam. Admission and
// finalization are atomic mutations; attempt evidence is an admission-bound,
// read-only snapshot used to authorize the final compare-and-set.
type AtomicEngine interface {
	CheckAccess(context.Context, quotaruntime.AccessCheckRequest) (quotaruntime.AccessCheckResult, error)
	Admit(context.Context, quotaruntime.AdmissionRequest) (quotaruntime.AdmissionResult, error)
	JournalDispatch(context.Context, quotaruntime.DispatchJournalRequest) (quotaruntime.MutationResult, error)
	ReadAttemptEvidence(context.Context, quotaruntime.ReadAttemptEvidenceRequest) (quotaruntime.ReadAttemptEvidenceResult, error)
	Finalize(context.Context, quotaruntime.FinalizationRequest) (quotaruntime.FinalizationResult, error)
}

type RuntimeOptions struct {
	Reader             ProjectionReader
	Engine             AtomicEngine
	APIKeyPeppers      accesscredential.PepperKeyring
	DelegationPeppers  accesscredential.PepperKeyring
	DelegationAudience string
	DelegationBarriers managementauth.DelegationRevocationBarrierStore
	KeyPrefix          string
}

type Target struct {
	ResourceType accesscontrol.GrantResourceType
	ResourceID   accesscontrol.ResourceID
	Permission   accesscontrol.GrantPermission
}

func (t Target) validate() error {
	return (accesscontrol.GrantResource{Type: t.ResourceType, ID: t.ResourceID}).Validate()
}

// Session is an opaque, process-local handle to one authenticated immutable
// access projection. A session can only be created by Runtime.Authenticate and
// can only be consumed by the Runtime instance that created it.
//
// The handle contains neither the presented credential nor its verifier.
type Session struct {
	state *sessionState
}

// AuthenticationRequest is the only access-runtime request that accepts raw
// credential material. The credential is verified once and is never retained
// in the returned Session or TenantContext.
type AuthenticationRequest struct {
	Credential string
}

type AuthenticationSource string

const (
	AuthenticationSourceAPIKey    AuthenticationSource = "api_key"
	AuthenticationSourceDelegated AuthenticationSource = "delegated_inference_session"
)

type Authentication struct {
	Result  quotaruntime.AccessCheckResult
	Tenant  TenantContext
	Session Session
	Source  AuthenticationSource
}

type AuthorizationRequest struct {
	Session Session
	Target  Target
}

type Authorization struct {
	Result quotaruntime.AccessCheckResult
	Tenant TenantContext
	Target Target
}

// AdmissionRequest contains only request-bound facts supplied by the Router.
// Authentication and immutable policy selection are carried by Session.
type AdmissionRequest struct {
	Session       Session
	Target        Target
	AdmissionID   string
	RequestDigest string
	LeaseDuration time.Duration
}

type DiscoveryRequest struct {
	Session      Session
	ResourceType accesscontrol.GrantResourceType
	Permission   accesscontrol.GrantPermission
}

type DiscoveryQuery struct {
	ResourceType accesscontrol.GrantResourceType
	Permission   accesscontrol.GrantPermission
}

// CatalogDiscoveryRequest groups every resource class required by one public
// catalog response. Session pins one projection for the complete request,
// preventing a policy revision from being mixed mid-list.
type CatalogDiscoveryRequest struct {
	Session Session
	Queries []DiscoveryQuery
}

type Discovery struct {
	Result      quotaruntime.AccessCheckResult
	Tenant      TenantContext
	ResourceIDs []string
}

type CatalogDiscovery struct {
	Result    quotaruntime.AccessCheckResult
	Tenant    TenantContext
	Resources map[accesscontrol.GrantResourceType][]string
}

// TenantContext is the immutable authenticated inference identity retained in
// process for one request. If it crosses a process boundary it must be encoded
// by the separately audience-bound tenant-context signer.
type TenantContext struct {
	AdmissionID     string
	NamespaceID     string
	QuotaPartition  string
	APIKeyID        string
	UserID          string
	TeamID          string
	PolicyRevision  uint64
	PolicyDigest    string
	PublicationID   string
	RuntimeEpoch    uint64
	RoutingRevision int64
	RoutingDigest   string
	BillingCurrency string
	RoutingClaims   map[string]routingsnapshot.ClaimValue
}

type Admission struct {
	Result        quotaruntime.AdmissionResult
	Tenant        TenantContext
	Rules         []quotaruntime.RuleBinding
	Target        Target
	RequestDigest string
	PreparedAt    time.Time

	state *admissionState
}

// DispatchJournalRequest records one physical dispatch before it can reach a
// backend. Digest is a stable fingerprint of the immutable dispatch plan.
type DispatchJournalRequest struct {
	Admission  Admission
	DispatchID string
	Ordinal    uint32
	Digest     string
}

// DispatchFacts contains the immutable, non-secret plan identity selected by
// routing. PrepareDispatch binds it to a genuine process-local Admission.
type DispatchFacts struct {
	DispatchID         string
	Ordinal            uint32
	DispatchPlanDigest string
}

// PreparedDispatch is an opaque, immutable bridge from access admission to a
// short-lived dispatch-capability issuer. Its fields are intentionally private:
// callers can inspect the non-secret values but cannot construct or retarget a
// prepared authority value.
type PreparedDispatch struct {
	state *preparedDispatchState
}

func (p PreparedDispatch) preparedState() *preparedDispatchState {
	if p.state == nil || p.state.owner == nil {
		return nil
	}
	return p.state
}

func (p PreparedDispatch) NamespaceID() string {
	state := p.preparedState()
	if state == nil {
		return ""
	}
	return state.namespaceID
}

func (p PreparedDispatch) QuotaPartition() string {
	state := p.preparedState()
	if state == nil {
		return ""
	}
	return state.quotaPartition
}

func (p PreparedDispatch) PublicationID() string {
	state := p.preparedState()
	if state == nil {
		return ""
	}
	return state.publicationID
}

func (p PreparedDispatch) RuntimeEpoch() uint64 {
	state := p.preparedState()
	if state == nil {
		return 0
	}
	return state.runtimeEpoch
}

func (p PreparedDispatch) RoutingRevision() int64 {
	state := p.preparedState()
	if state == nil {
		return 0
	}
	return state.routingRevision
}

func (p PreparedDispatch) RoutingDigest() string {
	state := p.preparedState()
	if state == nil {
		return ""
	}
	return state.routingDigest
}

func (p PreparedDispatch) AdmissionID() string {
	state := p.preparedState()
	if state == nil {
		return ""
	}
	return state.admissionID
}

func (p PreparedDispatch) AdmissionDigest() string {
	state := p.preparedState()
	if state == nil {
		return ""
	}
	return state.admissionDigest
}

func (p PreparedDispatch) DispatchID() string {
	state := p.preparedState()
	if state == nil {
		return ""
	}
	return state.dispatchID
}

func (p PreparedDispatch) Ordinal() uint32 {
	state := p.preparedState()
	if state == nil {
		return 0
	}
	return state.ordinal
}

func (p PreparedDispatch) DispatchPlanDigest() string {
	state := p.preparedState()
	if state == nil {
		return ""
	}
	return state.dispatchPlanDigest
}

// AttemptEvidenceDispatch is the immutable dispatch identity the request
// finalizer expects to find in the authoritative backend-attempt journal.
type AttemptEvidenceDispatch struct {
	DispatchID         string
	Ordinal            uint32
	DispatchPlanDigest string
	ModelID            string
	ModelRevision      int64
}

type AttemptEvidenceRequest struct {
	Admission  Admission
	Dispatches []AttemptEvidenceDispatch
}

// AttemptEvidenceObservation is one read-only dispatch result. Present is
// false when the dispatch intent was journaled but the backend invoker never
// began its bounded execution envelope.
type AttemptEvidenceObservation struct {
	DispatchID string
	Present    bool
	Evidence   quotaruntime.DispatchAttemptEvidence
}

type attemptEvidenceSnapshotState struct {
	owner           *runtimeIdentity
	admissionID     string
	admissionDigest string
	revision        uint64
	dispatchCount   uint32
}

// AttemptEvidenceSnapshot is an opaque, process-local proof that every
// journaled dispatch was observed at one stable attempt-journal revision.
// Finalize compare-and-sets that revision in the same Redis operation that
// applies actual usage and appends the terminal UsageEvent.
type AttemptEvidenceSnapshot struct {
	Dispatches []AttemptEvidenceObservation
	state      *attemptEvidenceSnapshotState
}

func (s AttemptEvidenceSnapshot) Observations() []AttemptEvidenceObservation {
	result := make([]AttemptEvidenceObservation, len(s.Dispatches))
	for index, observation := range s.Dispatches {
		result[index] = observation
		result[index].Evidence.Attempts = append(
			[]quotaruntime.AttemptEvidence(nil),
			observation.Evidence.Attempts...,
		)
	}
	return result
}

// SettlementRequest closes one admitted inference request. Aggregate contains
// only backend-authoritative actual usage. Event is the already validated,
// redacted durable usage envelope written atomically with quota settlement.
type SettlementRequest struct {
	Admission          Admission
	AttemptEvidence    AttemptEvidenceSnapshot
	Aggregate          usageaccounting.Aggregate
	FinalizationDigest string
	Event              string
	FenceID            string
}
