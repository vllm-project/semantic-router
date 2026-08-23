package dispatchauthority

import (
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
)

type admissionPreparer interface {
	prepare(accessruntime.Admission, accessruntime.DispatchFacts) (preparedIdentity, error)
}

type runtimeAdmissionPreparer struct {
	runtime *accessruntime.Runtime
}

func (preparer runtimeAdmissionPreparer) prepare(
	admission accessruntime.Admission,
	facts accessruntime.DispatchFacts,
) (preparedIdentity, error) {
	prepared, err := preparer.runtime.PrepareDispatch(admission, facts)
	if err != nil {
		return preparedIdentity{}, err
	}
	return preparedIdentity{
		namespaceID:        prepared.NamespaceID(),
		quotaPartition:     prepared.QuotaPartition(),
		publicationID:      prepared.PublicationID(),
		runtimeEpoch:       prepared.RuntimeEpoch(),
		routingRevision:    prepared.RoutingRevision(),
		routingDigest:      prepared.RoutingDigest(),
		admissionID:        prepared.AdmissionID(),
		admissionDigest:    prepared.AdmissionDigest(),
		dispatchID:         prepared.DispatchID(),
		ordinal:            prepared.Ordinal(),
		dispatchPlanDigest: prepared.DispatchPlanDigest(),
	}, nil
}

// MeteredAuthority is the only production signer for quota-admitted backend
// dispatches. It owns its capability issuer and never exposes the issuer's raw
// field-based signing methods.
type MeteredAuthority struct {
	mu       sync.RWMutex
	preparer admissionPreparer
	issuer   *backendinvoker.CapabilityIssuer
	identity *authorityIdentity
	closed   bool
}

// NewMeteredAuthority constructs a fail-closed authority around one access
// runtime. The signer defensively copies its keyring; callers retain ownership
// of the options and their source key material.
func NewMeteredAuthority(options MeteredAuthorityOptions) (*MeteredAuthority, error) {
	if options.Access == nil {
		return nil, fmt.Errorf("access runtime is required")
	}
	return newMeteredAuthority(runtimeAdmissionPreparer{runtime: options.Access}, options.Issuer)
}

func newMeteredAuthority(
	preparer admissionPreparer,
	issuerOptions backendinvoker.CapabilityIssuerOptions,
) (*MeteredAuthority, error) {
	if preparer == nil {
		return nil, fmt.Errorf("admission preparer is required")
	}
	issuer, err := backendinvoker.NewCapabilityIssuer(issuerOptions)
	if err != nil {
		return nil, fmt.Errorf("dispatch capability issuer: %w", err)
	}
	return &MeteredAuthority{
		preparer: preparer,
		issuer:   issuer,
		identity: &authorityIdentity{marker: 1},
	}, nil
}

// IssuePrimary signs one exact direct backend request after revalidating that
// its dispatch facts belong to an unmodified, allowed Admission.
func (authority *MeteredAuthority) IssuePrimary(request PrimaryIssueRequest) (string, error) {
	if authority == nil {
		return "", fmt.Errorf("metered dispatch authority is unavailable")
	}
	authority.mu.RLock()
	defer authority.mu.RUnlock()
	if authority.closed || authority.preparer == nil || authority.issuer == nil {
		return "", fmt.Errorf("metered dispatch authority is closed")
	}
	prepared, err := authority.preparer.prepare(request.Admission, request.Dispatch)
	if err != nil {
		return "", fmt.Errorf("prepare metered dispatch: %w", err)
	}
	prepared.requestID = request.RequestID
	return authority.issuer.Issue(capabilityRequest(prepared, primaryDispatchType, request.Final))
}

// IssueGrant signs a nested-dispatch grant after revalidating that its
// dispatch facts belong to an unmodified, allowed Admission. The grant cannot
// call a backend until VerifyGrant and IssueFromGrant bind the final request.
func (authority *MeteredAuthority) IssueGrant(request GrantIssueRequest) (string, error) {
	if authority == nil {
		return "", fmt.Errorf("metered dispatch authority is unavailable")
	}
	authority.mu.RLock()
	defer authority.mu.RUnlock()
	if authority.closed || authority.preparer == nil || authority.issuer == nil {
		return "", fmt.Errorf("metered dispatch authority is closed")
	}
	prepared, err := authority.preparer.prepare(request.Admission, request.Dispatch)
	if err != nil {
		return "", fmt.Errorf("prepare metered dispatch grant: %w", err)
	}
	prepared.requestID = request.RequestID
	return authority.issuer.IssueGrant(backendinvoker.DispatchGrantIssueRequest{
		NamespaceID:     prepared.namespaceID,
		QuotaPartition:  prepared.quotaPartition,
		PublicationID:   prepared.publicationID,
		RuntimeEpoch:    prepared.runtimeEpoch,
		RoutingRevision: prepared.routingRevision,
		RoutingDigest:   prepared.routingDigest,
		AdmissionID:     prepared.admissionID,
		AdmissionDigest: prepared.admissionDigest,
		RequestID:       prepared.requestID,
		Candidates: []backendinvoker.DispatchCandidate{{
			DispatchID: prepared.dispatchID, DispatchType: grantedDispatchType,
			Ordinal: int(prepared.ordinal), DispatchPlanDigest: prepared.dispatchPlanDigest,
			ModelID: request.Model.ID, ModelRevision: request.Model.Revision,
		}},
	})
}

// VerifyGrant authenticates an internal grant and returns authority-bound,
// process-local proof. A proof from another authority cannot be reused.
func (authority *MeteredAuthority) VerifyGrant(
	token string,
	expected GrantVerificationRequest,
) (VerifiedGrant, error) {
	if authority == nil {
		return VerifiedGrant{}, fmt.Errorf("metered dispatch authority is unavailable")
	}
	authority.mu.RLock()
	defer authority.mu.RUnlock()
	if authority.closed || authority.issuer == nil || authority.identity == nil {
		return VerifiedGrant{}, fmt.Errorf("metered dispatch authority is closed")
	}
	grant, err := authority.issuer.VerifyGrant(token)
	if err != nil {
		return VerifiedGrant{}, err
	}
	if err := verifyGrantContext(grant, expected); err != nil {
		return VerifiedGrant{}, err
	}
	return VerifiedGrant{
		grant: grant, owner: authority.identity,
		generation: expected.Generation, requestID: expected.RequestID,
	}, nil
}

// IssueFromGrant binds a verified nested grant to the exact final request. The
// grant's pinned model identity cannot be changed by the nested caller.
func (authority *MeteredAuthority) IssueFromGrant(
	verified VerifiedGrant,
	request FinalRequest,
) (string, error) {
	if authority == nil {
		return "", fmt.Errorf("metered dispatch authority is unavailable")
	}
	authority.mu.RLock()
	defer authority.mu.RUnlock()
	if authority.closed || authority.issuer == nil || authority.identity == nil {
		return "", fmt.Errorf("metered dispatch authority is closed")
	}
	if verified.owner == nil || verified.owner != authority.identity {
		return "", fmt.Errorf("verified metered dispatch grant is unavailable")
	}
	if err := verifySingleGrantModel(verified.grant, request.Model); err != nil {
		return "", err
	}
	return authority.issuer.IssueFromGrant(verified.grant, backendinvoker.DispatchFinalRequest{
		Method:     request.Method,
		Path:       request.Path,
		Query:      request.Query,
		WireFormat: request.WireFormat,
		Body:       request.Body,
	})
}

func verifySingleGrantModel(
	grant backendinvoker.VerifiedDispatchGrant,
	model ModelIdentity,
) error {
	candidates, ok := backendinvoker.VerifiedGrantCandidates(grant)
	if !ok || len(candidates) != 1 || candidates[0].ModelID != model.ID ||
		candidates[0].ModelRevision != model.Revision {
		return fmt.Errorf("dispatch grant model mismatch")
	}
	return nil
}

// Close is idempotent. It prevents new issuance, waits for in-flight issuance,
// and zeroes the authority-owned signing keys.
func (authority *MeteredAuthority) Close() error {
	if authority == nil {
		return nil
	}
	authority.mu.Lock()
	defer authority.mu.Unlock()
	if authority.closed {
		return nil
	}
	authority.closed = true
	authority.preparer = nil
	authority.identity = nil
	issuer := authority.issuer
	authority.issuer = nil
	return issuer.Close()
}

func capabilityRequest(
	prepared preparedIdentity,
	dispatchType string,
	request FinalRequest,
) backendinvoker.CapabilityIssueRequest {
	return backendinvoker.CapabilityIssueRequest{
		NamespaceID:     prepared.namespaceID,
		QuotaPartition:  prepared.quotaPartition,
		PublicationID:   prepared.publicationID,
		RuntimeEpoch:    prepared.runtimeEpoch,
		RoutingRevision: prepared.routingRevision,
		RoutingDigest:   prepared.routingDigest,
		AdmissionID:     prepared.admissionID,
		AdmissionDigest: prepared.admissionDigest,
		RequestID:       prepared.requestID,
		Candidates: []backendinvoker.DispatchCandidate{{
			DispatchID: prepared.dispatchID, DispatchType: dispatchType,
			Ordinal: int(prepared.ordinal), DispatchPlanDigest: prepared.dispatchPlanDigest,
			ModelID: request.Model.ID, ModelRevision: request.Model.Revision,
		}},
		Method:     request.Method,
		Path:       request.Path,
		Query:      request.Query,
		WireFormat: request.WireFormat,
		Body:       request.Body,
	}
}

func verifyGrantContext(
	grant backendinvoker.VerifiedDispatchGrant,
	expected GrantVerificationRequest,
) error {
	claims, ok := backendinvoker.VerifiedGrantClaims(grant)
	if !ok {
		return fmt.Errorf("verified dispatch grant claims are unavailable")
	}
	if expected.Generation.Validate() != nil || claims.NamespaceID != expected.Generation.NamespaceID ||
		claims.QuotaPartition != expected.Generation.QuotaPartition ||
		claims.PublicationID != expected.Generation.PublicationID ||
		claims.RuntimeEpoch != expected.Generation.RuntimeEpoch ||
		claims.RoutingRevision != expected.Generation.SnapshotRevision ||
		claims.RoutingDigest != expected.Generation.RoutingDigest ||
		claims.RequestID != expected.RequestID {
		return fmt.Errorf("dispatch grant request context mismatch")
	}
	return nil
}
