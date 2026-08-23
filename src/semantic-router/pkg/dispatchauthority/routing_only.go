package dispatchauthority

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
)

// RoutingOnlyAuthority is the separately named authority for an explicitly
// public namespace. It does not fabricate an access Admission: it derives a
// synthetic accounting identity only after validating the exact active
// publication and its process-local immutable routing snapshot.
type RoutingOnlyAuthority struct {
	mu           sync.RWMutex
	namespaceID  string
	publications RoutingPublicationReader
	snapshots    backendinvoker.RoutingSnapshotSource
	issuer       *backendinvoker.CapabilityIssuer
	identity     *authorityIdentity
	closed       bool
}

func NewRoutingOnlyAuthority(options RoutingOnlyAuthorityOptions) (*RoutingOnlyAuthority, error) {
	parsed, err := uuid.Parse(options.NamespaceID)
	if err != nil || parsed == uuid.Nil || parsed.String() != options.NamespaceID {
		return nil, fmt.Errorf("routing-only authority requires one canonical public namespace")
	}
	if options.Publications == nil {
		return nil, fmt.Errorf("routing-only publication reader is required")
	}
	issuer, err := backendinvoker.NewCapabilityIssuer(options.Issuer)
	if err != nil {
		return nil, fmt.Errorf("routing-only dispatch capability issuer: %w", err)
	}
	return &RoutingOnlyAuthority{
		namespaceID: options.NamespaceID, publications: options.Publications,
		issuer: issuer, identity: &authorityIdentity{marker: 1},
	}, nil
}

func (authority *RoutingOnlyAuthority) AttachRoutingSnapshots(source backendinvoker.RoutingSnapshotSource) error {
	if authority == nil || source == nil {
		return fmt.Errorf("routing-only snapshot source is required")
	}
	authority.mu.Lock()
	defer authority.mu.Unlock()
	if authority.closed {
		return fmt.Errorf("routing-only dispatch authority is closed")
	}
	if authority.snapshots != nil {
		return fmt.Errorf("routing-only snapshot source is already attached")
	}
	authority.snapshots = source
	return nil
}

func (authority *RoutingOnlyAuthority) IssuePrimary(
	ctx context.Context,
	request RoutingOnlyIssueRequest,
) (string, error) {
	authority.mu.RLock()
	defer authority.mu.RUnlock()
	prepared, err := authority.prepareLocked(ctx, request.Generation, request.RequestID, request.Dispatch, request.Final.Model)
	if err != nil {
		return "", err
	}
	return authority.issuer.Issue(capabilityRequest(prepared, primaryDispatchType, request.Final))
}

func (authority *RoutingOnlyAuthority) IssueGrant(
	ctx context.Context,
	request RoutingOnlyGrantIssueRequest,
) (string, error) {
	authority.mu.RLock()
	defer authority.mu.RUnlock()
	prepared, err := authority.prepareLocked(ctx, request.Generation, request.RequestID, request.Dispatch, request.Model)
	if err != nil {
		return "", err
	}
	return authority.issuer.IssueGrant(backendinvoker.DispatchGrantIssueRequest{
		NamespaceID: prepared.namespaceID, QuotaPartition: prepared.quotaPartition,
		PublicationID: prepared.publicationID, RuntimeEpoch: prepared.runtimeEpoch,
		RoutingRevision: prepared.routingRevision, RoutingDigest: prepared.routingDigest,
		AdmissionID: prepared.admissionID, AdmissionDigest: prepared.admissionDigest,
		RequestID: prepared.requestID,
		Candidates: []backendinvoker.DispatchCandidate{{
			DispatchID: prepared.dispatchID, DispatchType: grantedDispatchType,
			Ordinal: int(prepared.ordinal), DispatchPlanDigest: prepared.dispatchPlanDigest,
			ModelID: request.Model.ID, ModelRevision: request.Model.Revision,
		}},
	})
}

func (authority *RoutingOnlyAuthority) VerifyGrant(
	ctx context.Context,
	token string,
	expected GrantVerificationRequest,
) (VerifiedGrant, error) {
	if authority == nil {
		return VerifiedGrant{}, fmt.Errorf("routing-only dispatch authority is unavailable")
	}
	authority.mu.RLock()
	defer authority.mu.RUnlock()
	if authority.closed || authority.issuer == nil || authority.identity == nil {
		return VerifiedGrant{}, fmt.Errorf("routing-only dispatch authority is closed")
	}
	if err := authority.validateGenerationLocked(ctx, expected.Generation, ModelIdentity{}); err != nil {
		return VerifiedGrant{}, err
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

func (authority *RoutingOnlyAuthority) IssueFromGrant(
	ctx context.Context,
	verified VerifiedGrant,
	request FinalRequest,
) (string, error) {
	if authority == nil {
		return "", fmt.Errorf("routing-only dispatch authority is unavailable")
	}
	authority.mu.RLock()
	defer authority.mu.RUnlock()
	if authority.closed || authority.issuer == nil || authority.identity == nil {
		return "", fmt.Errorf("routing-only dispatch authority is closed")
	}
	if verified.owner == nil || verified.owner != authority.identity {
		return "", fmt.Errorf("verified routing-only dispatch grant is unavailable")
	}
	if err := verifySingleGrantModel(verified.grant, request.Model); err != nil {
		return "", err
	}
	if err := authority.validateGenerationLocked(ctx, verified.generation, request.Model); err != nil {
		return "", err
	}
	return authority.issuer.IssueFromGrant(verified.grant, backendinvoker.DispatchFinalRequest{
		Method: request.Method, Path: request.Path, Query: request.Query, WireFormat: request.WireFormat, Body: request.Body,
	})
}

func (authority *RoutingOnlyAuthority) prepareLocked(
	ctx context.Context,
	generation routingcontext.Generation,
	requestID string,
	facts accessruntime.DispatchFacts,
	model ModelIdentity,
) (preparedIdentity, error) {
	if authority == nil || authority.closed || authority.issuer == nil {
		return preparedIdentity{}, fmt.Errorf("routing-only dispatch authority is closed")
	}
	if !boundedIdentity(requestID) || !boundedIdentity(facts.DispatchID) || !validDigest(facts.DispatchPlanDigest) {
		return preparedIdentity{}, fmt.Errorf("routing-only dispatch identity is invalid")
	}
	if err := authority.validateGenerationLocked(ctx, generation, model); err != nil {
		return preparedIdentity{}, err
	}
	admissionID, admissionDigest := RoutingOnlyAdmissionIdentity(generation, requestID)
	return preparedIdentity{
		namespaceID: generation.NamespaceID, quotaPartition: generation.QuotaPartition,
		publicationID: generation.PublicationID, runtimeEpoch: generation.RuntimeEpoch,
		routingRevision: generation.SnapshotRevision, routingDigest: generation.RoutingDigest,
		admissionID: admissionID, admissionDigest: admissionDigest, requestID: requestID,
		dispatchID: facts.DispatchID, ordinal: facts.Ordinal,
		dispatchPlanDigest: facts.DispatchPlanDigest,
	}, nil
}

func (authority *RoutingOnlyAuthority) validateGenerationLocked(
	ctx context.Context,
	generation routingcontext.Generation,
	model ModelIdentity,
) error {
	if generation.Validate() != nil || generation.NamespaceID != authority.namespaceID || authority.publications == nil {
		return fmt.Errorf("routing-only generation is invalid")
	}
	current, ok := authority.publications.CurrentRoutingPublication(authority.namespaceID)
	if !ok || !current.Activated() || current.NamespaceID != generation.NamespaceID ||
		current.QuotaPartition != generation.QuotaPartition || current.PublicationID != generation.PublicationID ||
		current.RuntimeEpoch != generation.RuntimeEpoch || int64(current.DesiredRevision) != generation.SnapshotRevision ||
		current.RoutingDigest != generation.RoutingDigest {
		return fmt.Errorf("routing-only generation is not the active publication")
	}
	if authority.snapshots == nil {
		return fmt.Errorf("routing-only snapshot source is not attached")
	}
	snapshot, err := authority.snapshots.Snapshot(ctx, generation)
	if err != nil {
		return fmt.Errorf("load routing-only snapshot: %w", err)
	}
	if snapshot == nil || snapshot.NamespaceID != generation.NamespaceID ||
		snapshot.Revision != generation.SnapshotRevision {
		return fmt.Errorf("routing-only snapshot identity mismatch")
	}
	if model.ID != "" || model.Revision != 0 {
		selected, found := snapshot.Model(model.ID)
		if !found || selected.Revision != model.Revision {
			return fmt.Errorf("routing-only model revision is not in the active snapshot")
		}
	}
	return nil
}

func (authority *RoutingOnlyAuthority) Close() error {
	if authority == nil {
		return nil
	}
	authority.mu.Lock()
	defer authority.mu.Unlock()
	if authority.closed {
		return nil
	}
	authority.closed = true
	authority.publications = nil
	authority.snapshots = nil
	authority.identity = nil
	issuer := authority.issuer
	authority.issuer = nil
	authority.namespaceID = ""
	return issuer.Close()
}

// RoutingOnlyAdmissionIdentity returns the deterministic, non-secret
// accounting identity used by an explicitly public request. ExtProc uses the
// same value to construct the immutable terminal reference it may consume.
func RoutingOnlyAdmissionIdentity(generation routingcontext.Generation, requestID string) (string, string) {
	digest := sha256.New()
	for _, value := range []string{
		"vllm-sr/routing-only-admission/v1", generation.NamespaceID, generation.QuotaPartition,
		generation.PublicationID, fmt.Sprintf("%d", generation.RuntimeEpoch),
		fmt.Sprintf("%d", generation.SnapshotRevision), generation.RoutingDigest, requestID,
	} {
		var size [8]byte
		binary.BigEndian.PutUint64(size[:], uint64(len(value)))
		_, _ = digest.Write(size[:])
		_, _ = digest.Write([]byte(value))
	}
	encoded := hex.EncodeToString(digest.Sum(nil))
	return "public-" + encoded, encoded
}

func boundedIdentity(value string) bool {
	return value != "" && value == strings.TrimSpace(value) && len(value) <= 256 && !strings.ContainsRune(value, '\x00')
}

func validDigest(value string) bool {
	if len(value) != sha256.Size*2 || value != strings.ToLower(value) {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}
