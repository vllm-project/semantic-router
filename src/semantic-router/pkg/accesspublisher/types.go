package accesspublisher

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

var (
	ErrNoWork             = errors.New("no access publication work")
	ErrConflict           = errors.New("access publication compare-and-set conflict")
	ErrSuperseded         = errors.New("access publication desired revision was superseded")
	ErrEpochMismatch      = errors.New("access publication runtime epoch mismatch")
	ErrStagedCorrupt      = errors.New("access publication staged value is corrupt")
	ErrAcknowledgements   = errors.New("access publication acknowledgements are incomplete")
	ErrNotReady           = errors.New("access publication runtime is not ready")
	ErrPublicationChanged = errors.New("access publication head changed")
	ErrDirectoryFull      = errors.New("access publication namespace directory is full")
	ErrNamespaceNotFound  = errors.New("access publication namespace is not registered")
)

const (
	CredentialKindAPIKey     = "api_key"
	CredentialKindDelegation = "delegation"
)

// OutboxBatch is one coalesced namespace publication claim. RowIDs includes
// every earlier non-applied row that becomes superseded only if DesiredRevision
// is fully applied.
type OutboxBatch struct {
	NamespaceID     string
	DesiredRevision uint64
	RuntimeEpoch    uint64
	QuotaPartition  string
	RowIDs          []string
	WorkerID        string
	ClaimedAt       time.Time
}

func (b OutboxBatch) Validate() error {
	if strings.TrimSpace(b.NamespaceID) == "" || b.DesiredRevision == 0 || b.RuntimeEpoch == 0 {
		return fmt.Errorf("namespace, desired revision, and runtime epoch are required")
	}
	if strings.TrimSpace(b.QuotaPartition) == "" || strings.TrimSpace(b.WorkerID) == "" || b.ClaimedAt.IsZero() {
		return fmt.Errorf("quota partition, worker, and claim time are required")
	}
	if len(b.RowIDs) == 0 {
		return fmt.Errorf("outbox batch must contain at least one row")
	}
	seen := make(map[string]struct{}, len(b.RowIDs))
	for _, id := range b.RowIDs {
		if strings.TrimSpace(id) == "" {
			return fmt.Errorf("outbox row id is required")
		}
		if _, exists := seen[id]; exists {
			return fmt.Errorf("outbox row %q is duplicated", id)
		}
		seen[id] = struct{}{}
	}
	return nil
}

// DesiredState is loaded in one PostgreSQL repeatable-read transaction. The
// reader must verify that Revision is still the namespace's latest committed
// desired revision before returning.
type DesiredState struct {
	Namespace           accesscontrol.Namespace
	Revision            uint64
	RevisionTime        time.Time
	Keys                []accessprojection.Candidate
	Credentials         []CredentialCandidate
	ProviderCredentials []ProviderCredentialCandidate
	Routing             routingsnapshot.Bundle
	BarrierHints        []Barrier
}

type CredentialCandidate struct {
	Kind       string
	Credential accesscontrol.CredentialVersion
	Delegation *accessprojection.DelegationContext
}

// ProviderCredentialCandidate is the complete encrypted runtime image for one
// ProviderCredential referenced by the routing bundle. Versions contains the
// active version and every still-valid retiring version, never plaintext.
type ProviderCredentialCandidate struct {
	Credential providercredential.Credential
	Versions   []providercredential.Version
}

type AccessDocument struct {
	NamespaceID     string                      `json:"namespaceId"`
	QuotaPartition  string                      `json:"quotaPartition"`
	DesiredRevision uint64                      `json:"desiredRevision"`
	KeyID           string                      `json:"keyId"`
	Projection      accessprojection.Projection `json:"projection"`
	Digest          string                      `json:"digest"`
}

type CredentialDocument struct {
	NamespaceID     string                                `json:"namespaceId"`
	QuotaPartition  string                                `json:"quotaPartition"`
	DesiredRevision uint64                                `json:"desiredRevision"`
	Kind            string                                `json:"kind"`
	PublicID        string                                `json:"publicId"`
	Projection      accessprojection.CredentialProjection `json:"projection"`
	Digest          string                                `json:"digest"`
}

// ProviderCredentialDocument is immutable within one coupled publication.
// Credential contains binding metadata; Versions contains envelope-encrypted
// material only. The publication ID is part of the Valkey key rather than this
// canonical payload, so the publication digest remains content addressed.
type ProviderCredentialDocument struct {
	NamespaceID     string                        `json:"namespaceId"`
	QuotaPartition  string                        `json:"quotaPartition"`
	DesiredRevision uint64                        `json:"desiredRevision"`
	Credential      providercredential.Credential `json:"credential"`
	Versions        []providercredential.Version  `json:"versions"`
	Digest          string                        `json:"digest"`
}

type RoutingDocument struct {
	NamespaceID     string                   `json:"namespaceId"`
	DesiredRevision uint64                   `json:"desiredRevision"`
	Snapshot        routingsnapshot.Snapshot `json:"snapshot"`
	ResourceDigests map[string]string        `json:"resourceDigests"`
	Digest          string                   `json:"digest"`
}

type ManifestEntry struct {
	Revision uint64 `json:"revision"`
	Digest   string `json:"digest"`
}

type Manifest struct {
	NamespaceID         string                   `json:"namespaceId"`
	QuotaPartition      string                   `json:"quotaPartition"`
	DesiredRevision     uint64                   `json:"desiredRevision"`
	RuntimeEpoch        uint64                   `json:"runtimeEpoch"`
	PublicationID       string                   `json:"publicationId"`
	Access              map[string]ManifestEntry `json:"access"`
	Credentials         map[string]ManifestEntry `json:"credentials"`
	ProviderCredentials map[string]ManifestEntry `json:"providerCredentials"`
	RoutingDigest       string                   `json:"routingDigest"`
	RoutingResources    map[string]string        `json:"routingResources"`
	Digest              string                   `json:"digest"`
}

// Publication is a complete, immutable namespace candidate. Every map/slice
// is sorted or canonicalized by Compile before Digest and ID are assigned.
type Publication struct {
	ID                  string
	NamespaceID         string
	QuotaPartition      string
	DesiredRevision     uint64
	RuntimeEpoch        uint64
	Digest              string
	Access              []AccessDocument
	Credentials         []CredentialDocument
	ProviderCredentials []ProviderCredentialDocument
	Routing             RoutingDocument
	Manifest            Manifest
	BarrierHints        []Barrier
}

func (p Publication) Validate() error {
	if strings.TrimSpace(p.ID) == "" || strings.TrimSpace(p.NamespaceID) == "" ||
		strings.TrimSpace(p.QuotaPartition) == "" || p.DesiredRevision == 0 || p.RuntimeEpoch == 0 {
		return fmt.Errorf("publication identity is incomplete")
	}
	if !validDigest(p.Digest) || p.Manifest.PublicationID != p.ID || p.Manifest.Digest == "" {
		return fmt.Errorf("publication digest or manifest identity is invalid")
	}
	return nil
}

type Barrier struct {
	Kind       string `json:"kind"`
	ResourceID string `json:"resourceId"`
	Reason     string `json:"reason"`
}

func (b Barrier) Validate() error {
	if strings.TrimSpace(b.Kind) == "" || strings.TrimSpace(b.ResourceID) == "" || strings.TrimSpace(b.Reason) == "" {
		return fmt.Errorf("barrier kind, resource, and reason are required")
	}
	if len(b.Kind) > 64 || len(b.ResourceID) > 512 || len(b.Reason) > 128 {
		return fmt.Errorf("barrier field exceeds its bound")
	}
	return nil
}

func barrierKey(barrier Barrier) string {
	return barrier.Kind + "\x00" + barrier.ResourceID
}

func canonicalBarriers(input []Barrier) ([]Barrier, error) {
	byResource := make(map[string]Barrier, len(input))
	for _, barrier := range input {
		if err := barrier.Validate(); err != nil {
			return nil, err
		}
		key := barrierKey(barrier)
		if existing, ok := byResource[key]; !ok || barrier.Reason < existing.Reason {
			byResource[key] = barrier
		}
	}
	result := make([]Barrier, 0, len(byResource))
	for _, barrier := range byResource {
		result = append(result, barrier)
	}
	sort.Slice(result, func(i, j int) bool { return barrierKey(result[i]) < barrierKey(result[j]) })
	return result, nil
}

type AckStatus struct {
	Required []string
	Missing  []string
}

func (a AckStatus) Complete() bool { return len(a.Missing) == 0 }

type AppliedState struct {
	NamespaceID     string
	QuotaPartition  string
	RuntimeEpoch    uint64
	DesiredRevision uint64
	PublicationID   string
	AccessDigest    string
	RoutingDigest   string
}

type Readiness struct {
	Ready           bool
	Reason          string
	RuntimeEpoch    uint64
	DesiredRevision uint64
	AppliedRevision uint64
	AccessGate      string
	RoutingGate     string
	ProjectorLag    uint64
}

// ReplicaRegistration is the loaded publication identity a data-plane replica
// presents when joining or renewing the namespace lease. Registration never
// acknowledges a candidate publication; those acknowledgements are explicit
// after the replica has installed the corresponding deny barriers or routing
// snapshot.
type ReplicaRegistration struct {
	ReplicaID          string
	RuntimeEpoch       uint64
	AccessPublication  string
	RoutingPublication string
}

func (r ReplicaRegistration) Validate() error {
	if strings.TrimSpace(r.ReplicaID) == "" || len(r.ReplicaID) > 256 || r.RuntimeEpoch == 0 {
		return fmt.Errorf("replica id and runtime epoch are required")
	}
	if strings.ContainsRune(r.ReplicaID, 0) {
		return fmt.Errorf("replica id contains NUL")
	}
	return nil
}

// DesiredStateReader is intentionally independent of management repositories:
// publication needs one complete repeatable-read image rather than a sequence
// of independently consistent aggregate reads.
type DesiredStateReader interface {
	LoadDesiredState(context.Context, string, uint64) (DesiredState, error)
}

type OutboxStore interface {
	ClaimLatest(context.Context, string, time.Duration) (OutboxBatch, error)
	RecordStaged(context.Context, OutboxBatch, Publication) error
	Release(context.Context, OutboxBatch, error, time.Duration) error
	Fail(context.Context, OutboxBatch, error) error
	WithRevisionFence(context.Context, OutboxBatch, func(context.Context) error) error
	Applied(context.Context, string) (AppliedState, error)
}

type RuntimeStore interface {
	Prepare(context.Context, Publication) (PublicationPlan, error)
	InstallBarriers(context.Context, PublicationPlan) error
	Stage(context.Context, PublicationPlan) error
	ValidateStaged(context.Context, PublicationPlan) error
	BarrierAcknowledgements(context.Context, PublicationPlan) (AckStatus, error)
	RoutingAcknowledgements(context.Context, PublicationPlan) (AckStatus, error)
	Activate(context.Context, PublicationPlan) error
	Compact(context.Context, PublicationPlan, int) (bool, error)
	MarkApplied(context.Context, PublicationPlan) error
	ClearAppliedBarriers(context.Context, PublicationPlan) error
	ReconcileApplied(context.Context, AppliedState) error
	Readiness(context.Context, string, string) (Readiness, error)
}

type PublicationPlan struct {
	Publication      Publication
	Previous         *Manifest
	Barriers         []Barrier
	Supersedes       []string
	PriorAccessGate  string
	PriorRoutingGate string
}

func (p PublicationPlan) Restrictive() bool { return len(p.Barriers) > 0 }

func validDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, char := range value {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			return false
		}
	}
	return true
}
