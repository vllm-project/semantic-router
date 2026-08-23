package managementidentity

import (
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type ServiceAccountOwnerScope string

const (
	ServiceAccountOwnerCluster   ServiceAccountOwnerScope = "cluster"
	ServiceAccountOwnerNamespace ServiceAccountOwnerScope = "namespace"
)

type WorkloadClass string

const (
	WorkloadStandard WorkloadClass = "workload_standard"
	WorkloadStrong   WorkloadClass = "workload_strong"
)

type ServiceAccountStatus string

const (
	ServiceAccountActive   ServiceAccountStatus = "active"
	ServiceAccountDisabled ServiceAccountStatus = "disabled"
)

type ServiceCredentialStatus string

const (
	ServiceCredentialActive   ServiceCredentialStatus = "active"
	ServiceCredentialRetiring ServiceCredentialStatus = "retiring"
	ServiceCredentialRevoked  ServiceCredentialStatus = "revoked"
)

type MTLSMatcherKind string

const (
	MTLSMatcherSPIFFEID        MTLSMatcherKind = "spiffe_id"
	MTLSMatcherSANURI          MTLSMatcherKind = "san_uri"
	MTLSMatcherSANDNS          MTLSMatcherKind = "san_dns"
	MTLSMatcherSubjectDNDigest MTLSMatcherKind = "subject_dn_sha256"
)

const ServiceAccountIssuer = "urn:vllm-sr:service-account"

type ServiceAccount struct {
	ID          string                   `json:"id"`
	PrincipalID string                   `json:"principalId"`
	DisplayName string                   `json:"displayName"`
	OwnerScope  ServiceAccountOwnerScope `json:"ownerScope"`
	NamespaceID string                   `json:"namespaceId,omitempty"`
	Status      ServiceAccountStatus     `json:"status"`
	Revision    uint64                   `json:"revision"`
	CreatedAt   time.Time                `json:"createdAt"`
	UpdatedAt   time.Time                `json:"updatedAt"`
}

type ServiceCredential struct {
	ID               string                  `json:"id"`
	ServiceAccountID string                  `json:"serviceAccountId"`
	PublicID         string                  `json:"publicId"`
	WorkloadClass    WorkloadClass           `json:"workloadClass"`
	SourceAssuredAt  time.Time               `json:"sourceAssuredAt"`
	Status           ServiceCredentialStatus `json:"status"`
	NotBefore        time.Time               `json:"notBefore"`
	ExpiresAt        time.Time               `json:"expiresAt"`
	RevokedAt        *time.Time              `json:"revokedAt,omitempty"`
	CreatedAt        time.Time               `json:"createdAt"`
}

type MTLSIdentityMapping struct {
	ID              string
	MatcherKind     MTLSMatcherKind
	MatcherValue    string
	PrincipalID     string
	WorkloadClass   WorkloadClass
	SourceAssuredAt time.Time
	Status          managementauth.ResourceStatus
	Revision        uint64
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

type WorkloadActor struct {
	PrincipalID string
	ActorChain  []string
	RequestID   string
	SourceIP    netip.Addr
	Reason      string
	Session     managementauth.LiveSession
}

func (actor WorkloadActor) MutationActor() MutationActor {
	return MutationActor{
		PrincipalID: actor.PrincipalID, ActorChain: append([]string(nil), actor.ActorChain...),
		RequestID: actor.RequestID, SourceIP: actor.SourceIP, Reason: actor.Reason,
	}
}

// ServiceAccountResultScope is the repository visibility envelope for a list.
// Cluster authority is represented explicitly; namespace authority carries the
// canonical ResultScope produced by Management authorization.
type ServiceAccountResultScope struct {
	Cluster     bool
	NamespaceID string
	All         bool
	IDs         []string
}

type ServiceAccountListRequest struct {
	Scope    ServiceAccountResultScope
	Status   ServiceAccountStatus
	Cursor   string
	PageSize int
}

type ServiceCredentialListRequest struct {
	ServiceAccountID string
	Cursor           string
	PageSize         int
}

type MTLSMappingListRequest struct {
	Status   managementauth.ResourceStatus
	Cursor   string
	PageSize int
}

type WorkloadPage[T any] struct {
	Items      []T
	NextCursor string
	HasMore    bool
	PageSize   int
}

type CreateServiceAccountRequest struct {
	DisplayName         string
	OwnerScope          ServiceAccountOwnerScope
	NamespaceID         string
	CredentialExpiresAt time.Time
	CredentialClass     WorkloadClass
	IdempotencyKey      string
	Actor               WorkloadActor
}

type PatchServiceAccountRequest struct {
	ID               string
	ExpectedRevision uint64
	DisplayName      *string
	Status           *ServiceAccountStatus
	Actor            WorkloadActor
}

type DeleteServiceAccountRequest struct {
	ID               string
	ExpectedRevision uint64
	Actor            WorkloadActor
}

type RotateServiceCredentialRequest struct {
	ServiceAccountID string
	ExpectedRevision uint64
	ExpiresAt        time.Time
	WorkloadClass    WorkloadClass
	Overlap          time.Duration
	IdempotencyKey   string
	Actor            WorkloadActor
}

type RevokeServiceCredentialRequest struct {
	ServiceAccountID string
	CredentialID     string
	ExpectedRevision uint64
	Actor            WorkloadActor
}

type CreateMTLSMappingRequest struct {
	MatcherKind    MTLSMatcherKind
	MatcherValue   string
	PrincipalID    string
	WorkloadClass  WorkloadClass
	IdempotencyKey string
	Actor          WorkloadActor
}

type PatchMTLSMappingRequest struct {
	ID               string
	ExpectedRevision uint64
	Status           *managementauth.ResourceStatus
	WorkloadClass    *WorkloadClass
	Actor            WorkloadActor
}

type DeleteMTLSMappingRequest struct {
	ID               string
	ExpectedRevision uint64
	Actor            WorkloadActor
}

type ServiceCredentialSecret struct {
	ServiceAccount ServiceAccount    `json:"serviceAccount"`
	Credential     ServiceCredential `json:"credential"`
	Secret         string            `json:"secret"`
	DeliveryExpiry time.Time         `json:"deliveryExpiresAt"`
}

type ServiceCredentialSecretResult struct {
	ServiceAccount ServiceAccount
	Credential     ServiceCredential
	Secret         string
	DeliveryExpiry time.Time
	Replayed       bool
}

type StoredWorkloadSecret struct {
	Result managementcommand.ResourceResult
	Secret managementcommand.SecretResponse
}

type ServiceAccountCreateMutation struct {
	Account           ServiceAccount
	Credential        ServiceCredential
	SecretHMAC        []byte
	PepperVersion     string
	Command           managementcommand.Command
	Response          accesscredential.Envelope
	ResponseExpiresAt time.Time
	Actor             MutationActor
}

type ServiceCredentialRotateMutation struct {
	AccountID         string
	ExpectedRevision  uint64
	Credential        ServiceCredential
	SecretHMAC        []byte
	PepperVersion     string
	RetireAt          time.Time
	Command           managementcommand.Command
	Response          accesscredential.Envelope
	ResponseExpiresAt time.Time
	Actor             MutationActor
}

type WorkloadMutationResult struct {
	Kind                 string
	ID                   string
	Revision             uint64
	HTTPStatus           int
	Replayed             bool
	Stored               *StoredWorkloadSecret
	SessionIDs           []string
	RevokedCredentialIDs []string
}

type MTLSMappingCreateMutation struct {
	Mapping MTLSIdentityMapping
	Command managementcommand.Command
	Actor   MutationActor
}
