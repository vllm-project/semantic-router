package invitationmanagement

import (
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type Status string

const (
	StatusPending  Status = "pending"
	StatusAccepted Status = "accepted"
	StatusExpired  Status = "expired"
	StatusRevoked  Status = "revoked"
)

func (status Status) Valid() bool {
	switch status {
	case StatusPending, StatusAccepted, StatusExpired, StatusRevoked:
		return true
	default:
		return false
	}
}

type Actor struct {
	PrincipalID string
	ActorChain  []string
	RequestID   string
	SourceIP    netip.Addr
	Reason      string
}

type ExpectedIdentity struct {
	Issuer  string `json:"issuer"`
	Subject string `json:"subject,omitempty"`
	Email   string `json:"email,omitempty"`
}

// RoleGrant is the immutable Management authority materialized when an
// invitation is accepted. Source* pins the one delegation source that
// authorized this grant; permissions from several sources are never combined.
type RoleGrant struct {
	RoleID                  string   `json:"roleId"`
	RoleRevision            uint64   `json:"roleRevision"`
	RolePermissionsDigest   string   `json:"rolePermissionsDigest"`
	ScopeKind               string   `json:"scopeKind"`
	DelegationCeiling       []string `json:"delegationCeiling"`
	SourceBindingID         string   `json:"sourceBindingId"`
	SourceBindingRevision   uint64   `json:"sourceBindingRevision"`
	SourceRoleID            string   `json:"sourceRoleId"`
	SourcePermissionsDigest string   `json:"sourcePermissionsDigest"`
}

type RequestedRoleGrant struct {
	RoleID            string   `json:"roleId"`
	ScopeKind         string   `json:"scopeKind"`
	DelegationCeiling []string `json:"delegationCeiling,omitempty"`
}

type TeamAssignment struct {
	TeamID string                 `json:"teamId"`
	Role   accesscontrol.TeamRole `json:"role"`
}

type OnboardingSnapshot struct {
	RoleGrants                []RoleGrant     `json:"roleGrants"`
	Team                      *TeamAssignment `json:"team,omitempty"`
	SelfServicePolicyRevision uint64          `json:"selfServicePolicyRevision"`
	AccessPolicyID            string          `json:"accessPolicyId"`
	AccessPolicyRevision      uint64          `json:"accessPolicyRevision"`
	RateLimitPolicyID         string          `json:"rateLimitPolicyId"`
	RateLimitPolicyRevision   uint64          `json:"rateLimitPolicyRevision"`
	AutomaticFirstKey         bool            `json:"automaticFirstKey"`
}

type Invitation struct {
	ID                          string             `json:"invitationId"`
	NamespaceID                 string             `json:"namespaceId"`
	CreatedByPrincipalID        string             `json:"createdByPrincipalId"`
	Expected                    ExpectedIdentity   `json:"expectedIdentity"`
	DisplayName                 string             `json:"displayName"`
	Snapshot                    OnboardingSnapshot `json:"onboarding"`
	ExpiresAt                   time.Time          `json:"expiresAt"`
	Status                      Status             `json:"status"`
	AcceptedPrincipalID         string             `json:"acceptedPrincipalId,omitempty"`
	AcceptedUserID              string             `json:"acceptedUserId,omitempty"`
	AcceptedManagementSessionID string             `json:"acceptedManagementSessionId,omitempty"`
	AcceptedAt                  *time.Time         `json:"acceptedAt,omitempty"`
	Revision                    uint64             `json:"revision"`
	CreatedAt                   time.Time          `json:"createdAt"`
	UpdatedAt                   time.Time          `json:"updatedAt"`
}

func (invitation Invitation) EffectiveStatus(now time.Time) Status {
	if invitation.Status == StatusPending && !now.Before(invitation.ExpiresAt) {
		return StatusExpired
	}
	return invitation.Status
}

type Page struct {
	Items      []Invitation
	NextCursor string
	HasMore    bool
	PageSize   int
}

type ListRequest struct {
	NamespaceID string
	Status      Status
	Cursor      string
	PageSize    int
}

type InvitationCursor struct {
	ExpiresAt time.Time
	ID        string
}

type InvitationQuery struct {
	NamespaceID string
	Status      Status
	After       *InvitationCursor
	Limit       int
	Now         time.Time
}

type RepositoryPage struct {
	Items   []Invitation
	HasMore bool
}

type CreateRequest struct {
	NamespaceID    string
	Expected       ExpectedIdentity
	DisplayName    string
	RoleGrants     []RequestedRoleGrant
	Team           *TeamAssignment
	ExpiresAt      time.Time
	IdempotencyKey string
	Actor          Actor
}

type RotateRequest struct {
	NamespaceID      string
	InvitationID     string
	ExpectedRevision uint64
	ExpiresAt        *time.Time
	IdempotencyKey   string
	Actor            Actor
}

type RevokeRequest struct {
	NamespaceID      string
	InvitationID     string
	ExpectedRevision uint64
	Actor            Actor
}

type CreateMutation struct {
	Invitation        Invitation
	Requested         []RequestedRoleGrant
	Team              *TeamAssignment
	Command           managementcommand.Command
	TokenHMAC         []byte
	PepperVersion     string
	Response          accesscredential.Envelope
	ResponseExpiresAt time.Time
	Actor             Actor
}

type RotateMutation struct {
	NamespaceID       string
	InvitationID      string
	ExpectedRevision  uint64
	ExpiresAt         *time.Time
	Command           managementcommand.Command
	TokenHMAC         []byte
	PepperVersion     string
	Response          accesscredential.Envelope
	ResponseExpiresAt time.Time
	Actor             Actor
}

type MutationResult struct {
	Invitation Invitation
	HTTPStatus int
	Replayed   bool
	Stored     *StoredSecret
}

type StoredSecret struct {
	Result managementcommand.ResourceResult
	Secret managementcommand.SecretResponse
}

type SecretResult struct {
	Invitation    Invitation `json:"-"`
	Token         string
	CanonicalJSON []byte
	Replayed      bool
}

type AcceptanceIdentity struct {
	Issuer        string
	Subject       string
	VerifiedEmail string
	DisplayName   string
}

type FirstKeyRequest struct {
	NamespaceID   string
	UserID        string
	ContextTeamID string
	Name          string
	Now           time.Time
}

// PreparedFirstKey is secret-bearing transient transaction input. Plaintext is
// encrypted only in the invitation acceptance result and never persisted as a
// column, audit field, outbox payload, log, or trace.
type PreparedFirstKey struct {
	Key        accesscontrol.APIKey
	Credential accesscontrol.CredentialVersion
	Plaintext  []byte
}

type FirstKeyPreparer interface {
	PrepareFirstKey(FirstKeyRequest) (PreparedFirstKey, error)
	Close()
}

type AcceptRequest struct {
	Token                    string
	Identity                 AcceptanceIdentity
	AuthenticationSourceKind string
	AuthenticationSourceID   string
	EvidenceKind             string
	RequestID                string
	SourceIP                 netip.Addr
}

type AcceptMutation struct {
	InvitationID             string
	TokenHMAC                []byte
	PepperVersion            string
	Identity                 AcceptanceIdentity
	PrincipalID              string
	UserID                   string
	RoleBindingIDs           []string
	AccessBindingID          string
	RateLimitBindingID       string
	FirstKey                 *PreparedFirstKey
	SealResult               func(AcceptanceResult) (accesscredential.Envelope, time.Time, error)
	AuthenticationSourceKind string
	AuthenticationSourceID   string
	EvidenceKind             string
	Actor                    Actor
}

type AcceptanceResult struct {
	InvitationID      string    `json:"invitationId"`
	PrincipalID       string    `json:"principalId"`
	UserID            string    `json:"userId"`
	TeamID            string    `json:"teamId,omitempty"`
	APIKeyID          string    `json:"apiKeyId,omitempty"`
	APIKey            string    `json:"apiKey,omitempty"`
	DeliveryExpiresAt time.Time `json:"deliveryExpiresAt"`
}

type Accepted struct {
	Invitation    Invitation
	Result        AcceptanceResult
	CanonicalJSON []byte
	Replayed      bool
}

type AcceptanceEnvelope struct {
	Invitation Invitation
	Envelope   accesscredential.Envelope
	ExpiresAt  time.Time
	Replayed   bool
}

type PrivilegedOnboardingRequest struct {
	NamespaceID    string
	PrincipalID    string
	Email          string
	DisplayName    string
	RoleGrants     []RequestedRoleGrant
	Team           *TeamAssignment
	CreateFirstKey bool
	IdempotencyKey string
	Actor          Actor
	// PreparedSnapshot is an internal control-plane handoff. HTTP adapters use
	// the same pinned defaults to authorize exact policy targets and to execute
	// the mutation, eliminating a resolve/authorize/resolve race.
	PreparedSnapshot *OnboardingSnapshot
}

type PrivilegedOnboardingMutation struct {
	NamespaceID        string
	PrincipalID        string
	UserID             string
	Email              string
	DisplayName        string
	Snapshot           OnboardingSnapshot
	RoleBindingIDs     []string
	AccessBindingID    string
	RateLimitBindingID string
	FirstKey           *PreparedFirstKey
	Command            managementcommand.Command
	SealResult         func(AcceptanceResult) (accesscredential.Envelope, time.Time, error)
	Actor              Actor
}

type PrivilegedOnboardingResult struct {
	Result        AcceptanceResult
	CanonicalJSON []byte
	Replayed      bool
}
