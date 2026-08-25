package delegationmanagement

import (
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type SessionStatus string

const (
	SessionActive  SessionStatus = "active"
	SessionRevoked SessionStatus = "revoked"
	SessionExpired SessionStatus = "expired"
)

type Actor struct {
	PrincipalID         string
	ManagementSessionID string
	ActorChain          []string
	RequestID           string
	SourceIP            netip.Addr
}

type SelfServicePolicy struct {
	MaxDelegatedSessions   int
	DelegatedSessionTTL    time.Duration
	AllowTeamKeyDelegation bool
	Revision               uint64
}

type SelfContext struct {
	NamespaceID              string
	QuotaPartition           string
	PrincipalID              string
	ManagementSessionID      string
	ManagementSessionExpires time.Time
	UserID                   string
	Policy                   SelfServicePolicy
}

type EligibleKey struct {
	KeyID           string
	Name            string
	OwnerKind       accesscontrol.SubjectKind
	OwnerID         string
	ContextTeamID   string
	ExpiresAt       *time.Time
	DelegationEpoch uint64
	TeamID          string
	CreatedAt       time.Time
}

type Session struct {
	ID                  string
	PublicID            string
	NamespaceID         string
	QuotaPartition      string
	ManagementSessionID string
	PrincipalID         string
	APIKeyID            string
	DelegationEpoch     uint64
	UserID              string
	TeamID              string
	TokenHMAC           []byte
	PepperVersion       string
	Audience            string
	Status              SessionStatus
	NotBefore           time.Time
	ExpiresAt           time.Time
	RevokedAt           *time.Time
	Revision            uint64
	CreatedAt           time.Time
}

type Cursor struct {
	CreatedAt time.Time
	ID        string
}

type EligibleKeyQuery struct {
	NamespaceID         string
	PrincipalID         string
	ManagementSessionID string
	Search              string
	After               *Cursor
	Limit               int
}

type SessionQuery struct {
	NamespaceID string
	PrincipalID string
	APIKeyID    string
	After       *Cursor
	Limit       int
}

type Page[T any] struct {
	Items   []T
	HasMore bool
}

type ListRequest struct {
	NamespaceID         string
	PrincipalID         string
	ManagementSessionID string
	APIKeyID            string
	Search              string
	Cursor              string
	PageSize            int
}

type EligibleKeyRequest struct {
	NamespaceID         string
	PrincipalID         string
	ManagementSessionID string
	KeyID               string
}

type ResultPage[T any] struct {
	Items      []T
	NextCursor string
	HasMore    bool
	PageSize   int
}

type CreateRequest struct {
	NamespaceID    string
	KeyID          string
	IdempotencyKey string
	Actor          Actor
}

type RevokeRequest struct {
	NamespaceID string
	SessionID   string
	PrincipalID string
	APIKeyID    string
	Actor       Actor
}

type RevokeAllRequest struct {
	NamespaceID    string
	KeyID          string
	IdempotencyKey string
	Actor          Actor
}

type StoredSecret struct {
	Result          managementcommand.ResourceResult
	Secret          managementcommand.SecretResponse
	DesiredRevision uint64
}

type CreateMutation struct {
	Session           Session
	Command           managementcommand.Command
	Response          accesscredential.Envelope
	ResponseExpiresAt time.Time
	Actor             Actor
}

type MutationResult struct {
	Session         Session
	DesiredRevision uint64
	Replayed        bool
	Stored          *StoredSecret
}

type RevokeAllMutation struct {
	NamespaceID string
	KeyID       string
	Command     managementcommand.Command
	Actor       Actor
}

type RevokeAllResult struct {
	KeyID           string
	DelegationEpoch uint64
	DesiredRevision uint64
	QuotaPartition  string
	Replayed        bool
}

type SecretResult struct {
	Session       Session
	CanonicalJSON []byte
	Replayed      bool
}
