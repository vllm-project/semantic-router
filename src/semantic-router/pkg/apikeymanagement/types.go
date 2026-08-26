package apikeymanagement

import (
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

type Actor struct {
	PrincipalID string
	ActorChain  []string
	RequestID   string
	SourceIP    netip.Addr
}

type KeyPage struct {
	Items      []accesscontrol.APIKey
	NextCursor string
	HasMore    bool
	PageSize   int
	TotalCount *uint64
}

type CredentialMetadata struct {
	ID         string                         `json:"credentialId"`
	KeyID      string                         `json:"keyId"`
	KID        string                         `json:"kid"`
	Status     accesscontrol.CredentialStatus `json:"status"`
	Revealable bool                           `json:"revealable"`
	NotBefore  time.Time                      `json:"notBefore"`
	ExpiresAt  *time.Time                     `json:"expiresAt,omitempty"`
	RevokedAt  *time.Time                     `json:"revokedAt,omitempty"`
	CreatedAt  time.Time                      `json:"createdAt"`
}

type CredentialPage struct {
	Items      []CredentialMetadata
	NextCursor string
	HasMore    bool
	PageSize   int
}

type Owner struct {
	Kind accesscontrol.SubjectKind `json:"type"`
	ID   string                    `json:"id"`
}

type KeyMetadata struct {
	ID            string                     `json:"keyId"`
	Name          string                     `json:"name"`
	Owner         Owner                      `json:"owner"`
	ContextTeamID string                     `json:"contextTeamId,omitempty"`
	Status        accesscontrol.APIKeyStatus `json:"status"`
	ExpiresAt     *time.Time                 `json:"expiresAt,omitempty"`
	LastUsedAt    *time.Time                 `json:"lastUsedAt,omitempty"`
	Revision      uint64                     `json:"revision"`
	CreatedAt     time.Time                  `json:"createdAt"`
	UpdatedAt     time.Time                  `json:"updatedAt"`
	DeletedAt     *time.Time                 `json:"deletedAt,omitempty"`
}

// PolicyBindingReceipt is an immutable reference to a binding created in the
// same transaction as an API key. Mutable policy contents are deliberately not
// copied into the one-time response.
type PolicyBindingReceipt struct {
	PolicyID  string `json:"policyId"`
	BindingID string `json:"bindingId"`
}

type RateLimitOverrideReceipt struct {
	PolicyID  string `json:"policyId"`
	BindingID string `json:"bindingId"`
	Created   bool   `json:"created"`
}

// IssuedSecret is the canonical one-time response body encrypted in the
// Management command record. Policy receipts are immutable identities rather
// than snapshots that could become stale after issuance.
type IssuedSecret struct {
	Data                 KeyMetadata               `json:"data"`
	Credential           CredentialMetadata        `json:"credential"`
	Secret               string                    `json:"secret"`
	AccessPolicyBindings []PolicyBindingReceipt    `json:"accessPolicyBindings,omitempty"`
	RateLimitOverride    *RateLimitOverrideReceipt `json:"rateLimitOverride,omitempty"`
	DeliveryExpiresAt    time.Time                 `json:"deliveryExpiresAt"`
}

func keyMetadata(key accesscontrol.APIKey) KeyMetadata {
	return KeyMetadata{
		ID: string(key.ID), Name: key.Name,
		Owner:         Owner{Kind: key.Owner.Kind, ID: string(key.Owner.ID)},
		ContextTeamID: string(key.ContextTeamID), Status: key.Status,
		ExpiresAt: cloneTime(key.ExpiresAt), LastUsedAt: cloneTime(key.LastUsedAt),
		Revision: uint64(key.Revision), CreatedAt: key.CreatedAt, UpdatedAt: key.UpdatedAt,
		DeletedAt: cloneTime(key.DeletedAt),
	}
}

func cloneTime(value *time.Time) *time.Time {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

type SecretMutationResult struct {
	Key                  accesscontrol.APIKey
	Credential           CredentialMetadata
	Secret               string
	AccessPolicyBindings []PolicyBindingReceipt
	RateLimitOverride    *RateLimitOverrideReceipt
	CanonicalJSON        []byte
	ResponseRevision     uint64
	Replayed             bool
}

type MutationResult struct {
	Key        accesscontrol.APIKey
	HTTPStatus int
	Replayed   bool
	Stored     *StoredSecret
}

type ListKeysRequest struct {
	NamespaceID  string
	Status       accesscontrol.APIKeyStatus
	OwnerKind    accesscontrol.SubjectKind
	OwnerID      string
	Search       string
	Cursor       string
	PageSize     int
	IncludeTotal bool
	Scope        accesscontrol.ResultScope
}

type ListCredentialsRequest struct {
	NamespaceID string
	KeyID       string
	Status      accesscontrol.CredentialStatus
	Cursor      string
	PageSize    int
}

type CreateRequest struct {
	NamespaceID       string
	Name              string
	Owner             Owner
	ContextTeamID     string
	ExpiresAt         *time.Time
	Revealable        *bool
	AccessPolicyIDs   []string
	RateLimitOverride *RateLimitOverrideInput
	IdempotencyKey    string
	Actor             Actor
}

type InlineRateLimitPolicyInput struct {
	Name        string                           `json:"name"`
	Description string                           `json:"description,omitempty"`
	Rules       []policymanagement.RateLimitRule `json:"rules"`
}

// RateLimitOverrideInput is an explicit sum type. Exactly one of PolicyID and
// InlinePolicy must be present when the override itself is supplied.
type RateLimitOverrideInput struct {
	PolicyID     string                      `json:"policyId,omitempty"`
	InlinePolicy *InlineRateLimitPolicyInput `json:"inlinePolicy,omitempty"`
}

type RenameRequest struct {
	NamespaceID      string
	KeyID            string
	ExpectedRevision uint64
	Name             string
	Actor            Actor
}

type LifecycleRequest struct {
	NamespaceID      string
	KeyID            string
	ExpectedRevision uint64
	IdempotencyKey   string
	Actor            Actor
}

type RenewRequest struct {
	NamespaceID      string
	KeyID            string
	ExpectedRevision uint64
	ExpiresAt        *time.Time
	IdempotencyKey   string
	Actor            Actor
}

type ReassignRequest struct {
	NamespaceID      string
	KeyID            string
	ExpectedRevision uint64
	Owner            Owner
	ContextTeamID    string
	IdempotencyKey   string
	Actor            Actor
}

type RotateRequest struct {
	NamespaceID      string
	KeyID            string
	ExpectedRevision uint64
	Overlap          time.Duration
	Revealable       *bool
	IdempotencyKey   string
	Actor            Actor
}

type RevokeCredentialRequest struct {
	NamespaceID      string
	KeyID            string
	CredentialID     string
	ExpectedRevision uint64
	Actor            Actor
}

type RevealRequest struct {
	NamespaceID  string
	KeyID        string
	CredentialID string
	Actor        Actor
}

type KeyCursor struct {
	CreatedAt time.Time
	ID        string
}

type CredentialCursor struct {
	CreatedAt time.Time
	ID        string
}

type KeyQuery struct {
	NamespaceID  string
	Status       accesscontrol.APIKeyStatus
	OwnerKind    accesscontrol.SubjectKind
	OwnerID      string
	Search       string
	Scope        accesscontrol.ResultScope
	After        *KeyCursor
	Limit        int
	IncludeTotal bool
}

type CredentialQuery struct {
	NamespaceID string
	KeyID       string
	Status      accesscontrol.CredentialStatus
	After       *CredentialCursor
	Limit       int
}

type RepositoryPage[T any] struct {
	Items      []T
	HasMore    bool
	TotalCount *uint64
}

type StoredSecret struct {
	Result managementcommand.ResourceResult
	Secret managementcommand.SecretResponse
}

type CreateMutation struct {
	Key               accesscontrol.APIKey
	Credential        accesscontrol.CredentialVersion
	AccessBindings    []policymanagement.AccessPolicyBinding
	RateLimitOverride *RateLimitOverrideMutation
	Command           managementcommand.Command
	Response          accesscredential.Envelope
	ResponseExpiresAt time.Time
	Actor             Actor
}

type RateLimitOverrideMutation struct {
	// PolicyID and InlinePolicy are a strict one-of. Binding.PolicyID always
	// carries the resolved policy ID used by the allocation.
	PolicyID     string
	InlinePolicy *policymanagement.RateLimitPolicy
	Binding      policymanagement.RateLimitBinding
}

type RotateMutation struct {
	NamespaceID          string
	KeyID                string
	ExpectedRevision     uint64
	Credential           accesscontrol.CredentialVersion
	PreviousCredentialID string
	RetireAt             *time.Time
	Command              managementcommand.Command
	Response             accesscredential.Envelope
	ResponseExpiresAt    time.Time
	Actor                Actor
}

type UpdateMutation struct {
	Key              accesscontrol.APIKey
	ExpectedRevision uint64
	Command          managementcommand.Command
	Actor            Actor
	Action           string
	Reason           string
}

type RevealSnapshot struct {
	NamespaceID string
	Credential  accesscontrol.CredentialVersion
}
