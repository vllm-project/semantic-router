// Package namespacemanagement owns the Namespace aggregate and its mandatory
// control-plane policies. It deliberately has no HTTP or PostgreSQL concerns.
package namespacemanagement

import (
	"context"
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

const SecurityPolicySeedVersion uint64 = 2

const (
	ActionUnknownUsageFenceWaive = "unknown_usage_fence.waive"
	ActionSecurityPolicyLoosen   = "management_security_policy.loosen"
	ActionSecretDeliver          = "key.secret_deliver"
	ActionSecretReveal           = "key.secret_reveal"
	ActionRoleDelegate           = "role.delegate"
)

type Namespace struct {
	ID               string
	Name             string
	QuotaPartitionID string
	BillingCurrency  string
	Status           accesscontrol.NamespaceStatus
	Revision         uint64
	RuntimeEpoch     uint64
	CreatedAt        time.Time
	UpdatedAt        time.Time
}

type SelfServicePolicy struct {
	NamespaceID              string
	MaxKeysPerUser           int
	MaxDelegatedSessions     int
	DelegatedSessionTTL      time.Duration
	AllowTeamKeyDelegation   bool
	AutomaticFirstKey        bool
	TeamAdminCapabilities    []accesscontrol.TeamAdminCapability
	DefaultAccessPolicyID    string
	DefaultRateLimitPolicyID string
	Revision                 uint64
	SeedVersion              uint64
	UpdatedAt                time.Time
}

type ManagementSecurityPolicy struct {
	NamespaceID        string
	ActionRequirements map[string]managementauth.ActionRequirement
	SeedVersion        uint64
	Revision           uint64
	UpdatedAt          time.Time
}

type RoutingClaimSchema struct {
	NamespaceID string
	Definitions map[string]accessmanagement.ClaimDefinition
	Revision    uint64
	UpdatedAt   time.Time
}

type Actor struct {
	PrincipalID string
	ActorChain  []string
	RequestID   string
	SourceIP    netip.Addr
	Reason      string
}

type ResultScope struct {
	All          bool
	NamespaceIDs []string
}

type ListRequest struct {
	Scope    ResultScope
	Status   string
	Cursor   string
	PageSize int
}

type NamespaceCursor struct {
	CreatedAt time.Time
	ID        string
}

type NamespaceQuery struct {
	Scope  ResultScope
	Status string
	After  *NamespaceCursor
	Limit  int
}

type RepositoryPage[T any] struct {
	Items   []T
	HasMore bool
}

type Page[T any] struct {
	Items      []T
	NextCursor string
	HasMore    bool
	PageSize   int
}

type MutationResult struct {
	Kind       string
	ID         string
	Revision   uint64
	Replayed   bool
	HTTPStatus int
}

type CreateNamespaceRequest struct {
	Name            string
	BillingCurrency string
	IdempotencyKey  string
	Actor           Actor
}

type CreateNamespaceMutation struct {
	Namespace     Namespace
	SelfService   SelfServicePolicy
	Security      ManagementSecurityPolicy
	RoutingClaims RoutingClaimSchema
	Command       managementcommand.Command
	Actor         Actor
}

type PatchNamespaceRequest struct {
	NamespaceID      string
	ExpectedRevision uint64
	Status           accesscontrol.NamespaceStatus
	Actor            Actor
}

type DeleteNamespaceRequest struct {
	NamespaceID      string
	ExpectedRevision uint64
	Actor            Actor
}

type PatchSelfServicePolicyRequest struct {
	NamespaceID                string
	ExpectedRevision           uint64
	MaxKeysPerUser             *int
	MaxDelegatedSessions       *int
	DelegatedSessionTTLSeconds *int64
	AllowTeamKeyDelegation     *bool
	AutomaticFirstKey          *bool
	TeamAdminCapabilities      *[]accesscontrol.TeamAdminCapability
	DefaultAccessPolicyID      *string
	DefaultRateLimitPolicyID   *string
	Actor                      Actor
}

type PatchManagementSecurityPolicyRequest struct {
	NamespaceID        string
	ExpectedRevision   uint64
	ActionRequirements map[string]managementauth.ActionRequirement
	Session            managementauth.LiveSession
	Actor              Actor
}

type PatchRoutingClaimSchemaRequest struct {
	NamespaceID      string
	ExpectedRevision uint64
	Definitions      map[string]accessmanagement.ClaimDefinition
	Actor            Actor
}

type Repository interface {
	RepositoryLifecycle
	NamespaceReader
	NamespaceMutationRepository
	SelfServicePolicyRepository
	ManagementSecurityPolicyRepository
	RoutingClaimSchemaRepository
}

type RepositoryLifecycle interface {
	Ready(context.Context, *managementcommand.Codec) error
	Replay(context.Context, managementcommand.Command) (MutationResult, bool, error)
}

type NamespaceReader interface {
	GetNamespace(context.Context, string) (Namespace, error)
	ListNamespaces(context.Context, NamespaceQuery) (RepositoryPage[Namespace], error)
}

type NamespaceMutationRepository interface {
	CreateNamespace(context.Context, CreateNamespaceMutation) (MutationResult, error)
	PatchNamespace(context.Context, Namespace, uint64, Actor) (MutationResult, error)
	DeleteNamespace(context.Context, string, uint64, Actor) (MutationResult, error)
}

type SelfServicePolicyRepository interface {
	GetSelfServicePolicy(context.Context, string) (SelfServicePolicy, error)
	PatchSelfServicePolicy(context.Context, SelfServicePolicy, uint64, Actor) (MutationResult, error)
}

type ManagementSecurityPolicyRepository interface {
	GetManagementSecurityPolicy(context.Context, string) (ManagementSecurityPolicy, error)
	PatchManagementSecurityPolicy(context.Context, ManagementSecurityPolicy, uint64, Actor) (MutationResult, error)
}

type RoutingClaimSchemaRepository interface {
	GetRoutingClaimSchema(context.Context, string) (RoutingClaimSchema, error)
	PatchRoutingClaimSchema(context.Context, RoutingClaimSchema, uint64, Actor) (MutationResult, error)
}
