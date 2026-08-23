// Package policybulk owns bounded asynchronous policy-binding commands. It is
// intentionally resource-specific: callers cannot submit arbitrary tasks, and
// every queued item is a normalized, secret-free binding command.
package policybulk

import (
	"context"
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

const (
	MaximumItems               = 1000
	AccessBindingOperationKind = "access_policy_bindings.bulk_apply"
	RateBindingOperationKind   = "rate_limit_bindings.bulk_apply"
)

type ItemKind string

const (
	ItemKindAccessBinding ItemKind = "access_policy_binding"
	ItemKindRateBinding   ItemKind = "rate_limit_binding"
)

type OperationState string

const (
	OperationPending            OperationState = "pending"
	OperationRunning            OperationState = "running"
	OperationSucceeded          OperationState = "succeeded"
	OperationPartiallySucceeded OperationState = "partially_succeeded"
	OperationFailed             OperationState = "failed"
	OperationCancelled          OperationState = "cancelled"
)

type ItemFailure struct {
	ItemID string `json:"itemId"`
	Code   string `json:"code"`
	Reason string `json:"reason"`
}

type Operation struct {
	ID                  string
	NamespaceID         string
	Kind                string
	OriginPrincipalID   string
	ActorChain          []string
	Version             uint64
	State               OperationState
	Completed           uint64
	Failed              uint64
	Total               uint64
	TargetIDs           []string
	Targets             []OperationTarget
	DesiredRevision     uint64
	PublicationRevision uint64
	AppliedRevision     uint64
	ItemErrors          []ItemFailure
	CreatedAt           time.Time
	UpdatedAt           time.Time
	CompletedAt         *time.Time
}

type OperationTarget struct {
	ItemID       string                        `json:"itemId"`
	Kind         ItemKind                      `json:"kind"`
	PolicyID     string                        `json:"policyId,omitempty"`
	InlinePolicy bool                          `json:"inlinePolicy,omitempty"`
	Subject      policymanagement.Subject      `json:"subject"`
	Mode         accesscontrol.RateBindingMode `json:"mode,omitempty"`
}

type InlineRateLimitPolicy struct {
	Name        string
	Description string
	Rules       []policymanagement.RateLimitRule
}

type AccessBindingItem struct {
	ItemID   string
	PolicyID string
	Subject  policymanagement.Subject
}

type RateBindingItem struct {
	ItemID       string
	PolicyID     string
	InlinePolicy *InlineRateLimitPolicy
	Subject      policymanagement.Subject
	Mode         accesscontrol.RateBindingMode
}

type EnqueueAccessRequest struct {
	NamespaceID    string
	Items          []AccessBindingItem
	IdempotencyKey string
	Actor          policymanagement.Actor
}

type EnqueueRateRequest struct {
	NamespaceID    string
	Items          []RateBindingItem
	IdempotencyKey string
	Actor          policymanagement.Actor
}

type EnqueueResult struct {
	Operation Operation
	Replayed  bool
}

type ListRequest struct {
	NamespaceID       string
	OriginPrincipalID string
	Kind              string
	State             OperationState
	Cursor            string
	PageSize          int
	Visibility        OperationVisibility
}

type OperationVisibility struct {
	PrincipalID string
	Operation   accesscontrol.ResultScope
	Access      accesscontrol.ResultScope
	Rate        accesscontrol.ResultScope
}

type Page struct {
	Items      []Operation
	NextCursor string
	HasMore    bool
	PageSize   int
}

type AuthorizationRequest struct {
	NamespaceID  string
	PrincipalID  string
	ActorChain   []string
	Kind         ItemKind
	ItemID       string
	PolicyID     string
	InlinePolicy bool
	Subject      policymanagement.Subject
}

type ExecutionAuthorizer interface {
	AuthorizePolicyBulkItem(context.Context, AuthorizationRequest) error
}

// BindingService is the ordinary synchronous policy domain path.  The worker
// intentionally calls it instead of writing binding tables itself; retries
// therefore inherit the same validation, idempotency, audit, outbox, and
// compound inline-policy transaction semantics as Management API calls.
type BindingService interface {
	CreateAccessBinding(context.Context, policymanagement.CreateAccessBindingRequest) (policymanagement.MutationResult, error)
	CreateRateBinding(context.Context, policymanagement.CreateRateBindingRequest) (policymanagement.MutationResult, error)
	CreateInlineRateBinding(context.Context, policymanagement.CreateInlineRateBindingRequest) (policymanagement.InlineRateBindingResult, error)
}

type OperationContext struct {
	RequestID string
	SourceIP  netip.Addr
	ExpiresAt time.Time
}

type ItemClaim struct {
	OperationID       string
	NamespaceID       string
	OriginPrincipalID string
	ActorChain        []string
	Context           OperationContext
	ItemKind          ItemKind
	Access            *AccessBindingItem
	Rate              *RateBindingItem
	Attempt           int
	LeaseOwner        string
	LeaseToken        string
	LeaseExpiresAt    time.Time
}

type ItemResult struct {
	BindingID string
	PolicyID  string
}

type CancelRequest struct {
	NamespaceID     string
	OperationID     string
	ExpectedVersion uint64
	IdempotencyKey  string
	Actor           policymanagement.Actor
}

type CancelResult struct {
	Operation Operation
	Replayed  bool
}

type Cursor struct {
	CreatedAt time.Time
	ID        string
}

type OperationQuery struct {
	NamespaceID       string
	OriginPrincipalID string
	Kind              string
	State             OperationState
	After             *Cursor
	Limit             int
	Visibility        OperationVisibility
}

type RepositoryPage struct {
	Items   []Operation
	HasMore bool
}

// Repository is a typed durable queue.  It accepts only policy binding items;
// it is not an arbitrary task transport.
type Repository interface {
	Ready(context.Context, *managementcommand.Codec) error
	EnqueueAccess(context.Context, managementcommand.Command, Operation, OperationContext, []AccessBindingItem) (EnqueueResult, error)
	EnqueueRate(context.Context, managementcommand.Command, Operation, OperationContext, []RateBindingItem) (EnqueueResult, error)
	Get(context.Context, string, string) (Operation, error)
	List(context.Context, OperationQuery) (RepositoryPage, error)
	Cancel(context.Context, managementcommand.Command, CancelRequest) (CancelResult, error)
	Claim(context.Context, string, time.Time, time.Duration, int) (ItemClaim, bool, error)
	Complete(context.Context, ItemClaim, ItemResult, time.Time) (Operation, error)
	Fail(context.Context, ItemClaim, ItemFailure, bool, time.Time, time.Time, int) (Operation, error)
}
