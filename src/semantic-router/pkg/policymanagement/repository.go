package policymanagement

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type Cursor struct {
	CreatedAt time.Time
	ID        string
}

type PolicyQuery struct {
	NamespaceID string
	Status      accesscontrol.PolicyStatus
	Search      string
	Scope       accesscontrol.ResultScope
	After       *Cursor
	Limit       int
}

type BindingQuery struct {
	NamespaceID  string
	PolicyID     string
	Subject      *Subject
	Status       accesscontrol.BindingStatus
	Mode         accesscontrol.RateBindingMode
	Scope        accesscontrol.ResultScope
	After        *Cursor
	Limit        int
	IncludeTotal bool
}

type RepositoryPage[T any] struct {
	Items      []T
	HasMore    bool
	TotalCount *uint64
}

type Repository interface {
	RepositoryLifecycle
	AccessPolicyRepository
	RateLimitPolicyRepository
	AccessBindingRepository
	RateBindingReader
	RateBindingMutationRepository
}

type RepositoryLifecycle interface {
	Ready(context.Context, *managementcommand.Codec) error
	Replay(context.Context, managementcommand.Command) (MutationResult, bool, error)
}

type AccessPolicyRepository interface {
	GetAccessPolicy(context.Context, string, string) (AccessPolicy, error)
	ListAccessPolicies(context.Context, PolicyQuery) (RepositoryPage[AccessPolicy], error)
	CreateAccessPolicy(context.Context, CreateAccessPolicyMutation) (MutationResult, error)
	UpdateAccessPolicy(context.Context, AccessPolicy, uint64, Actor) (MutationResult, error)
	DeleteAccessPolicy(context.Context, string, string, uint64, Actor) (MutationResult, error)
}

type RateLimitPolicyRepository interface {
	GetRateLimitPolicy(context.Context, string, string) (RateLimitPolicy, error)
	ListRateLimitPolicies(context.Context, PolicyQuery) (RepositoryPage[RateLimitPolicy], error)
	CreateRateLimitPolicy(context.Context, CreateRateLimitPolicyMutation) (MutationResult, error)
	UpdateRateLimitPolicy(context.Context, RateLimitPolicy, uint64, Actor) (MutationResult, error)
	DeleteRateLimitPolicy(context.Context, string, string, uint64, Actor) (MutationResult, error)
}

type AccessBindingRepository interface {
	GetAccessBinding(context.Context, string, string) (AccessPolicyBinding, error)
	ListAccessBindings(context.Context, BindingQuery) (RepositoryPage[AccessPolicyBinding], error)
	CreateAccessBinding(context.Context, CreateAccessBindingMutation) (MutationResult, error)
	UpdateAccessBinding(context.Context, string, string, uint64, accesscontrol.BindingStatus, Actor) (MutationResult, error)
	DeleteAccessBinding(context.Context, string, string, uint64, Actor) (MutationResult, error)
}

type RateBindingReader interface {
	GetRateBinding(context.Context, string, string) (RateLimitBinding, error)
	ListRateBindings(context.Context, BindingQuery) (RepositoryPage[RateLimitBinding], error)
}

type RateBindingMutationRepository interface {
	CreateRateBinding(context.Context, CreateRateBindingMutation) (MutationResult, error)
	CreateInlineRateBinding(context.Context, CreateInlineRateBindingMutation) (InlineRateBindingResult, error)
	UpdateRateBinding(context.Context, string, string, uint64, accesscontrol.BindingStatus, Actor) (MutationResult, error)
	DeleteRateBinding(context.Context, string, string, uint64, Actor) (MutationResult, error)
}
