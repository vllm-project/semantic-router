package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

type PolicyManagementLifecycle interface {
	Ready(context.Context) error
}

type AccessPolicyManagementService interface {
	GetAccessPolicy(context.Context, string, string) (policymanagement.AccessPolicy, error)
	ListAccessPolicies(context.Context, policymanagement.ListPoliciesRequest) (policymanagement.Page[policymanagement.AccessPolicy], error)
	CreateAccessPolicy(context.Context, policymanagement.CreateAccessPolicyRequest) (policymanagement.MutationResult, error)
	UpdateAccessPolicy(context.Context, policymanagement.UpdateAccessPolicyRequest) (policymanagement.MutationResult, error)
	DeleteAccessPolicy(context.Context, policymanagement.DeletePolicyRequest) (policymanagement.MutationResult, error)
}

type RateLimitPolicyManagementService interface {
	GetRateLimitPolicy(context.Context, string, string) (policymanagement.RateLimitPolicy, error)
	ListRateLimitPolicies(context.Context, policymanagement.ListPoliciesRequest) (policymanagement.Page[policymanagement.RateLimitPolicy], error)
	CreateRateLimitPolicy(context.Context, policymanagement.CreateRateLimitPolicyRequest) (policymanagement.MutationResult, error)
	UpdateRateLimitPolicy(context.Context, policymanagement.UpdateRateLimitPolicyRequest) (policymanagement.MutationResult, error)
	DeleteRateLimitPolicy(context.Context, policymanagement.DeletePolicyRequest) (policymanagement.MutationResult, error)
}

type AccessBindingManagementService interface {
	GetAccessBinding(context.Context, string, string) (policymanagement.AccessPolicyBinding, error)
	ListAccessBindings(context.Context, policymanagement.ListBindingsRequest) (policymanagement.Page[policymanagement.AccessPolicyBinding], error)
	CreateAccessBinding(context.Context, policymanagement.CreateAccessBindingRequest) (policymanagement.MutationResult, error)
	UpdateAccessBinding(context.Context, policymanagement.UpdateBindingRequest) (policymanagement.MutationResult, error)
	DeleteAccessBinding(context.Context, policymanagement.DeleteBindingRequest) (policymanagement.MutationResult, error)
}

type RateBindingReadService interface {
	GetRateBinding(context.Context, string, string) (policymanagement.RateLimitBinding, error)
	ListRateBindings(context.Context, policymanagement.ListBindingsRequest) (policymanagement.Page[policymanagement.RateLimitBinding], error)
}

type RateBindingMutationService interface {
	CreateRateBinding(context.Context, policymanagement.CreateRateBindingRequest) (policymanagement.MutationResult, error)
	CreateInlineRateBinding(context.Context, policymanagement.CreateInlineRateBindingRequest) (policymanagement.InlineRateBindingResult, error)
	UpdateRateBinding(context.Context, policymanagement.UpdateBindingRequest) (policymanagement.MutationResult, error)
	DeleteRateBinding(context.Context, policymanagement.DeleteBindingRequest) (policymanagement.MutationResult, error)
}

type PolicyBindingManagementService interface {
	AccessBindingManagementService
	RateBindingReadService
	RateBindingMutationService
}

type PolicyManagementService interface {
	PolicyManagementLifecycle
	AccessPolicyManagementService
	RateLimitPolicyManagementService
	PolicyBindingManagementService
}

type PolicyRoutesOptions struct {
	Service       PolicyManagementService
	Bulk          PolicyBulkService
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Scopes        ResultScopeResolver
	Now           func() time.Time
}

type PolicyBulkService interface {
	Ready(context.Context) error
	EnqueueAccessBindings(context.Context, policybulk.EnqueueAccessRequest) (policybulk.EnqueueResult, error)
	EnqueueRateBindings(context.Context, policybulk.EnqueueRateRequest) (policybulk.EnqueueResult, error)
}
