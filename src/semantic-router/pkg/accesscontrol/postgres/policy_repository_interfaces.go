package postgres

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type AccessPolicyReader interface {
	GetAccessPolicy(context.Context, accesscontrol.NamespaceID, accesscontrol.AccessPolicyID) (accesscontrol.AccessPolicy, error)
}

type AccessPolicyWriter interface {
	CreateAccessPolicy(context.Context, accesscontrol.AccessPolicy, MutationMeta) (MutationResult[accesscontrol.AccessPolicy], error)
	UpdateAccessPolicy(context.Context, accesscontrol.AccessPolicy, accesscontrol.Revision, MutationMeta) (MutationResult[accesscontrol.AccessPolicy], error)
}

type AccessPolicyRepository interface {
	AccessPolicyReader
	AccessPolicyWriter
}

type RateLimitPolicyReader interface {
	GetRateLimitPolicy(context.Context, accesscontrol.NamespaceID, accesscontrol.RateLimitPolicyID) (accesscontrol.RateLimitPolicy, error)
}

type RateLimitPolicyWriter interface {
	CreateRateLimitPolicy(context.Context, accesscontrol.RateLimitPolicy, MutationMeta) (MutationResult[accesscontrol.RateLimitPolicy], error)
	UpdateRateLimitPolicy(context.Context, accesscontrol.RateLimitPolicy, accesscontrol.Revision, MutationMeta) (MutationResult[accesscontrol.RateLimitPolicy], error)
}

type RateLimitPolicyRepository interface {
	RateLimitPolicyReader
	RateLimitPolicyWriter
}

type PolicyRepository interface {
	AccessPolicyRepository
	RateLimitPolicyRepository
}
