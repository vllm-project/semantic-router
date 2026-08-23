package postgres

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type AccessPolicyBindingReader interface {
	GetAccessPolicyBinding(context.Context, accesscontrol.NamespaceID, accesscontrol.PolicyBindingID) (accesscontrol.AccessPolicyBinding, error)
}

type AccessPolicyBindingWriter interface {
	CreateAccessPolicyBinding(context.Context, accesscontrol.AccessPolicyBinding, MutationMeta) (MutationResult[accesscontrol.AccessPolicyBinding], error)
	SetAccessPolicyBindingStatus(context.Context, accesscontrol.NamespaceID, accesscontrol.PolicyBindingID, accesscontrol.Revision, accesscontrol.BindingStatus, MutationMeta) (MutationResult[accesscontrol.AccessPolicyBinding], error)
}

type AccessPolicyBindingRepository interface {
	AccessPolicyBindingReader
	AccessPolicyBindingWriter
}

type RateLimitBindingReader interface {
	GetRateLimitBinding(context.Context, accesscontrol.NamespaceID, accesscontrol.PolicyBindingID) (accesscontrol.RateLimitBinding, error)
}

type RateLimitBindingWriter interface {
	CreateRateLimitBinding(context.Context, accesscontrol.RateLimitBinding, MutationMeta) (MutationResult[accesscontrol.RateLimitBinding], error)
	SetRateLimitBindingStatus(context.Context, accesscontrol.NamespaceID, accesscontrol.PolicyBindingID, accesscontrol.Revision, accesscontrol.BindingStatus, MutationMeta) (MutationResult[accesscontrol.RateLimitBinding], error)
}

type RateLimitBindingRepository interface {
	RateLimitBindingReader
	RateLimitBindingWriter
}

type BindingRepository interface {
	AccessPolicyBindingRepository
	RateLimitBindingRepository
}
