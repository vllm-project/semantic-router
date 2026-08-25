package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
)

type AccessPolicyReadService interface {
	Ready(context.Context) error
	Inspect(context.Context, string, accessmanagement.Subject) (accessmanagement.AuthorizationContext, error)
	GetEffectivePolicy(context.Context, string, accessmanagement.Subject) (accessmanagement.EffectivePolicy, error)
	GetQuota(context.Context, string, accessmanagement.Subject) (accessmanagement.EffectiveQuota, error)
}

type AccessRoutingReadService interface {
	GetRoutingCatalog(context.Context, string, accessmanagement.Subject) (accessmanagement.RoutingCatalog, error)
	GetRoutingContext(context.Context, string, accessmanagement.Subject) (accessmanagement.RoutingContext, error)
	UpdateRoutingContext(context.Context, accessmanagement.UpdateRoutingContextRequest) (accessmanagement.RoutingContext, error)
	Check(context.Context, accessmanagement.AccessCheckRequest) (accessmanagement.AccessCheckResult, error)
}

type AccessReadService interface {
	AccessPolicyReadService
	AccessRoutingReadService
}

type AccessReadRoutesOptions struct {
	Service       AccessReadService
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Now           func() time.Time
}
