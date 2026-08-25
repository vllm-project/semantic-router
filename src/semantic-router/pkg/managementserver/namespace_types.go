package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/namespacemanagement"
)

type NamespaceResourceService interface {
	Ready(context.Context) error
	GetNamespace(context.Context, string) (namespacemanagement.Namespace, error)
	ListNamespaces(context.Context, namespacemanagement.ListRequest) (namespacemanagement.Page[namespacemanagement.Namespace], error)
}

type NamespaceMutationService interface {
	CreateNamespace(context.Context, namespacemanagement.CreateNamespaceRequest) (namespacemanagement.MutationResult, error)
	PatchNamespace(context.Context, namespacemanagement.PatchNamespaceRequest) (namespacemanagement.MutationResult, error)
	DeleteNamespace(context.Context, namespacemanagement.DeleteNamespaceRequest) (namespacemanagement.MutationResult, error)
}

type NamespacePolicyService interface {
	GetSelfServicePolicy(context.Context, string) (namespacemanagement.SelfServicePolicy, error)
	PatchSelfServicePolicy(context.Context, namespacemanagement.PatchSelfServicePolicyRequest) (namespacemanagement.MutationResult, error)
	GetManagementSecurityPolicy(context.Context, string) (namespacemanagement.ManagementSecurityPolicy, error)
	PatchManagementSecurityPolicy(context.Context, namespacemanagement.PatchManagementSecurityPolicyRequest) (namespacemanagement.MutationResult, error)
	GetRoutingClaimSchema(context.Context, string) (namespacemanagement.RoutingClaimSchema, error)
}

type NamespaceClaimSchemaService interface {
	PatchRoutingClaimSchema(context.Context, namespacemanagement.PatchRoutingClaimSchemaRequest) (namespacemanagement.MutationResult, error)
}

type NamespaceManagementService interface {
	NamespaceResourceService
	NamespaceMutationService
	NamespacePolicyService
	NamespaceClaimSchemaService
}

type NamespaceResultScopeResolver interface {
	ResolveNamespaceResultScope(context.Context, string) (namespacemanagement.ResultScope, error)
}

type NamespaceRoutesOptions struct {
	Service       NamespaceManagementService
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Scopes        NamespaceResultScopeResolver
	Now           func() time.Time
}
