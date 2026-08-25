package managementidentity

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

type IdentityRepositoryLifecycle interface {
	Ready(context.Context) error
}

type PrincipalRepository interface {
	GetPrincipal(context.Context, string) (Principal, error)
	ListPrincipals(context.Context, ListRequest) (PrincipalPage, error)
	CreatePrincipal(context.Context, CreatePrincipal) (MutationResult, error)
	UpdatePrincipal(context.Context, UpdatePrincipal) (MutationResult, error)
	DeletePrincipal(context.Context, string, uint64, MutationActor) (MutationResult, error)
}

type RoleRepository interface {
	GetRole(context.Context, string) (Role, error)
	ListRoles(context.Context, string, ListRequest) (RolePage, error)
	CreateRole(context.Context, CreateRole) (MutationResult, error)
	UpdateRole(context.Context, UpdateRole) (MutationResult, error)
	DeleteRole(context.Context, string, uint64, MutationActor) (MutationResult, error)
}

type RoleBindingRepository interface {
	GetRoleBinding(context.Context, string) (RoleBinding, error)
	ListRoleBindings(context.Context, string, ListRequest) (RoleBindingPage, error)
	CreateRoleBinding(context.Context, CreateRoleBinding) (MutationResult, error)
	UpdateRoleBinding(context.Context, UpdateRoleBinding) (MutationResult, error)
	DeleteRoleBinding(context.Context, string, uint64, MutationActor) (MutationResult, error)
}

type PrincipalDirectoryRepository interface {
	GetPrincipalUserLink(context.Context, string, string) (PrincipalUserLink, error)
	GetPrincipalDirectoryEntry(context.Context, string, string) (PrincipalDirectoryEntry, error)
	ListPrincipalDirectory(context.Context, PrincipalDirectoryRequest) (PrincipalDirectoryPage, error)
	ListPrincipalUserLinks(context.Context, PrincipalUserLinkListRequest) (PrincipalUserLinkPage, error)
	ListPrincipalLinks(context.Context, string, ListRequest) (PrincipalUserLinkPage, error)
}

type PrincipalLinkMutationRepository interface {
	PutPrincipalUserLink(context.Context, LinkMutation) (MutationResult, error)
	DeletePrincipalUserLink(context.Context, LinkMutation) (MutationResult, error)
}

type SessionPolicyRepository interface {
	LoadSessionPolicy(context.Context) (managementauth.SessionPolicy, error)
	UpdateSessionPolicy(context.Context, managementauth.SessionPolicy, uint64, MutationActor) (MutationResult, error)
	RevokePrincipalSessions(context.Context, string) ([]managementauth.SessionMutation, error)
}

type IdentityResourceRepository interface {
	PrincipalRepository
	RoleRepository
	RoleBindingRepository
}

type IdentityPolicyRepository interface {
	PrincipalDirectoryRepository
	PrincipalLinkMutationRepository
	SessionPolicyRepository
}

type Repository interface {
	IdentityRepositoryLifecycle
	IdentityResourceRepository
	IdentityPolicyRepository
}
