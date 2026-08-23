package managementidentity

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

// LifecycleRepository is the PostgreSQL authority for self identity, durable
// Management sessions, and trusted issuers. It remains separate from the
// principal and role repository so each persistence boundary has one purpose.
type LifecycleRepository interface {
	Ready(context.Context) error

	LoadSelf(context.Context, string, string) (SelfView, error)
	ListManagementSessions(context.Context, string, ListRequest) (ManagementSessionPage, error)
	RevokeSelfManagementSession(context.Context, string, string, MutationActor) (managementauth.SessionMutation, error)
	RevokeManagementSession(context.Context, SessionRevocationCommand) (managementauth.SessionMutation, MutationResult, error)
	RevokePrincipalManagementSessions(context.Context, PrincipalSessionRevocationCommand) (PrincipalSessionRevocation, error)

	GetTrustedIdentityIssuer(context.Context, string) (TrustedIdentityIssuer, error)
	ListTrustedIdentityIssuers(context.Context, ListRequest) (TrustedIdentityIssuerPage, error)
	CreateTrustedIdentityIssuer(context.Context, CreateTrustedIdentityIssuer) (IssuerMutation, error)
	UpdateTrustedIdentityIssuer(context.Context, UpdateTrustedIdentityIssuer) (IssuerMutation, error)
	DeleteTrustedIdentityIssuer(context.Context, string, uint64, MutationActor) (IssuerMutation, error)
	RefreshTrustedIdentityIssuer(context.Context, RefreshTrustedIdentityIssuer) (IssuerMutation, error)

	ApplyBackchannelLogout(context.Context, BackchannelLogout) (BackchannelLogoutResult, error)
}
