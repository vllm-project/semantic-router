package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
)

type InvitationManagementService interface {
	Ready(context.Context) error
	Get(context.Context, string, string) (invitationmanagement.Invitation, error)
	List(context.Context, invitationmanagement.ListRequest) (invitationmanagement.Page, error)
	Create(context.Context, invitationmanagement.CreateRequest) (invitationmanagement.SecretResult, error)
	Rotate(context.Context, invitationmanagement.RotateRequest) (invitationmanagement.SecretResult, error)
	Revoke(context.Context, invitationmanagement.RevokeRequest) (invitationmanagement.MutationResult, error)
	PrepareOnboarding(context.Context, string, string, []invitationmanagement.RequestedRoleGrant,
		*invitationmanagement.TeamAssignment) (invitationmanagement.OnboardingSnapshot, error)
	Onboard(context.Context, invitationmanagement.PrivilegedOnboardingRequest) (invitationmanagement.PrivilegedOnboardingResult, error)
}

type InvitationRoutesOptions struct {
	Service       InvitationManagementService
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Now           func() time.Time
}
