package invitationmanagement

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type Repository interface {
	Ready(context.Context, *managementcommand.Codec, []string, []string) error
	Get(context.Context, string, string) (Invitation, error)
	GetByID(context.Context, string) (Invitation, []byte, string, error)
	List(context.Context, InvitationQuery) (RepositoryPage, error)
	ReplaySecret(context.Context, managementcommand.Command) (StoredSecret, bool, error)
	ResolveSnapshot(context.Context, string, string, []RequestedRoleGrant, *TeamAssignment) (OnboardingSnapshot, error)
	Create(context.Context, CreateMutation) (MutationResult, error)
	Rotate(context.Context, RotateMutation) (MutationResult, error)
	Revoke(context.Context, RevokeRequest) (MutationResult, error)
	Onboard(context.Context, PrivilegedOnboardingMutation) (AcceptanceEnvelope, error)
}
