package invitationmanagement

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type Repository interface {
	RepositoryLifecycle
	InvitationReader
	InvitationCommandReplay
	InvitationMutationRepository
	InvitationOnboardingRepository
}

type RepositoryLifecycle interface {
	Ready(context.Context, *managementcommand.Codec, []string, []string) error
}

type InvitationReader interface {
	Get(context.Context, string, string) (Invitation, error)
	GetByID(context.Context, string) (Invitation, []byte, string, error)
	List(context.Context, InvitationQuery) (RepositoryPage, error)
}

type InvitationCommandReplay interface {
	ReplaySecret(context.Context, managementcommand.Command) (StoredSecret, bool, error)
}

type InvitationMutationRepository interface {
	Create(context.Context, CreateMutation) (MutationResult, error)
	Rotate(context.Context, RotateMutation) (MutationResult, error)
	Revoke(context.Context, RevokeRequest) (MutationResult, error)
}

type InvitationOnboardingRepository interface {
	ResolveSnapshot(context.Context, string, string, []RequestedRoleGrant, *TeamAssignment) (OnboardingSnapshot, error)
	Onboard(context.Context, PrivilegedOnboardingMutation) (AcceptanceEnvelope, error)
}
