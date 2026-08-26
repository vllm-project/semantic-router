package apikeymanagement

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type Repository interface {
	RepositoryLifecycle
	KeyReader
	KeyMutationRepository
	CredentialRepository
	CredentialRevealRepository
}

// PublicationWaiter closes the desired-state to data-plane handoff before a
// newly issued credential is returned to its caller. The one-time secret must
// never be observable before every Router replica can authenticate it through
// the shared applied projection.
type PublicationWaiter interface {
	WaitAPIKeyActive(context.Context, string, string, string) error
}

type RepositoryLifecycle interface {
	Ready(context.Context, *managementcommand.Codec) error
	ReplaySecret(context.Context, managementcommand.Command) (StoredSecret, bool, error)
	ReplayMutation(context.Context, managementcommand.Command) (MutationResult, bool, error)
}

type KeyReader interface {
	GetKey(context.Context, string, string) (accesscontrol.APIKey, error)
	ListKeys(context.Context, KeyQuery) (RepositoryPage[accesscontrol.APIKey], error)
}

type KeyMutationRepository interface {
	CreateKey(context.Context, CreateMutation) (MutationResult, error)
	UpdateKey(context.Context, accesscontrol.APIKey, uint64, Actor, string) (MutationResult, error)
	UpdateKeyAction(context.Context, UpdateMutation) (MutationResult, error)
	DeleteKey(context.Context, string, string, uint64, Actor) (MutationResult, error)
}

type CredentialRepository interface {
	ListCredentials(context.Context, CredentialQuery) (RepositoryPage[RevealSnapshot], error)
	GetCredential(context.Context, string, string, string) (RevealSnapshot, error)
	GetActiveCredential(context.Context, string, string) (RevealSnapshot, error)
	RotateCredential(context.Context, RotateMutation) (MutationResult, error)
	RevokeCredential(context.Context, string, string, string, uint64, Actor) (MutationResult, error)
}

type CredentialRevealRepository interface {
	GetRevealSnapshot(context.Context, string, string, string) (RevealSnapshot, error)
	RecordReveal(context.Context, RevealSnapshot, uint64, Actor) error
}
