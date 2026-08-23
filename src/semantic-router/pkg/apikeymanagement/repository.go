package apikeymanagement

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type Repository interface {
	Ready(context.Context, *managementcommand.Codec) error
	ReplaySecret(context.Context, managementcommand.Command) (StoredSecret, bool, error)
	ReplayMutation(context.Context, managementcommand.Command) (MutationResult, bool, error)

	GetKey(context.Context, string, string) (accesscontrol.APIKey, error)
	ListKeys(context.Context, KeyQuery) (RepositoryPage[accesscontrol.APIKey], error)
	CreateKey(context.Context, CreateMutation) (MutationResult, error)
	UpdateKey(context.Context, accesscontrol.APIKey, uint64, Actor, string) (MutationResult, error)
	UpdateKeyAction(context.Context, UpdateMutation) (MutationResult, error)
	DeleteKey(context.Context, string, string, uint64, Actor) (MutationResult, error)

	ListCredentials(context.Context, CredentialQuery) (RepositoryPage[RevealSnapshot], error)
	GetCredential(context.Context, string, string, string) (RevealSnapshot, error)
	GetActiveCredential(context.Context, string, string) (RevealSnapshot, error)
	RotateCredential(context.Context, RotateMutation) (MutationResult, error)
	RevokeCredential(context.Context, string, string, string, uint64, Actor) (MutationResult, error)
	GetRevealSnapshot(context.Context, string, string, string) (RevealSnapshot, error)
	RecordReveal(context.Context, RevealSnapshot, Actor) error
}
