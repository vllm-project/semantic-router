package delegationmanagement

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type Repository interface {
	Ready(context.Context, *managementcommand.Codec) error
	ResolveSelf(context.Context, string, string, string, bool) (SelfContext, error)
	ListEligibleKeys(context.Context, EligibleKeyQuery) (Page[EligibleKey], error)
	GetEligibleKey(context.Context, string, string, string, string) (EligibleKey, error)
	GetKey(context.Context, string, string) (accesscontrol.APIKey, error)
	ListSessions(context.Context, SessionQuery) (Page[Session], error)
	GetSession(context.Context, string, string) (Session, error)
	ReplaySecret(context.Context, managementcommand.Command) (StoredSecret, bool, error)
	Create(context.Context, CreateMutation) (MutationResult, error)
	Revoke(context.Context, RevokeRequest) (MutationResult, error)
	RevokeAll(context.Context, RevokeAllMutation) (RevokeAllResult, error)
}

type PublicationWaiter interface {
	WaitActive(context.Context, Session, uint64) error
	WaitApplied(context.Context, string, string, uint64) error
}
