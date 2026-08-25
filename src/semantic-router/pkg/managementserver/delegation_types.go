package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
)

type DelegationReadService interface {
	Ready(context.Context) error
	ResolveSelf(context.Context, string, string, string) (delegationmanagement.SelfContext, error)
	GetKey(context.Context, string, string) (accesscontrol.APIKey, error)
	GetSession(context.Context, string, string) (delegationmanagement.Session, error)
	ListEligibleKeys(context.Context, delegationmanagement.ListRequest) (delegationmanagement.ResultPage[delegationmanagement.EligibleKey], error)
	GetEligibleKey(context.Context, delegationmanagement.EligibleKeyRequest) (delegationmanagement.EligibleKey, error)
}

type DelegationMutationService interface {
	ListSessions(context.Context, delegationmanagement.ListRequest) (delegationmanagement.ResultPage[delegationmanagement.Session], error)
	Create(context.Context, delegationmanagement.CreateRequest) (delegationmanagement.SecretResult, error)
	Revoke(context.Context, delegationmanagement.RevokeRequest) (delegationmanagement.MutationResult, error)
	RevokeAll(context.Context, delegationmanagement.RevokeAllRequest) (delegationmanagement.RevokeAllResult, error)
}

type DelegationManagementService interface {
	DelegationReadService
	DelegationMutationService
}

type DelegationRoutesOptions struct {
	Service       DelegationManagementService
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Now           func() time.Time
}
