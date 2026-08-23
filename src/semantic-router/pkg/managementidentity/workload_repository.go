package managementidentity

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type ServiceAccountCursor struct {
	CreatedAt time.Time
	ID        string
}

type ServiceCredentialCursor = ServiceAccountCursor

type MTLSMappingCursor = ServiceAccountCursor

type ServiceAccountQuery struct {
	Scope  ServiceAccountResultScope
	Status ServiceAccountStatus
	After  *ServiceAccountCursor
	Limit  int
}

type ServiceCredentialQuery struct {
	ServiceAccountID string
	After            *ServiceCredentialCursor
	Limit            int
}

type MTLSMappingQuery struct {
	Status string
	After  *MTLSMappingCursor
	Limit  int
}

type WorkloadRepositoryPage[T any] struct {
	Items   []T
	HasMore bool
}

type WorkloadIdentityRepository interface {
	ReadyWorkloadIdentity(context.Context, *managementcommand.Codec, bool) error
	ReplaySecret(context.Context, managementcommand.Command) (StoredWorkloadSecret, bool, error)

	GetServiceAccount(context.Context, string) (ServiceAccount, error)
	ListServiceAccounts(context.Context, ServiceAccountQuery) (WorkloadRepositoryPage[ServiceAccount], error)
	ListServiceCredentials(context.Context, ServiceCredentialQuery) (WorkloadRepositoryPage[ServiceCredential], error)
	GetServiceCredential(context.Context, string, string) (ServiceCredential, error)
	CreateServiceAccount(context.Context, ServiceAccountCreateMutation) (WorkloadMutationResult, error)
	PatchServiceAccount(context.Context, ServiceAccount, uint64, MutationActor) (WorkloadMutationResult, error)
	DeleteServiceAccount(context.Context, string, uint64, MutationActor) (WorkloadMutationResult, error)
	RotateServiceCredential(context.Context, ServiceCredentialRotateMutation) (WorkloadMutationResult, error)
	RevokeServiceCredential(context.Context, string, string, uint64, MutationActor) (WorkloadMutationResult, error)

	GetMTLSMapping(context.Context, string) (MTLSIdentityMapping, error)
	ListMTLSMappings(context.Context, MTLSMappingQuery) (WorkloadRepositoryPage[MTLSIdentityMapping], error)
	CreateMTLSMapping(context.Context, MTLSMappingCreateMutation) (WorkloadMutationResult, error)
	PatchMTLSMapping(context.Context, MTLSIdentityMapping, uint64, MutationActor) (WorkloadMutationResult, error)
	DeleteMTLSMapping(context.Context, string, uint64, MutationActor) (WorkloadMutationResult, error)
	ResolveMTLSIdentity(context.Context, string, string, time.Time) (VerifiedMTLSMapping, error)
}

type VerifiedMTLSMapping struct {
	MappingID       string
	PrincipalID     string
	WorkloadClass   string
	SourceAssuredAt time.Time
}
