package managementserver

import (
	"context"
	"time"

	credentialmanagement "github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/management"
)

type ProviderCredentialService interface {
	Ready(context.Context) error
	Create(ctx context.Context, request credentialmanagement.CreateRequest) (credentialmanagement.MutationResult, error)
	Get(ctx context.Context, namespaceID, credentialID string) (credentialmanagement.Metadata, error)
	List(ctx context.Context, request credentialmanagement.ListRequest) (credentialmanagement.ListResult, error)
	Rename(ctx context.Context, request credentialmanagement.RenameRequest) (credentialmanagement.MutationResult, error)
	Rotate(ctx context.Context, request credentialmanagement.RotateRequest) (credentialmanagement.MutationResult, error)
	Disable(ctx context.Context, request credentialmanagement.LifecycleRequest) (credentialmanagement.MutationResult, error)
	Reactivate(ctx context.Context, request credentialmanagement.LifecycleRequest) (credentialmanagement.MutationResult, error)
	Delete(ctx context.Context, request credentialmanagement.LifecycleRequest) (credentialmanagement.MutationResult, error)
}

type ProviderCredentialRoutesOptions struct {
	Service       ProviderCredentialService
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Scopes        ResultScopeResolver
	Now           func() time.Time
}
