package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
)

type APIKeyManagementService interface {
	Ready(context.Context) error
	Get(context.Context, string, string) (accesscontrol.APIKey, error)
	List(context.Context, apikeymanagement.ListKeysRequest) (apikeymanagement.KeyPage, error)
	Create(context.Context, apikeymanagement.CreateRequest) (apikeymanagement.SecretMutationResult, error)
	Rename(context.Context, apikeymanagement.RenameRequest) (apikeymanagement.MutationResult, error)
	Enable(context.Context, apikeymanagement.LifecycleRequest) (apikeymanagement.MutationResult, error)
	Disable(context.Context, apikeymanagement.LifecycleRequest) (apikeymanagement.MutationResult, error)
	Renew(context.Context, apikeymanagement.RenewRequest) (apikeymanagement.MutationResult, error)
	Reassign(context.Context, apikeymanagement.ReassignRequest) (apikeymanagement.MutationResult, error)
	Delete(context.Context, apikeymanagement.LifecycleRequest) (apikeymanagement.MutationResult, error)
	ListCredentials(context.Context, apikeymanagement.ListCredentialsRequest) (apikeymanagement.CredentialPage, error)
	Rotate(context.Context, apikeymanagement.RotateRequest) (apikeymanagement.SecretMutationResult, error)
	RevokeCredential(context.Context, apikeymanagement.RevokeCredentialRequest) (apikeymanagement.MutationResult, error)
	Reveal(context.Context, apikeymanagement.RevealRequest) (string, error)
}

type APIKeyRoutesOptions struct {
	Service       APIKeyManagementService
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Scopes        ResultScopeResolver
	Now           func() time.Time
}
