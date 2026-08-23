// Package managementserver exposes the Router-native Management API over
// net/http. It depends on authenticated-session and authorization seams rather
// than a Dashboard proxy or Dashboard-owned identity state.
package managementserver

import (
	"context"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

type ProviderCatalog interface {
	List(context.Context, providercatalog.ListRequest) (providercatalog.ListResult, error)
	Get(context.Context, string) (providercatalog.DetailResult, error)
	PrepareDiscovery(context.Context, string, providercatalog.DiscoverModelsRequest) (providercatalog.DiscoveryPlan, error)
}

type ProviderDiscovery interface {
	Execute(context.Context, providerdiscovery.ExecuteRequest) (providerdiscovery.Result, error)
}

type ProviderCatalogAdministration interface {
	BootstrapRegistry(context.Context, uint64) (providercatalog.PublicationState, error)
	Activate(context.Context, string, uint64) (providercatalog.PublicationState, error)
}

type NamespaceResolver interface {
	ResolveNamespace(context.Context, *http.Request) (string, error)
}

type NamespaceResolverFunc func(context.Context, *http.Request) (string, error)

func (resolve NamespaceResolverFunc) ResolveNamespace(ctx context.Context, request *http.Request) (string, error) {
	return resolve(ctx, request)
}

type SessionAuthenticator interface {
	Authenticate(context.Context, string, string, time.Time) (managementauth.AuthenticatedSession, error)
}

type AuthorizationRequest struct {
	Operation   managementapi.OperationContract
	Session     managementauth.AuthenticatedSession
	NamespaceID string
	Targets     map[string][]accesscontrol.ScopedTarget
	Conditions  map[string]bool
	SpecialAuth map[string]bool
	Recorded    map[string]bool
}

type AuthorizationDecision struct {
	// AuthorityDigest binds discovery claims to the exact authorization facts
	// accepted by the authorizer. It is required for discovery and ignored for
	// catalog-only reads.
	AuthorityDigest string
}

type Authorizer interface {
	Authorize(context.Context, AuthorizationRequest) (AuthorizationDecision, error)
}

type AuthorizerFunc func(context.Context, AuthorizationRequest) (AuthorizationDecision, error)

func (authorize AuthorizerFunc) Authorize(
	ctx context.Context,
	request AuthorizationRequest,
) (AuthorizationDecision, error) {
	return authorize(ctx, request)
}

type ProviderRoutesOptions struct {
	Catalog       ProviderCatalog
	Discovery     ProviderDiscovery
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Now           func() time.Time
}

type ProviderCatalogAdministrationRoutesOptions struct {
	Administration ProviderCatalogAdministration
	Sessions       SessionAuthenticator
	Authorization  Authorizer
	Now            func() time.Time
}
