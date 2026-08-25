package managementcomposition

import (
	"context"
	"errors"
	"fmt"
	"strings"

	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	catalogapplication "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog/application"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	routingapplication "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement/application"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	managementauthpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/postgres"
	authorizationpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	identityapplication "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity/application"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/namespacemanagement"
	credentialmanagement "github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/management"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingruntime"
)

// buildDurableRouting composes the PostgreSQL-only Management surface. Access,
// quota, usage, delegation, and Agent services are deliberately absent; this
// process owns identity, Namespace, Provider, credential, and routing authoring
// only. Native Access uses the richer composition in Factory.Build.
func (factory *Factory) buildDurableRouting(
	ctx context.Context,
	dependencies routingruntime.ManagementDependencies,
) (routingruntime.ManagementAPI, error) {
	if err := validateDurableRoutingDependencies(dependencies); err != nil {
		return nil, err
	}
	builtInRecipes, err := factory.loadBuiltInRecipes()
	if err != nil {
		return nil, err
	}
	modelProber := factory.modelProber
	if modelProber == nil {
		modelProber = dependencies.ModelProber
	}
	if modelProber == nil {
		return nil, errors.New("durable routing composition requires a Router-owned Model prober")
	}
	commandCodec, err := managementcommand.NewCodec(
		dependencies.Keyrings.Routing.ManagementCommand.Symmetric(),
	)
	if err != nil {
		return nil, fmt.Errorf("compose Management command codec: %w", err)
	}
	owned := &application{}
	owned.addCloser(commandCodec.Close)
	fail := func(cause error) (routingruntime.ManagementAPI, error) {
		_ = owned.Close()
		return nil, cause
	}
	identity, err := factory.composeDurableIdentity(ctx, dependencies, commandCodec, owned)
	if err != nil {
		return fail(err)
	}
	credentialRoutes, err := factory.composeDurableCredentials(dependencies, commandCodec, identity, owned)
	if err != nil {
		return fail(err)
	}
	routing, err := factory.composeDurableRoutingApplication(
		ctx, dependencies, commandCodec, identity, builtInRecipes, modelProber, owned,
	)
	if err != nil {
		return fail(err)
	}
	server, err := dependencies.Catalog.NewManagementServer(catalogapplication.ManagementServerOptions{
		Namespaces: identity.namespaces, Sessions: identity.application.SessionAuthenticator(),
		Authorization: identity.application.Authorizer(), AdditionalRoutes: []managementserver.RouteRegistrar{
			identity.application, identity.namespaceRoutes, credentialRoutes, routing,
		}, Now: factory.now,
	})
	if err != nil {
		return fail(fmt.Errorf("compose durable routing HTTP server: %w", err))
	}
	owned.server = server
	return owned, nil
}

type durableIdentityComposition struct {
	application     *identityapplication.IdentityApplication
	authorityStore  *authorizationpostgres.Store
	namespaceRoutes *managementserver.NamespaceRoutes
	namespaces      managementserver.ExplicitNamespaceResolver
}

func (factory *Factory) composeDurableIdentity(
	ctx context.Context,
	dependencies routingruntime.ManagementDependencies,
	commandCodec *managementcommand.Codec,
	owned *application,
) (*durableIdentityComposition, error) {
	assertions := factory.assertionVerifier
	issuerKeys := factory.issuerKeyCache
	logoutVerifier := factory.backchannelLogoutVerifier
	if assertions == nil {
		assertionComposition, err := composeAssertionVerifier(dependencies)
		if err != nil {
			return nil, err
		}
		owned.addCloser(assertionComposition.Close)
		assertions = assertionComposition.Verifier()
		issuerKeys = assertionComposition.keys
		logoutVerifier = assertionComposition.verifier
	}
	exchanges, err := managementauthpostgres.NewIdentityExchangeCoordinator(
		dependencies.Database, dependencies.SessionStore,
	)
	if err != nil {
		return nil, fmt.Errorf("compose durable Management identity exchange: %w", err)
	}
	identity, err := identityapplication.New(ctx, identityapplication.Options{
		Database: dependencies.Database, SessionStore: dependencies.SessionStore,
		CommandCodec: commandCodec, SessionTokenCodec: managementTokenCodec(dependencies),
		ServiceCredentialPeppers: pepperSymmetric(dependencies.Keyrings.ServiceAccounts),
		BootstrapToken:           dependencies.BootstrapToken, BootstrapTokenPresent: dependencies.BootstrapTokenPresent,
		RecoveryToken:            dependencies.RecoveryToken,
		BootstrapIdempotencyKeys: dependencies.Keyrings.Routing.BootstrapIdempotency.Symmetric(),
		BootstrapResponseKEKs:    dependencies.Keyrings.ResponseKEK,
		WorkloadCursorKeyring:    dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		WorkloadResponseKEKs:     dependencies.Keyrings.ResponseKEK,
		WorkloadIdempotencyTTL:   defaultIdempotencyTTL, WorkloadSecretDeliveryTTL: defaultSecretDeliveryTTL,
		MTLSListenerEnabled: factory.mtlsListenerEnabled, AssertionVerifier: assertions,
		IssuerKeyCache: issuerKeys, BackchannelLogoutVerifier: logoutVerifier,
		Exchanges: exchanges, Now: factory.now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose durable Management identity: %w", err)
	}
	owned.addCloser(identity.Close)
	authorityStore, err := authorizationpostgres.New(dependencies.Database)
	if err != nil {
		return nil, fmt.Errorf("compose Management authority store: %w", err)
	}
	namespaceRepository, err := accesspostgres.NewNamespaceManagementRepository(dependencies.AccessStore)
	if err != nil {
		return nil, fmt.Errorf("compose Namespace repository: %w", err)
	}
	namespaceService, err := namespacemanagement.NewService(namespacemanagement.Options{
		Repository: namespaceRepository, CommandCodec: commandCodec,
		CursorKeyring:  dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, Now: factory.now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose Namespace Management: %w", err)
	}
	owned.addCloser(func() error { namespaceService.Close(); return nil })
	namespaceRoutes, err := managementserver.NewNamespaceRoutes(managementserver.NamespaceRoutesOptions{
		Service: namespaceService, Sessions: identity.SessionAuthenticator(),
		Authorization: identity.Authorizer(), Scopes: authorityStore, Now: factory.now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose Namespace Management routes: %w", err)
	}
	return &durableIdentityComposition{
		application: identity, authorityStore: authorityStore,
		namespaceRoutes: namespaceRoutes, namespaces: managementserver.ExplicitNamespaceResolver{},
	}, nil
}

func (factory *Factory) composeDurableCredentials(
	dependencies routingruntime.ManagementDependencies,
	commandCodec *managementcommand.Codec,
	identity *durableIdentityComposition,
	owned *application,
) (*managementserver.ProviderCredentialRoutes, error) {
	credentials, err := credentialmanagement.NewService(credentialmanagement.Options{
		Repository: dependencies.AccessStore, Catalog: dependencies.Catalog.Catalog,
		Egress: dependencies.EgressPolicy, CredentialCodec: dependencies.ProviderCredentialCodec,
		CommandCodec:    commandCodec,
		CursorKeyring:   dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		IdempotencyTTL:  defaultIdempotencyTTL,
		RetiringOverlap: defaultCredentialRetirement, Now: factory.now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose ProviderCredential Management: %w", err)
	}
	owned.addCloser(credentials.Close)
	routes, err := managementserver.NewProviderCredentialRoutes(managementserver.ProviderCredentialRoutesOptions{
		Service: credentials, Namespaces: identity.namespaces,
		Sessions:      identity.application.SessionAuthenticator(),
		Authorization: identity.application.Authorizer(), Now: factory.now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose ProviderCredential Management routes: %w", err)
	}
	return routes, nil
}

func (factory *Factory) composeDurableRoutingApplication(
	ctx context.Context,
	dependencies routingruntime.ManagementDependencies,
	commandCodec *managementcommand.Codec,
	identity *durableIdentityComposition,
	builtInRecipes routingmanagement.BuiltInRecipeDistribution,
	modelProber routingmanagement.Prober,
	owned *application,
) (*routingapplication.Application, error) {
	routing, err := routingapplication.NewApplication(routingapplication.ApplicationOptions{
		DB: dependencies.Database,
		ModelCompiler: providercatalog.ModelCompiler{
			Catalog: dependencies.Catalog.Coordinator, Registry: dependencies.Catalog.Registry,
			Credentials: dependencies.AccessStore, Egress: dependencies.EgressPolicy,
		},
		DiscoveryClaims:    dependencies.Catalog.Discovery.Claims,
		CredentialVersions: dependencies.ProviderCredentialResolver,
		Prober:             modelProber, ValidatePublication: factory.validateRoutingPublication,
		CommandCodec:   commandCodec,
		CursorKeyring:  dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, Namespaces: identity.namespaces,
		Sessions:       identity.application.SessionAuthenticator(),
		Authorization:  identity.application.Authorizer(),
		BuiltInRecipes: builtInRecipes, Now: factory.now,
		ManifestCodec: mustRoutingManifestCodec(dependencies.Catalog.Registry),
	})
	if err != nil {
		return nil, fmt.Errorf("compose Routing Management: %w", err)
	}
	owned.addCloser(routing.Close)
	if err := routing.ReconcileBuiltInRecipes(ctx); err != nil {
		return nil, fmt.Errorf("install built-in Recipes: %w", err)
	}
	owned.workers = append(owned.workers, routing)
	return routing, nil
}

func managementTokenCodec(dependencies routingruntime.ManagementDependencies) managementauth.TokenCodec {
	return managementauth.TokenCodec{
		Keyring:  cloneSigning(dependencies.Keyrings.ManagementSigning),
		Issuer:   managementTokenIssuer,
		Audience: managementTokenAudience,
		MaxSkew:  managementTokenClockSkew,
	}
}

func validateDurableRoutingDependencies(dependencies routingruntime.ManagementDependencies) error {
	if dependencies.Database == nil || dependencies.AccessStore == nil || dependencies.SessionStore == nil ||
		dependencies.Catalog == nil || dependencies.Catalog.Catalog == nil ||
		dependencies.Catalog.Coordinator == nil || dependencies.Catalog.Registry == nil ||
		dependencies.Catalog.Discovery == nil {
		return errors.New("durable routing Management dependencies are incomplete")
	}
	if !validPolicyWorkerID(dependencies.ReplicaID) || strings.TrimSpace(dependencies.ReplicaID) == "" {
		return errors.New("durable routing Management requires a stable replica identity")
	}
	return nil
}
