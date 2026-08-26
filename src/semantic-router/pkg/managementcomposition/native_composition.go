package managementcomposition

import (
	"context"
	"errors"
	"fmt"

	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	catalogapplication "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog/application"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	routingapplication "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement/application"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	authorizationpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	identityapplication "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity/application"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/namespacemanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
	policybulkpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
	credentialmanagement "github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/management"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

type nativeCompositionBuilder struct {
	ctx                  context.Context
	factory              *Factory
	dependencies         routingruntime.ManagementDependencies
	owned                *application
	commandCodec         *managementcommand.Codec
	builtInRecipes       routingmanagement.BuiltInRecipeDistribution
	modelProber          routingmanagement.Prober
	accessIdentity       *accessIdentityComposition
	identity             *identityapplication.IdentityApplication
	authorityStore       *authorizationpostgres.Store
	authorizationRuntime managementauthorization.Runtime
	namespaces           managementserver.ExplicitNamespaceResolver
	subjects             *subjectmanagement.Service
	routing              *routingapplication.Application
	policies             *policymanagement.Service
	bulk                 *policybulk.Service
	quotaRuntime         *quotaruntime.RedisEngine
	routes               []managementserver.RouteRegistrar
}

func newNativeCompositionBuilder(
	ctx context.Context,
	factory *Factory,
	dependencies routingruntime.ManagementDependencies,
) (*nativeCompositionBuilder, error) {
	builtInRecipes, err := factory.loadBuiltInRecipes()
	if err != nil {
		return nil, err
	}
	modelProber := factory.modelProber
	if modelProber == nil {
		modelProber = dependencies.ModelProber
	}
	if modelProber == nil {
		return nil, errors.New("management composition requires a Router-owned Model prober")
	}
	commandCodec, err := managementcommand.NewCodec(
		dependencies.Keyrings.Routing.ManagementCommand.Symmetric(),
	)
	if err != nil {
		return nil, fmt.Errorf("compose Management command codec: %w", err)
	}
	owned := &application{}
	owned.addCloser(commandCodec.Close)
	return &nativeCompositionBuilder{
		ctx: ctx, factory: factory, dependencies: dependencies, owned: owned,
		commandCodec: commandCodec, builtInRecipes: builtInRecipes, modelProber: modelProber,
		namespaces: managementserver.ExplicitNamespaceResolver{},
	}, nil
}

type identityVerifierSet struct {
	assertions     managementauth.SubjectAssertionVerifier
	issuerKeys     managementidentity.IssuerKeyCache
	logoutVerifier managementauth.BackchannelLogoutVerifier
}

func (builder *nativeCompositionBuilder) identityVerifiers() (identityVerifierSet, error) {
	verifiers := identityVerifierSet{
		assertions:     builder.factory.assertionVerifier,
		issuerKeys:     builder.factory.issuerKeyCache,
		logoutVerifier: builder.factory.backchannelLogoutVerifier,
	}
	if verifiers.assertions != nil {
		return verifiers, nil
	}
	composition, err := composeAssertionVerifier(builder.dependencies)
	if err != nil {
		return identityVerifierSet{}, err
	}
	builder.owned.addCloser(composition.Close)
	verifiers.assertions = composition.Verifier()
	verifiers.issuerKeys = composition.keys
	verifiers.logoutVerifier = composition.verifier
	return verifiers, nil
}

func (builder *nativeCompositionBuilder) composeIdentity() error {
	verifiers, err := builder.identityVerifiers()
	if err != nil {
		return err
	}
	builder.accessIdentity, err = composeAccessIdentity(
		builder.dependencies, builder.commandCodec, builder.factory.defaultRevealable,
		builder.factory.keyPrefix, builder.factory.now,
	)
	if err != nil {
		return err
	}
	builder.owned.addCloser(builder.accessIdentity.Close)
	builder.identity, err = identityapplication.New(builder.ctx, identityapplication.Options{
		Database: builder.dependencies.Database, Valkey: builder.dependencies.Redis,
		SessionStore: builder.dependencies.SessionStore, KeyPrefix: builder.factory.keyPrefix,
		CommandCodec: builder.commandCodec,
		SessionTokenCodec: managementauth.TokenCodec{
			Keyring: cloneSigning(builder.dependencies.Keyrings.ManagementSigning),
			Issuer:  managementTokenIssuer, Audience: managementTokenAudience, MaxSkew: managementTokenClockSkew,
		},
		ServiceCredentialPeppers: pepperSymmetric(builder.dependencies.Keyrings.ServiceAccounts),
		BootstrapToken:           builder.dependencies.BootstrapToken,
		BootstrapTokenPresent:    builder.dependencies.BootstrapTokenPresent,
		RecoveryToken:            builder.dependencies.RecoveryToken,
		BootstrapIdempotencyKeys: builder.dependencies.Keyrings.Routing.BootstrapIdempotency.Symmetric(),
		BootstrapResponseKEKs:    builder.dependencies.Keyrings.ResponseKEK,
		WorkloadCursorKeyring:    builder.dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		WorkloadResponseKEKs:     builder.dependencies.Keyrings.ResponseKEK,
		WorkloadIdempotencyTTL:   defaultIdempotencyTTL, WorkloadSecretDeliveryTTL: defaultSecretDeliveryTTL,
		MTLSListenerEnabled: builder.factory.mtlsListenerEnabled,
		AssertionVerifier:   verifiers.assertions, IssuerKeyCache: verifiers.issuerKeys,
		BackchannelLogoutVerifier: verifiers.logoutVerifier,
		Now:                       builder.factory.now, Exchanges: builder.accessIdentity.exchanges,
	})
	if err != nil {
		return fmt.Errorf("compose Management identity: %w", err)
	}
	builder.owned.addCloser(builder.identity.Close)
	builder.authorityStore, err = authorizationpostgres.New(builder.dependencies.Database)
	if err != nil {
		return fmt.Errorf("compose Management authority store: %w", err)
	}
	builder.authorizationRuntime = managementauthorization.Runtime{Loader: builder.authorityStore}
	return builder.composeNamespace()
}

func (builder *nativeCompositionBuilder) composeNamespace() error {
	repository, err := accesspostgres.NewNamespaceManagementRepository(builder.dependencies.AccessStore)
	if err != nil {
		return fmt.Errorf("compose Namespace repository: %w", err)
	}
	service, err := namespacemanagement.NewService(namespacemanagement.Options{
		Repository: repository, CommandCodec: builder.commandCodec,
		CursorKeyring:  builder.dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose Namespace Management: %w", err)
	}
	builder.owned.addCloser(func() error { service.Close(); return nil })
	routes, err := managementserver.NewNamespaceRoutes(managementserver.NamespaceRoutesOptions{
		Service: service, Sessions: builder.identity.SessionAuthenticator(),
		Authorization: builder.identity.Authorizer(), Scopes: builder.authorityStore, Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose Namespace Management routes: %w", err)
	}
	builder.routes = append(builder.routes, builder.identity, routes)
	return nil
}

func (builder *nativeCompositionBuilder) composeSubjectsAndCredentials() error {
	repository, err := accesspostgres.NewSubjectRepository(builder.dependencies.AccessStore)
	if err != nil {
		return fmt.Errorf("compose subject repository: %w", err)
	}
	builder.subjects, err = subjectmanagement.NewService(subjectmanagement.Options{
		Repository: repository, CommandCodec: builder.commandCodec,
		CursorKeyring:  builder.dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose subject Management: %w", err)
	}
	builder.owned.addCloser(func() error { builder.subjects.Close(); return nil })
	subjectRoutes, err := managementserver.NewSubjectRoutes(managementserver.SubjectRoutesOptions{
		Service: builder.subjects, Namespaces: builder.namespaces,
		Sessions: builder.identity.SessionAuthenticator(), Authorization: builder.identity.Authorizer(), Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose subject Management routes: %w", err)
	}
	accessRoutes, err := builder.composeAccessIdentityRoutes()
	if err != nil {
		return err
	}
	credentialRoutes, err := builder.composeProviderCredentialRoutes()
	if err != nil {
		return err
	}
	builder.routes = append(builder.routes, subjectRoutes)
	builder.routes = append(builder.routes, accessRoutes...)
	builder.routes = append(builder.routes, credentialRoutes)
	return nil
}

func (builder *nativeCompositionBuilder) composeAccessIdentityRoutes() ([]managementserver.RouteRegistrar, error) {
	common := func() (managementserver.SessionAuthenticator, managementserver.Authorizer) {
		return builder.identity.SessionAuthenticator(), builder.identity.Authorizer()
	}
	sessions, authorizer := common()
	apiKeys, err := managementserver.NewAPIKeyRoutes(managementserver.APIKeyRoutesOptions{
		Service: builder.accessIdentity.apiKeys, Namespaces: builder.namespaces,
		Sessions: sessions, Authorization: authorizer, Now: builder.factory.now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose API-key Management routes: %w", err)
	}
	delegations, err := managementserver.NewDelegationRoutes(managementserver.DelegationRoutesOptions{
		Service: builder.accessIdentity.delegations, Namespaces: builder.namespaces,
		Sessions: sessions, Authorization: authorizer, Now: builder.factory.now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose delegation Management routes: %w", err)
	}
	invitations, err := managementserver.NewInvitationRoutes(managementserver.InvitationRoutesOptions{
		Service: builder.accessIdentity.invitations, Namespaces: builder.namespaces,
		Sessions: sessions, Authorization: authorizer, Now: builder.factory.now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose invitation Management routes: %w", err)
	}
	return []managementserver.RouteRegistrar{apiKeys, delegations, invitations}, nil
}

func (builder *nativeCompositionBuilder) composeProviderCredentialRoutes() (*managementserver.ProviderCredentialRoutes, error) {
	service, err := credentialmanagement.NewService(credentialmanagement.Options{
		Repository: builder.dependencies.AccessStore, Catalog: builder.dependencies.Catalog.Catalog,
		Egress: builder.dependencies.EgressPolicy, CredentialCodec: builder.dependencies.ProviderCredentialCodec,
		CommandCodec:    builder.commandCodec,
		CursorKeyring:   builder.dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		IdempotencyTTL:  defaultIdempotencyTTL,
		RetiringOverlap: defaultCredentialRetirement, Now: builder.factory.now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose ProviderCredential Management: %w", err)
	}
	builder.owned.addCloser(service.Close)
	routes, err := managementserver.NewProviderCredentialRoutes(managementserver.ProviderCredentialRoutesOptions{
		Service: service, Namespaces: builder.namespaces,
		Sessions:      builder.identity.SessionAuthenticator(),
		Authorization: builder.identity.Authorizer(), Now: builder.factory.now,
	})
	if err != nil {
		return nil, fmt.Errorf("compose ProviderCredential Management routes: %w", err)
	}
	return routes, nil
}

func (builder *nativeCompositionBuilder) composeRouting() error {
	var err error
	builder.routing, err = routingapplication.NewApplication(routingapplication.ApplicationOptions{
		DB: builder.dependencies.Database,
		ModelCompiler: providercatalog.ModelCompiler{
			Catalog: builder.dependencies.Catalog.Coordinator, Registry: builder.dependencies.Catalog.Registry,
			Credentials: builder.dependencies.AccessStore, Egress: builder.dependencies.EgressPolicy,
		},
		DiscoveryClaims:    builder.dependencies.Catalog.Discovery.Claims,
		CredentialVersions: builder.dependencies.ProviderCredentialResolver,
		Prober:             builder.modelProber, ValidatePublication: builder.factory.validateRoutingPublication,
		CommandCodec:   builder.commandCodec,
		CursorKeyring:  builder.dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, Namespaces: builder.namespaces,
		Sessions: builder.identity.SessionAuthenticator(), Authorization: builder.identity.Authorizer(),
		BuiltInRecipes: builder.builtInRecipes, Now: builder.factory.now,
		ManifestCodec: mustRoutingManifestCodec(builder.dependencies.Catalog.Registry),
	})
	if err != nil {
		return fmt.Errorf("compose Routing Management: %w", err)
	}
	builder.owned.addCloser(builder.routing.Close)
	if err := builder.routing.ReconcileBuiltInRecipes(builder.ctx); err != nil {
		return fmt.Errorf("install built-in Recipes: %w", err)
	}
	builder.owned.workers = append(builder.owned.workers, builder.routing)
	builder.routes = append(builder.routes, builder.routing)
	return nil
}

func (builder *nativeCompositionBuilder) composePolicies() error {
	repository, err := accesspostgres.NewPolicyManagementRepository(builder.dependencies.AccessStore)
	if err != nil {
		return fmt.Errorf("compose policy repository: %w", err)
	}
	builder.policies, err = policymanagement.NewService(policymanagement.Options{
		Repository: repository, CommandCodec: builder.commandCodec,
		CursorKeyring:  builder.dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose policy Management: %w", err)
	}
	builder.owned.addCloser(func() error { builder.policies.Close(); return nil })
	executionAuthorizer, err := managementserver.NewPolicyBulkExecutionAuthorizer(builder.authorizationRuntime)
	if err != nil {
		return fmt.Errorf("compose policy execution authorizer: %w", err)
	}
	bulkRepository, err := policybulkpostgres.NewRepository(builder.dependencies.Database)
	if err != nil {
		return fmt.Errorf("compose policy operation repository: %w", err)
	}
	builder.bulk, err = policybulk.NewService(policybulk.Options{
		Repository: bulkRepository, Policies: builder.policies, Authorization: executionAuthorizer,
		CommandCodec:   builder.commandCodec,
		CursorKeyring:  builder.dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, WorkerID: builder.dependencies.ReplicaID,
		WorkerConcurrency: policyWorkerConcurrency, PollInterval: policyWorkerPollInterval,
		ClaimLease: policyWorkerClaimLease, MaximumAttempts: policyWorkerMaximumAttempts,
		Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose policy operation worker: %w", err)
	}
	builder.owned.addCloser(func() error { builder.bulk.Close(); return nil })
	builder.owned.workers = append(builder.owned.workers, builder.bulk)
	routes, err := managementserver.NewPolicyRoutes(managementserver.PolicyRoutesOptions{
		Service: builder.policies, Bulk: builder.bulk, Namespaces: builder.namespaces,
		Sessions:      builder.identity.SessionAuthenticator(),
		Authorization: builder.identity.Authorizer(), Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose policy Management routes: %w", err)
	}
	builder.routes = append(builder.routes, routes)
	return nil
}

func (builder *nativeCompositionBuilder) composeAccessRuntime() error {
	var err error
	builder.quotaRuntime, err = quotaruntime.NewRedisEngine(
		builder.dependencies.Redis, quotaruntime.RedisEngineOptions{KeyPrefix: builder.factory.keyPrefix},
	)
	if err != nil {
		return fmt.Errorf("compose quota reconciliation runtime: %w", err)
	}
	if err := builder.composeAccessReadRoutes(); err != nil {
		return err
	}
	return builder.composeQuotaReconciliationRoutes()
}

func (builder *nativeCompositionBuilder) composeAccessReadRoutes() error {
	applied, err := accessruntime.NewRedisProjectionReader(accessruntime.RedisProjectionReaderOptions{
		Client: builder.dependencies.Redis, KeyPrefix: builder.factory.keyPrefix,
	})
	if err != nil {
		return fmt.Errorf("compose applied access-policy reader: %w", err)
	}
	waiter, err := delegationmanagement.NewRedisPublicationWaiter(
		builder.dependencies.Redis, builder.factory.keyPrefix,
	)
	if err != nil {
		return fmt.Errorf("compose access publication waiter: %w", err)
	}
	routing, err := newAccessRoutingPublicationReader(
		builder.dependencies.Redis, builder.factory.keyPrefix,
	)
	if err != nil {
		return fmt.Errorf("compose applied routing-publication reader: %w", err)
	}
	service, err := accessmanagement.NewService(accessmanagement.ServiceOptions{
		Repository: builder.dependencies.AccessStore, Applied: applied,
		Routing: routing,
		Meters:  builder.quotaRuntime, Waiter: waiter,
	})
	if err != nil {
		return fmt.Errorf("compose access read service: %w", err)
	}
	routes, err := managementserver.NewAccessReadRoutes(managementserver.AccessReadRoutesOptions{
		Service: service, Namespaces: builder.namespaces,
		Sessions:      builder.identity.SessionAuthenticator(),
		Authorization: builder.identity.Authorizer(), Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose access read routes: %w", err)
	}
	builder.routes = append(builder.routes, routes)
	return nil
}

func (builder *nativeCompositionBuilder) composeQuotaReconciliationRoutes() error {
	service, err := quotareconciliation.NewService(quotareconciliation.Options{
		Repository: builder.dependencies.AccessStore, Runtime: builder.quotaRuntime,
		WaiveAuth: builder.dependencies.AccessStore, CommandCodec: builder.commandCodec,
		CursorKeyring:  builder.dependencies.Keyrings.Routing.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, WorkerID: builder.dependencies.ReplicaID,
		WorkerConcurrency: quotaReconciliationWorkers, PollInterval: policyWorkerPollInterval,
		ClaimLease: policyWorkerClaimLease, Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose quota reconciliation: %w", err)
	}
	builder.owned.addCloser(func() error { service.Close(); return nil })
	builder.owned.workers = append(builder.owned.workers, service)
	unknownUsage, err := managementserver.NewUnknownUsageRoutes(managementserver.UnknownUsageRoutesOptions{
		Service: service, Namespaces: builder.namespaces,
		Sessions:      builder.identity.SessionAuthenticator(),
		Authorization: builder.identity.Authorizer(), Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose unknown-usage Management routes: %w", err)
	}
	detailReader, err := managementserver.NewUnknownUsageOperationDetailReader(service, builder.identity.Authorizer())
	if err != nil {
		return fmt.Errorf("compose unknown-usage Operation reader: %w", err)
	}
	operations, err := managementserver.NewOperationRoutes(managementserver.OperationRoutesOptions{
		Service: builder.bulk, DetailReaders: []managementserver.OperationDetailReader{detailReader},
		Namespaces: builder.namespaces, Sessions: builder.identity.SessionAuthenticator(),
		Authorization: builder.identity.Authorizer(), Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose Management Operation routes: %w", err)
	}
	builder.routes = append(builder.routes, operations, unknownUsage)
	return nil
}

func (builder *nativeCompositionBuilder) composeObservabilityAndAgent() error {
	observability, err := composeObservability(
		builder.dependencies, builder.authorizationRuntime, builder.namespaces,
		builder.identity.SessionAuthenticator(), builder.identity.Authorizer(),
		builder.subjects, builder.accessIdentity.apiKeys, builder.factory.now,
	)
	if err != nil {
		return fmt.Errorf("compose Management observability: %w", err)
	}
	builder.owned.addCloser(observability.Close)
	statistics, err := composeStatistics(
		builder.dependencies, builder.authorizationRuntime, builder.namespaces,
		builder.identity.SessionAuthenticator(), builder.factory.now,
	)
	if err != nil {
		return fmt.Errorf("compose Management statistics: %w", err)
	}
	diagnostics, err := composeRuntimeDiagnostics(
		builder.dependencies, builder.factory.keyPrefix, builder.factory.maxUsageBacklog,
		builder.identity.SessionAuthenticator(), builder.identity.Authorizer(), builder.factory.now,
	)
	if err != nil {
		return fmt.Errorf("compose Management runtime diagnostics: %w", err)
	}
	agent, err := composeAgentRuntime(
		builder.ctx, builder.dependencies, builder.commandCodec, builder.authorityStore,
		builder.authorizationRuntime, builder.namespaces, builder.identity.SessionAuthenticator(),
		builder.identity.Authorizer(), builder.routing, builder.builtInRecipes,
		builder.factory.agentInferenceEndpoint, builder.factory.keyPrefix, builder.factory.now,
	)
	if err != nil {
		return fmt.Errorf("compose Router-native Agent runtime: %w", err)
	}
	builder.owned.addCloser(agent.Close)
	builder.owned.workers = append(builder.owned.workers, agent.workers...)
	builder.routes = append(builder.routes, observability.routes, statistics, diagnostics, agent.routes)
	return nil
}

func (builder *nativeCompositionBuilder) composeServer() error {
	server, err := builder.dependencies.Catalog.NewManagementServer(catalogapplication.ManagementServerOptions{
		Namespaces: builder.namespaces, Sessions: builder.identity.SessionAuthenticator(),
		Authorization: builder.identity.Authorizer(), AdditionalRoutes: builder.routes, Now: builder.factory.now,
	})
	if err != nil {
		return fmt.Errorf("compose Management HTTP server: %w", err)
	}
	disabledOperations := []string(nil)
	if len(builder.dependencies.RecoveryToken) == 0 {
		disabledOperations = append(disabledOperations, "postAuthRecovery")
	}
	if err := managementserver.ValidateRegisteredOperations(server, disabledOperations...); err != nil {
		return fmt.Errorf("validate Management registry route coverage: %w", err)
	}
	builder.owned.server = server
	return nil
}
