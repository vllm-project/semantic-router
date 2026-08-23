// Package managementcomposition binds Router-native Management domains to
// process-owned managedruntime resources. It has no Dashboard dependency and
// exposes one Management authentication authority.
package managementcomposition

import (
	"context"
	"crypto/ed25519"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"time"

	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	catalogmanaged "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog/managed"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	routingmanaged "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement/managed"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managedruntime"
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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

const (
	defaultIdempotencyTTL       = time.Hour
	defaultSecretDeliveryTTL    = 10 * time.Minute
	defaultCredentialRetirement = 30 * time.Second
	managementTokenIssuer       = "vllm-sr"
	managementTokenAudience     = "vllm-sr-management"
	managementTokenClockSkew    = 5 * time.Second
	policyWorkerConcurrency     = 4
	policyWorkerPollInterval    = 250 * time.Millisecond
	policyWorkerClaimLease      = 30 * time.Second
	policyWorkerMaximumAttempts = 8
	quotaReconciliationWorkers  = 2
)

// Options exposes test and embedding seams. Production composes the Router's
// dynamic trusted-issuer verifier from managed dependencies when no verifier
// is injected.
type Options struct {
	AssertionVerifier         managementauth.SubjectAssertionVerifier
	IssuerKeyCache            managementidentity.IssuerKeyCache
	BackchannelLogoutVerifier managementauth.BackchannelLogoutVerifier
	ModelProber               routingmanagement.Prober
	BuiltInRecipes            *routingmanagement.BuiltInRecipeDistribution
	Now                       func() time.Time
}

type Factory struct {
	keyPrefix                  string
	agentInferenceEndpoint     string
	assertionVerifier          managementauth.SubjectAssertionVerifier
	issuerKeyCache             managementidentity.IssuerKeyCache
	backchannelLogoutVerifier  managementauth.BackchannelLogoutVerifier
	modelProber                routingmanagement.Prober
	builtInRecipes             *routingmanagement.BuiltInRecipeDistribution
	builtInRecipeDirectory     string
	defaultRevealable          bool
	maxUsageBacklog            int64
	mtlsListenerEnabled        bool
	validateRoutingPublication routingmanagement.PublicationValidator
	now                        func() time.Time
}

func NewFactory(cfg *config.RouterConfig, options Options) (*Factory, error) {
	if cfg == nil || cfg.ControlPlane.Mode != config.ControlPlaneModeManaged {
		return nil, errors.New("management composition requires managed control-plane mode")
	}
	if cfg.ManagementAPI.Auth.Mode != config.ManagementAuthModeRouter {
		return nil, errors.New("management composition requires Router-native authentication")
	}
	if len(cfg.ManagementAPI.Auth.Tokens) != 0 || len(cfg.ManagementAPI.Auth.Roles) != 0 {
		return nil, errors.New("management composition rejects static tokens and roles")
	}
	if cfg.AccessRuntimeStore == nil || cfg.AccessRuntimeStore.Redis.KeyPrefix == "" {
		return nil, errors.New("management composition requires a durable runtime key prefix")
	}
	if strings.TrimSpace(cfg.Agent.PublicInferenceEndpoint) == "" {
		return nil, errors.New("management composition requires the Agent public inference endpoint")
	}
	identityOverrides := 0
	if options.AssertionVerifier != nil {
		identityOverrides++
	}
	if options.IssuerKeyCache != nil {
		identityOverrides++
	}
	if options.BackchannelLogoutVerifier != nil {
		identityOverrides++
	}
	if identityOverrides != 0 && identityOverrides != 3 {
		return nil, errors.New("management identity verifier overrides must be supplied as one complete set")
	}
	var builtInRecipes *routingmanagement.BuiltInRecipeDistribution
	if options.BuiltInRecipes != nil {
		cloned := cloneBuiltInRecipeDistribution(*options.BuiltInRecipes)
		if err := cloned.Validate(); err != nil {
			return nil, fmt.Errorf("management built-in Recipe distribution is invalid: %w", err)
		}
		builtInRecipes = &cloned
	}
	builtInRecipeDirectory := ""
	if cfg.ConfigBaseDir != "" {
		builtInRecipeDirectory = filepath.Join(
			cfg.ConfigBaseDir, routingmanagement.BuiltInRecipeDistributionRelativeDirectory,
		)
	}
	bootstrap := *cfg
	validateRoutingPublication := func(snapshot *routingsnapshot.Snapshot) error {
		_, err := config.CompileManagedRoutingSnapshot(&bootstrap, snapshot)
		return err
	}
	return &Factory{
		keyPrefix:                  cfg.AccessRuntimeStore.Redis.KeyPrefix,
		agentInferenceEndpoint:     cfg.Agent.PublicInferenceEndpoint,
		assertionVerifier:          options.AssertionVerifier,
		issuerKeyCache:             options.IssuerKeyCache,
		backchannelLogoutVerifier:  options.BackchannelLogoutVerifier,
		modelProber:                options.ModelProber,
		builtInRecipes:             builtInRecipes,
		builtInRecipeDirectory:     builtInRecipeDirectory,
		defaultRevealable:          cfg.Access.Credentials.Reveal.Enabled,
		maxUsageBacklog:            cfg.Access.Enforcement.MaxUsageBacklog,
		mtlsListenerEnabled:        cfg.ManagementAPI.TLS.ClientCABundleFile != "" || cfg.ManagementAPI.TLS.ClientCABundleEnv != "",
		validateRoutingPublication: validateRoutingPublication,
		now:                        options.Now,
	}, nil
}

func (factory *Factory) Build(
	ctx context.Context,
	dependencies managedruntime.ManagementDependencies,
) (managedruntime.ManagedAPI, error) {
	if factory == nil || factory.keyPrefix == "" {
		return nil, errors.New("management composition factory is unavailable")
	}
	if err := validateDependencies(dependencies); err != nil {
		return nil, err
	}
	builtInRecipes, buildErr := factory.loadBuiltInRecipes()
	if buildErr != nil {
		return nil, buildErr
	}
	modelProber := factory.modelProber
	if modelProber == nil {
		modelProber = dependencies.ModelProber
	}
	if modelProber == nil {
		return nil, errors.New("management composition requires a Router-owned Model prober")
	}

	commandCodec, buildErr := managementcommand.NewCodec(
		dependencies.Keyrings.ControlPlane.ManagementCommand.Symmetric(),
	)
	if buildErr != nil {
		return nil, fmt.Errorf("compose Management command codec: %w", buildErr)
	}
	owned := &application{}
	owned.addCloser(commandCodec.Close)
	fail := func(err error) (managedruntime.ManagedAPI, error) {
		_ = owned.Close()
		return nil, err
	}
	assertions := factory.assertionVerifier
	issuerKeys := factory.issuerKeyCache
	logoutVerifier := factory.backchannelLogoutVerifier
	if assertions == nil {
		assertionComposition, err := composeAssertionVerifier(dependencies)
		if err != nil {
			return fail(err)
		}
		owned.addCloser(assertionComposition.Close)
		assertions = assertionComposition.Verifier()
		issuerKeys = assertionComposition.keys
		logoutVerifier = assertionComposition.verifier
	}

	accessIdentity, buildErr := composeAccessIdentity(
		dependencies,
		commandCodec,
		factory.defaultRevealable,
		factory.keyPrefix,
		factory.now,
	)
	if buildErr != nil {
		return fail(buildErr)
	}
	owned.addCloser(accessIdentity.Close)

	identity, buildErr := identityapplication.New(ctx, identityapplication.Options{
		Database: dependencies.Database, Valkey: dependencies.Redis,
		SessionStore: dependencies.SessionStore, KeyPrefix: factory.keyPrefix,
		CommandCodec: commandCodec,
		SessionTokenCodec: managementauth.TokenCodec{
			Keyring: cloneSigning(dependencies.Keyrings.ManagementSigning),
			Issuer:  managementTokenIssuer, Audience: managementTokenAudience,
			MaxSkew: managementTokenClockSkew,
		},
		ServiceCredentialPeppers:  pepperSymmetric(dependencies.Keyrings.ServiceAccounts),
		BootstrapToken:            dependencies.BootstrapToken,
		BootstrapTokenPresent:     dependencies.BootstrapTokenPresent,
		RecoveryToken:             dependencies.RecoveryToken,
		BootstrapIdempotencyKeys:  dependencies.Keyrings.ControlPlane.BootstrapIdempotency.Symmetric(),
		BootstrapResponseKEKs:     dependencies.Keyrings.ResponseKEK,
		WorkloadCursorKeyring:     dependencies.Keyrings.ControlPlane.ManagementCursor.Symmetric(),
		WorkloadResponseKEKs:      dependencies.Keyrings.ResponseKEK,
		WorkloadIdempotencyTTL:    defaultIdempotencyTTL,
		WorkloadSecretDeliveryTTL: defaultSecretDeliveryTTL,
		MTLSListenerEnabled:       factory.mtlsListenerEnabled,
		AssertionVerifier:         assertions,
		IssuerKeyCache:            issuerKeys,
		BackchannelLogoutVerifier: logoutVerifier,
		Now:                       factory.now,
		Exchanges:                 accessIdentity.exchanges,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose Management identity: %w", buildErr))
	}
	owned.addCloser(identity.Close)
	authorityStore, buildErr := authorizationpostgres.New(dependencies.Database)
	if buildErr != nil {
		return fail(fmt.Errorf("compose Management authority store: %w", buildErr))
	}
	authorizationRuntime := managementauthorization.Runtime{Loader: authorityStore}

	namespaceRepository, buildErr := accesspostgres.NewNamespaceManagementRepository(dependencies.AccessStore)
	if buildErr != nil {
		return fail(fmt.Errorf("compose Namespace repository: %w", buildErr))
	}
	namespaceService, buildErr := namespacemanagement.NewService(namespacemanagement.Options{
		Repository: namespaceRepository, CommandCodec: commandCodec,
		CursorKeyring:  dependencies.Keyrings.ControlPlane.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose Namespace Management: %w", buildErr))
	}
	owned.addCloser(func() error { namespaceService.Close(); return nil })
	namespaceRoutes, buildErr := managementserver.NewNamespaceRoutes(managementserver.NamespaceRoutesOptions{
		Service: namespaceService, Sessions: identity.SessionAuthenticator(),
		Authorization: identity.Authorizer(), Scopes: authorityStore, Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose Namespace Management routes: %w", buildErr))
	}

	namespaces := managementserver.ExplicitNamespaceResolver{}
	subjectRepository, buildErr := accesspostgres.NewSubjectRepository(dependencies.AccessStore)
	if buildErr != nil {
		return fail(fmt.Errorf("compose subject repository: %w", buildErr))
	}
	subjects, buildErr := subjectmanagement.NewService(subjectmanagement.Options{
		Repository: subjectRepository, CommandCodec: commandCodec,
		CursorKeyring:  dependencies.Keyrings.ControlPlane.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose subject Management: %w", buildErr))
	}
	owned.addCloser(func() error { subjects.Close(); return nil })
	subjectRoutes, buildErr := managementserver.NewSubjectRoutes(managementserver.SubjectRoutesOptions{
		Service: subjects, Namespaces: namespaces,
		Sessions: identity.SessionAuthenticator(), Authorization: identity.Authorizer(), Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose subject Management routes: %w", buildErr))
	}
	apiKeyRoutes, buildErr := managementserver.NewAPIKeyRoutes(managementserver.APIKeyRoutesOptions{
		Service: accessIdentity.apiKeys, Namespaces: namespaces,
		Sessions: identity.SessionAuthenticator(), Authorization: identity.Authorizer(), Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose API-key Management routes: %w", buildErr))
	}
	delegationRoutes, buildErr := managementserver.NewDelegationRoutes(managementserver.DelegationRoutesOptions{
		Service: accessIdentity.delegations, Namespaces: namespaces,
		Sessions: identity.SessionAuthenticator(), Authorization: identity.Authorizer(), Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose delegation Management routes: %w", buildErr))
	}
	invitationRoutes, buildErr := managementserver.NewInvitationRoutes(managementserver.InvitationRoutesOptions{
		Service: accessIdentity.invitations, Namespaces: namespaces,
		Sessions: identity.SessionAuthenticator(), Authorization: identity.Authorizer(), Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose invitation Management routes: %w", buildErr))
	}

	credentials, buildErr := credentialmanagement.NewService(credentialmanagement.Options{
		Repository: dependencies.AccessStore, Catalog: dependencies.Catalog.Catalog,
		Egress: dependencies.EgressPolicy, CredentialCodec: dependencies.ProviderCredentialCodec,
		CommandCodec:    commandCodec,
		CursorKeyring:   dependencies.Keyrings.ControlPlane.ManagementCursor.Symmetric(),
		IdempotencyTTL:  defaultIdempotencyTTL,
		RetiringOverlap: defaultCredentialRetirement, Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose ProviderCredential Management: %w", buildErr))
	}
	owned.addCloser(credentials.Close)
	credentialRoutes, buildErr := managementserver.NewProviderCredentialRoutes(managementserver.ProviderCredentialRoutesOptions{
		Service: credentials, Namespaces: namespaces,
		Sessions: identity.SessionAuthenticator(), Authorization: identity.Authorizer(), Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose ProviderCredential Management routes: %w", buildErr))
	}

	routing, buildErr := routingmanaged.NewApplication(routingmanaged.ApplicationOptions{
		DB: dependencies.Database,
		ModelCompiler: providercatalog.ModelCompiler{
			Catalog:     dependencies.Catalog.Coordinator,
			Registry:    dependencies.Catalog.Registry,
			Credentials: dependencies.AccessStore,
			Egress:      dependencies.EgressPolicy,
		},
		DiscoveryClaims:    dependencies.Catalog.Discovery.Claims,
		CredentialVersions: dependencies.ProviderCredentialResolver,
		Prober:             modelProber, ValidatePublication: factory.validateRoutingPublication,
		CommandCodec:   commandCodec,
		CursorKeyring:  dependencies.Keyrings.ControlPlane.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, Namespaces: namespaces,
		Sessions: identity.SessionAuthenticator(), Authorization: identity.Authorizer(),
		BuiltInRecipes: builtInRecipes, Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose Routing Management: %w", buildErr))
	}
	owned.addCloser(routing.Close)
	if err := routing.ReconcileBuiltInRecipes(ctx); err != nil {
		return fail(fmt.Errorf("install built-in Recipes: %w", err))
	}
	owned.workers = append(owned.workers, routing)

	policyRepository, buildErr := accesspostgres.NewPolicyManagementRepository(dependencies.AccessStore)
	if buildErr != nil {
		return fail(fmt.Errorf("compose policy repository: %w", buildErr))
	}
	policies, buildErr := policymanagement.NewService(policymanagement.Options{
		Repository: policyRepository, CommandCodec: commandCodec,
		CursorKeyring:  dependencies.Keyrings.ControlPlane.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose policy Management: %w", buildErr))
	}
	owned.addCloser(func() error { policies.Close(); return nil })

	executionAuthorizer, buildErr := managementserver.NewPolicyBulkExecutionAuthorizer(
		authorizationRuntime,
	)
	if buildErr != nil {
		return fail(fmt.Errorf("compose policy execution authorizer: %w", buildErr))
	}
	bulkRepository, buildErr := policybulkpostgres.NewRepository(dependencies.Database)
	if buildErr != nil {
		return fail(fmt.Errorf("compose policy operation repository: %w", buildErr))
	}
	bulk, buildErr := policybulk.NewService(policybulk.Options{
		Repository: bulkRepository, Policies: policies, Authorization: executionAuthorizer,
		CommandCodec:   commandCodec,
		CursorKeyring:  dependencies.Keyrings.ControlPlane.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, WorkerID: dependencies.ReplicaID,
		WorkerConcurrency: policyWorkerConcurrency, PollInterval: policyWorkerPollInterval,
		ClaimLease: policyWorkerClaimLease, MaximumAttempts: policyWorkerMaximumAttempts,
		Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose policy operation worker: %w", buildErr))
	}
	owned.addCloser(func() error { bulk.Close(); return nil })
	owned.workers = append(owned.workers, bulk)

	policyRoutes, buildErr := managementserver.NewPolicyRoutes(managementserver.PolicyRoutesOptions{
		Service: policies, Bulk: bulk, Namespaces: namespaces,
		Sessions: identity.SessionAuthenticator(), Authorization: identity.Authorizer(), Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose policy Management routes: %w", buildErr))
	}
	quotaRuntime, buildErr := quotaruntime.NewRedisEngine(dependencies.Redis, quotaruntime.RedisEngineOptions{
		KeyPrefix: factory.keyPrefix,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose quota reconciliation runtime: %w", buildErr))
	}
	appliedPolicies, buildErr := accessruntime.NewRedisProjectionReader(accessruntime.RedisProjectionReaderOptions{
		Client: dependencies.Redis, KeyPrefix: factory.keyPrefix,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose applied access-policy reader: %w", buildErr))
	}
	publicationWaiter, buildErr := delegationmanagement.NewRedisPublicationWaiter(dependencies.Redis, factory.keyPrefix)
	if buildErr != nil {
		return fail(fmt.Errorf("compose access publication waiter: %w", buildErr))
	}
	accessReads, buildErr := accessmanagement.NewService(accessmanagement.ServiceOptions{
		Repository: dependencies.AccessStore, Applied: appliedPolicies,
		Meters: quotaRuntime, Waiter: publicationWaiter,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose access read service: %w", buildErr))
	}
	accessReadRoutes, buildErr := managementserver.NewAccessReadRoutes(managementserver.AccessReadRoutesOptions{
		Service: accessReads, Namespaces: namespaces,
		Sessions: identity.SessionAuthenticator(), Authorization: identity.Authorizer(), Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose access read routes: %w", buildErr))
	}
	quotaReconciliation, buildErr := quotareconciliation.NewService(quotareconciliation.Options{
		Repository: dependencies.AccessStore, Runtime: quotaRuntime, WaiveAuth: dependencies.AccessStore,
		CommandCodec:   commandCodec,
		CursorKeyring:  dependencies.Keyrings.ControlPlane.ManagementCursor.Symmetric(),
		IdempotencyTTL: defaultIdempotencyTTL, WorkerID: dependencies.ReplicaID,
		WorkerConcurrency: quotaReconciliationWorkers, PollInterval: policyWorkerPollInterval,
		ClaimLease: policyWorkerClaimLease, Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose quota reconciliation: %w", buildErr))
	}
	owned.addCloser(func() error { quotaReconciliation.Close(); return nil })
	owned.workers = append(owned.workers, quotaReconciliation)
	unknownUsageRoutes, buildErr := managementserver.NewUnknownUsageRoutes(managementserver.UnknownUsageRoutesOptions{
		Service: quotaReconciliation, Namespaces: namespaces,
		Sessions: identity.SessionAuthenticator(), Authorization: identity.Authorizer(), Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose unknown-usage Management routes: %w", buildErr))
	}
	unknownUsageOperationReader, buildErr := managementserver.NewUnknownUsageOperationDetailReader(
		quotaReconciliation, identity.Authorizer(),
	)
	if buildErr != nil {
		return fail(fmt.Errorf("compose unknown-usage Operation reader: %w", buildErr))
	}
	operationRoutes, buildErr := managementserver.NewOperationRoutes(managementserver.OperationRoutesOptions{
		Service: bulk, DetailReaders: []managementserver.OperationDetailReader{unknownUsageOperationReader},
		Namespaces: namespaces, Sessions: identity.SessionAuthenticator(),
		Authorization: identity.Authorizer(), Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose Management Operation routes: %w", buildErr))
	}
	observability, buildErr := composeObservability(
		dependencies, authorizationRuntime, namespaces, identity.SessionAuthenticator(), identity.Authorizer(),
		subjects, accessIdentity.apiKeys, factory.now,
	)
	if buildErr != nil {
		return fail(fmt.Errorf("compose Management observability: %w", buildErr))
	}
	owned.addCloser(observability.Close)
	statistics, buildErr := composeStatistics(
		dependencies, authorizationRuntime, namespaces, identity.SessionAuthenticator(), factory.now,
	)
	if buildErr != nil {
		return fail(fmt.Errorf("compose Management statistics: %w", buildErr))
	}
	runtimeDiagnostics, buildErr := composeRuntimeDiagnostics(
		dependencies, factory.keyPrefix, factory.maxUsageBacklog,
		identity.SessionAuthenticator(), identity.Authorizer(), factory.now,
	)
	if buildErr != nil {
		return fail(fmt.Errorf("compose Management runtime diagnostics: %w", buildErr))
	}
	agentRuntime, buildErr := composeAgentRuntime(
		ctx, dependencies, commandCodec, authorityStore, authorizationRuntime,
		namespaces, identity.SessionAuthenticator(), identity.Authorizer(), routing,
		builtInRecipes, factory.agentInferenceEndpoint, factory.keyPrefix, factory.now,
	)
	if buildErr != nil {
		return fail(fmt.Errorf("compose Router-native Agent runtime: %w", buildErr))
	}
	owned.addCloser(agentRuntime.Close)
	owned.workers = append(owned.workers, agentRuntime.workers...)

	server, buildErr := dependencies.Catalog.NewManagementServer(catalogmanaged.ManagementServerOptions{
		Namespaces: namespaces, Sessions: identity.SessionAuthenticator(),
		Authorization: identity.Authorizer(), AdditionalRoutes: []managementserver.RouteRegistrar{
			identity, namespaceRoutes, subjectRoutes, apiKeyRoutes, delegationRoutes, invitationRoutes,
			credentialRoutes, routing, policyRoutes, accessReadRoutes, operationRoutes, unknownUsageRoutes,
			observability.routes, statistics, runtimeDiagnostics, agentRuntime.routes,
		}, Now: factory.now,
	})
	if buildErr != nil {
		return fail(fmt.Errorf("compose managed HTTP server: %w", buildErr))
	}
	disabledOperations := []string(nil)
	if len(dependencies.RecoveryToken) == 0 {
		disabledOperations = append(disabledOperations, "postAuthRecovery")
	}
	if err := managementserver.ValidateRegisteredOperations(server, disabledOperations...); err != nil {
		return fail(fmt.Errorf("validate Management registry route coverage: %w", err))
	}
	owned.server = server
	return owned, nil
}

func (factory *Factory) loadBuiltInRecipes() (routingmanagement.BuiltInRecipeDistribution, error) {
	if factory.builtInRecipes != nil {
		return cloneBuiltInRecipeDistribution(*factory.builtInRecipes), nil
	}
	if factory.builtInRecipeDirectory == "" {
		return routingmanagement.BuiltInRecipeDistribution{}, errors.New(
			"management composition requires a canonical built-in Recipe distribution directory",
		)
	}
	distribution, err := routingmanagement.LoadBuiltInRecipeDistribution(factory.builtInRecipeDirectory)
	if err != nil {
		return routingmanagement.BuiltInRecipeDistribution{}, fmt.Errorf("load built-in Recipe distribution: %w", err)
	}
	return distribution, nil
}

func cloneBuiltInRecipeDistribution(
	source routingmanagement.BuiltInRecipeDistribution,
) routingmanagement.BuiltInRecipeDistribution {
	result := source
	result.Recipes = make([]routingmanagement.BuiltInRecipe, len(source.Recipes))
	for index, member := range source.Recipes {
		result.Recipes[index] = member
		result.Recipes[index].Input.Document = append([]byte(nil), member.Input.Document...)
	}
	return result
}

func validateDependencies(dependencies managedruntime.ManagementDependencies) error {
	if dependencies.Database == nil || dependencies.Redis == nil || dependencies.AccessStore == nil ||
		dependencies.SessionStore == nil || dependencies.Catalog == nil || dependencies.Catalog.Catalog == nil ||
		dependencies.Catalog.Coordinator == nil || dependencies.Catalog.Registry == nil ||
		dependencies.Catalog.Discovery == nil || strings.TrimSpace(dependencies.DelegationAudience) == "" {
		return errors.New("management composition dependencies are incomplete")
	}
	if !validPolicyWorkerID(dependencies.ReplicaID) {
		return errors.New("management composition requires a stable replica identity")
	}
	return nil
}

func validPolicyWorkerID(value string) bool {
	if value == "" || len(value) > 128 || strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if character < 0x20 || character == 0x7f {
			return false
		}
	}
	return true
}

type managementHTTP interface {
	Register(*http.ServeMux)
	Ready(context.Context) error
}

type backgroundWorker interface {
	Run(context.Context) error
}

type application struct {
	server    managementHTTP
	workers   []backgroundWorker
	closers   []func() error
	closeOnce sync.Once
	closeErr  error
}

func (application *application) Register(mux *http.ServeMux) {
	if application == nil || application.server == nil {
		panic("Management composition is unavailable")
	}
	application.server.Register(mux)
}

func (application *application) Ready(ctx context.Context) error {
	if application == nil || application.server == nil {
		return errors.New("management composition is unavailable")
	}
	return application.server.Ready(ctx)
}

func (application *application) Run(ctx context.Context) error {
	if application == nil || len(application.workers) == 0 {
		return errors.New("management composition worker is unavailable")
	}
	workerContext, cancel := context.WithCancel(ctx)
	defer cancel()
	errorsByWorker := make(chan error, len(application.workers))
	for _, worker := range application.workers {
		go func(active backgroundWorker) { errorsByWorker <- active.Run(workerContext) }(worker)
	}
	var failures []error
	for range application.workers {
		err := <-errorsByWorker
		switch {
		case workerContext.Err() != nil && (err == nil || errors.Is(err, context.Canceled)):
			// A sibling already stopped the application, or the owner cancelled it.
		case err == nil:
			failures = append(failures, errors.New("management worker exited before cancellation"))
			cancel()
		case !errors.Is(err, context.Canceled):
			failures = append(failures, err)
			cancel()
		}
	}
	if len(failures) > 0 {
		return errors.Join(failures...)
	}
	return ctx.Err()
}

func (application *application) addCloser(closer func() error) {
	if application == nil || closer == nil {
		panic("Management composition closer is required")
	}
	application.closers = append(application.closers, closer)
}

func (application *application) Close() error {
	if application == nil {
		return nil
	}
	application.closeOnce.Do(func() {
		var closeErrors []error
		for index := len(application.closers) - 1; index >= 0; index-- {
			closeErrors = append(closeErrors, application.closers[index]())
		}
		application.server = nil
		application.workers = nil
		application.closers = nil
		application.closeErr = errors.Join(closeErrors...)
	})
	return application.closeErr
}

func cloneSigning(source securitykeyring.Signing) securitykeyring.Signing {
	result := securitykeyring.Signing{
		ActiveVersion: source.ActiveVersion,
		Private:       make(map[string]ed25519.PrivateKey, len(source.Private)),
		Public:        make(map[string]ed25519.PublicKey, len(source.Public)),
	}
	for version, key := range source.Private {
		result.Private[version] = ed25519.PrivateKey(append([]byte(nil), key...))
	}
	for version, key := range source.Public {
		result.Public[version] = ed25519.PublicKey(append([]byte(nil), key...))
	}
	return result
}

func pepperSymmetric(source accesscredential.PepperKeyring) securitykeyring.Symmetric {
	return securitykeyring.Symmetric{ActiveVersion: source.ActiveVersion, Keys: source.Keys}
}

var (
	_ managedruntime.ManagementFactory = (*Factory)(nil)
	_ managedruntime.ManagedAPI        = (*application)(nil)
)
