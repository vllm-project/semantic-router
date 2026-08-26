package routingruntime

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	_ "github.com/lib/pq"
	"github.com/redis/go-redis/v9"

	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/modelprobe"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	catalogapplication "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog/application"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	managementauthpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/postgres"
	managementauthredis "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/redis"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/backendresolver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotarecovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

const (
	defaultStartupTimeout    = 10 * time.Second
	defaultDiscoveryClaimTTL = 5 * time.Minute
)

// New creates the strict file-authoritative or durable composition. Construction
// opens each durable client once, validates it, and never loads product
// provider metadata in an inference data-plane component.
func New(ctx context.Context, cfg *config.RouterConfig, options Options) (*Runtime, error) {
	if cfg == nil {
		return nil, errors.New("router configuration is required")
	}
	capabilities, err := runtimecapabilities.Derive(cfg)
	if err != nil {
		return nil, fmt.Errorf("derive runtime capabilities: %w", err)
	}
	if capabilities.FileRouting {
		if !nilLike(options.ManagementFactory) {
			return nil, errors.New("file-authoritative routing rejects a Management factory")
		}
		return newFileAuthorityRuntime(cfg, options)
	}
	if capabilities.ManagementAPI && nilLike(options.ManagementFactory) {
		return nil, errors.New("enabled Management API requires a Management factory")
	}
	if !capabilities.ManagementAPI && !nilLike(options.ManagementFactory) {
		return nil, errors.New("disabled Management API rejects a Management factory")
	}
	timeout := options.StartupTimeout
	if timeout == 0 {
		timeout = defaultStartupTimeout
	}
	if timeout <= 0 {
		return nil, errors.New("durable routing startup timeout must be positive")
	}
	startupContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return newDurableRuntime(startupContext, cfg, capabilities, options)
}

type durableFoundation struct {
	protocolCodecs        *protocolcodec.Registry
	credentialAdapters    backendresolver.StaticRegistry
	discoveryAdapters     *providerdiscovery.Registry
	providerRegistry      *providercatalog.Registry
	bootstrapToken        []byte
	bootstrapTokenPresent func() (bool, error)
	recoveryToken         []byte
	database              *sql.DB
	postgresDSN           string
	redisClient           *redis.Client
	egressPolicy          backendegress.Policy
	accessStore           *accesspostgres.Store
	sessionStore          *managementauthpostgres.Store
	providerCodec         providercredential.Codec
	providerResolver      backendresolver.Resolver
	replicaOptions        catalogapplication.ReplicaOptions
	catalogApplication    *catalogapplication.Application
	modelProber           *modelprobe.Prober
}

func newDurableRuntime(
	ctx context.Context,
	cfg *config.RouterConfig,
	capabilities runtimecapabilities.RuntimeCapabilities,
	options Options,
) (_ *Runtime, resultErr error) {
	keyrings, err := loadDeploymentKeyrings(cfg)
	if err != nil {
		return nil, err
	}
	runtime := &Runtime{capabilities: capabilities, keyrings: keyrings}
	defer func() {
		if resultErr != nil {
			_ = runtime.Close()
		}
	}()
	foundation, err := newDurableFoundation(ctx, cfg, capabilities, options, runtime)
	if err != nil {
		return nil, err
	}
	defer zero(foundation.bootstrapToken)
	defer zero(foundation.recoveryToken)
	quotaEngine, err := composeNativeAccess(runtime, cfg, capabilities, foundation)
	if err != nil {
		return nil, err
	}
	if err := composeDispatchAuthority(ctx, runtime, cfg, capabilities, foundation); err != nil {
		return nil, err
	}
	if err := composePublicationRuntime(runtime, cfg, capabilities, options, foundation, quotaEngine); err != nil {
		return nil, err
	}
	if err := composeManagementRuntime(ctx, runtime, cfg, capabilities, options, foundation); err != nil {
		return nil, err
	}
	return runtime, nil
}

func newDurableFoundation(
	ctx context.Context,
	cfg *config.RouterConfig,
	capabilities runtimecapabilities.RuntimeCapabilities,
	options Options,
	runtime *Runtime,
) (_ *durableFoundation, resultErr error) {
	foundation := &durableFoundation{bootstrapTokenPresent: func() (bool, error) { return false, nil }}
	defer func() {
		if resultErr != nil {
			zero(foundation.bootstrapToken)
			zero(foundation.recoveryToken)
		}
	}()
	if err := composeProviderFoundation(cfg, capabilities, options, runtime, foundation); err != nil {
		return nil, err
	}
	if err := openDurableFoundation(ctx, cfg, capabilities, runtime, foundation); err != nil {
		return nil, err
	}
	if err := composeCatalogFoundation(ctx, cfg, capabilities, options, runtime, foundation); err != nil {
		return nil, err
	}
	return foundation, nil
}

func composeProviderFoundation(
	cfg *config.RouterConfig,
	capabilities runtimecapabilities.RuntimeCapabilities,
	options Options,
	runtime *Runtime,
	foundation *durableFoundation,
) error {
	foundation.protocolCodecs = protocolcodec.NewBuiltinRegistry()
	runtime.protocolCodecs = foundation.protocolCodecs
	var err error
	foundation.credentialAdapters, err = backendresolver.BuiltinRegistry()
	if err != nil {
		return fmt.Errorf("compose provider credential adapters: %w", err)
	}
	foundation.discoveryAdapters, err = providerdiscovery.BuiltinRegistry()
	if err != nil {
		return fmt.Errorf("compose provider discovery adapters: %w", err)
	}
	foundation.providerRegistry, err = composeProviderRegistry(
		options.ProviderIntegrations, options.ProviderBackendCompilers,
		foundation.protocolCodecs, foundation.credentialAdapters, foundation.discoveryAdapters,
	)
	if err != nil {
		return err
	}
	if !capabilities.ManagementAPI {
		return nil
	}
	foundation.bootstrapToken, foundation.bootstrapTokenPresent, err = readBootstrapToken(
		cfg.ManagementAPI.Auth.Bootstrap.TokenFile,
		cfg.ManagementAPI.Auth.Bootstrap.TokenEnv,
	)
	if err != nil {
		return fmt.Errorf("read Management bootstrap token: %w", err)
	}
	foundation.recoveryToken, err = readRecoveryToken(
		cfg.ManagementAPI.Auth.Recovery.Enabled,
		cfg.ManagementAPI.Auth.Recovery.TokenFile,
		cfg.ManagementAPI.Auth.Recovery.TokenEnv,
	)
	if err != nil {
		return fmt.Errorf("read Management recovery token: %w", err)
	}
	return nil
}

func openDurableFoundation(
	ctx context.Context,
	cfg *config.RouterConfig,
	capabilities runtimecapabilities.RuntimeCapabilities,
	runtime *Runtime,
	foundation *durableFoundation,
) error {
	var err error
	foundation.database, foundation.postgresDSN, err = openPostgreSQL(ctx, cfg.AccessStore.Postgres)
	if err != nil {
		return err
	}
	runtime.database = foundation.database
	if capabilities.DistributedState {
		foundation.redisClient, err = openValkey(ctx, cfg.AccessRuntimeStore.Redis)
		if err != nil {
			return err
		}
		client := foundation.redisClient
		runtime.redis = client
		runtime.redisReady = func(ctx context.Context) error { return client.Ping(ctx).Err() }
		runtime.redisClose = client.Close
		runtime.accessKeyPrefix = cfg.AccessRuntimeStore.Redis.KeyPrefix
	}
	if capabilities.NativeAccess {
		runtime.responseTerminals, err = composeDistributedResponseTerminalStore(
			foundation.redisClient, cfg.AccessRuntimeStore.Redis.KeyPrefix,
		)
		if err != nil {
			return fmt.Errorf("compose shared response terminal store: %w", err)
		}
	} else {
		runtime.responseTerminals = backendinvoker.NewLocalResponseTerminalStore()
	}
	foundation.egressPolicy, err = backendegress.LoadFile(cfg.BackendEgress.PolicyFile)
	if err != nil {
		return fmt.Errorf("load backend egress policy: %w", err)
	}
	foundation.accessStore, err = accesspostgres.New(foundation.database)
	if err != nil {
		return err
	}
	if capabilities.ManagementAPI || capabilities.NativeAccess {
		foundation.sessionStore, err = managementauthpostgres.New(foundation.database)
		if err != nil {
			return err
		}
	}
	foundation.providerCodec = providercredential.Codec{Keyring: runtime.keyrings.ProviderKEK}
	if err := seedDurableRoutingDatabase(
		ctx, foundation.database, cfg, foundation.providerCodec, foundation.providerRegistry.Snapshot(),
	); err != nil {
		return fmt.Errorf("seed durable routing authority: %w", err)
	}
	foundation.providerResolver = backendresolver.Resolver{
		Loader: foundation.accessStore, Codec: foundation.providerCodec, Registry: foundation.credentialAdapters,
	}
	return nil
}

func composeCatalogFoundation(
	ctx context.Context,
	cfg *config.RouterConfig,
	capabilities runtimecapabilities.RuntimeCapabilities,
	options Options,
	runtime *Runtime,
	foundation *durableFoundation,
) error {
	discoveryClaims, err := providerdiscovery.NewClaimCodec(providerdiscovery.ClaimKeyset{
		ActiveKeyID: runtime.keyrings.Routing.DiscoveryClaim.ActiveVersion(),
		Keys:        runtime.keyrings.Routing.DiscoveryClaim.Symmetric().Keys,
	})
	if err != nil {
		return fmt.Errorf("compose provider discovery claims: %w", err)
	}
	foundation.replicaOptions, err = catalogReplicaOptions(options, capabilities)
	if err != nil {
		return err
	}
	runtime.replicaID = foundation.replicaOptions.ReplicaID
	discoveryTTL := options.DiscoveryClaimTTL
	if discoveryTTL == 0 {
		discoveryTTL = defaultDiscoveryClaimTTL
	}
	foundation.catalogApplication, err = catalogapplication.NewApplication(catalogapplication.ApplicationOptions{
		DB: foundation.database, Registry: foundation.providerRegistry,
		DiscoveryAdapters: foundation.discoveryAdapters, CredentialMetadata: foundation.accessStore,
		Credentials: foundation.providerResolver, EgressPolicy: foundation.egressPolicy,
		DialTimeout:       options.BackendDialTimeout,
		CatalogCursorKeys: runtime.keyrings.Routing.CatalogCursor.Symmetric(),
		DiscoveryClaims:   discoveryClaims, DiscoveryClaimTTL: discoveryTTL,
		Replica: foundation.replicaOptions,
	})
	if err != nil {
		return fmt.Errorf("compose Provider Catalog: %w", err)
	}
	runtime.catalog = foundation.catalogApplication.Replica
	runtime.catalogClose = foundation.catalogApplication.Close
	foundation.modelProber, err = modelprobe.New(modelprobe.Options{
		Credentials: foundation.providerResolver, Codecs: foundation.protocolCodecs,
		Transport: foundation.catalogApplication.Discovery.Transport,
	})
	if err != nil {
		return fmt.Errorf("compose Model probe: %w", err)
	}
	return nil
}

func composeNativeAccess(
	runtime *Runtime,
	cfg *config.RouterConfig,
	capabilities runtimecapabilities.RuntimeCapabilities,
	foundation *durableFoundation,
) (*quotaruntime.RedisEngine, error) {
	if !capabilities.NativeAccess {
		return nil, nil
	}
	delegationBarriers, err := managementauthredis.New(managementauthredis.Options{
		Client: foundation.redisClient, KeyPrefix: cfg.AccessRuntimeStore.Redis.KeyPrefix,
		Loader: foundation.sessionStore,
	})
	if err != nil {
		return nil, fmt.Errorf("compose delegated inference revocation barriers: %w", err)
	}
	access, quotaEngine, err := buildInferenceAccess(
		foundation.redisClient, cfg.AccessRuntimeStore.Redis, runtime.keyrings, delegationBarriers,
	)
	if err != nil {
		return nil, err
	}
	runtime.access = access
	outcomeRepository, err := outcomefeedback.NewPostgresRepository(foundation.database)
	if err != nil {
		return nil, fmt.Errorf("compose outcome feedback repository: %w", err)
	}
	outcomeLimiter, err := outcomefeedback.NewRedisAbuseLimiter(outcomefeedback.RedisAbuseLimiterOptions{
		Client: foundation.redisClient, KeyPrefix: cfg.AccessRuntimeStore.Redis.KeyPrefix,
	})
	if err != nil {
		return nil, fmt.Errorf("compose outcome feedback abuse limit: %w", err)
	}
	runtime.outcomeProjection, err = outcomefeedback.NewRedisProjectionStore(outcomefeedback.RedisProjectionStoreOptions{
		Client: foundation.redisClient, KeyPrefix: cfg.AccessRuntimeStore.Redis.KeyPrefix,
	})
	if err != nil {
		return nil, fmt.Errorf("compose outcome feedback projection store: %w", err)
	}
	runtime.outcomeFeedback, err = outcomefeedback.NewService(outcomefeedback.ServiceOptions{
		Repository: outcomeRepository, Limiter: outcomeLimiter,
	})
	if err != nil {
		return nil, fmt.Errorf("compose outcome feedback service: %w", err)
	}
	runtime.outcomeProjector, err = outcomefeedback.NewProjector(outcomefeedback.ProjectorOptions{
		Repository: outcomeRepository, Publisher: runtime.outcomeProjection,
	})
	if err != nil {
		return nil, fmt.Errorf("compose outcome feedback projector: %w", err)
	}
	return quotaEngine, nil
}

func composeDispatchAuthority(
	ctx context.Context,
	runtime *Runtime,
	cfg *config.RouterConfig,
	capabilities runtimecapabilities.RuntimeCapabilities,
	foundation *durableFoundation,
) error {
	capabilityLifetime, err := cfg.BackendDispatch.CapabilityLifetime()
	if err != nil {
		return err
	}
	capabilityKeys := runtime.keyrings.Routing.BackendDispatch.Symmetric()
	defer zeroSymmetric(&capabilityKeys)
	issuerOptions := backendinvoker.CapabilityIssuerOptions{
		Audience: cfg.BackendDispatch.Audience,
		Keyring: backendinvoker.SigningKeyring{
			ActiveVersion: capabilityKeys.ActiveVersion,
			Keys:          capabilityKeys.Keys, MaxLifetime: capabilityLifetime,
		},
		Lifetime: capabilityLifetime,
	}
	publicNamespaceID := ""
	if !capabilities.NativeAccess {
		publicNamespaceID, err = resolvePublicRoutingNamespace(ctx, foundation.database)
		if err != nil {
			return err
		}
		runtime.publicNamespaceID = publicNamespaceID
	}
	if capabilities.NativeAccess {
		runtime.dispatchCapabilities, err = dispatchauthority.NewMeteredRuntime(
			dispatchauthority.MeteredAuthorityOptions{Access: runtime.access, Issuer: issuerOptions},
		)
	} else {
		runtime.dispatchCapabilities, err = dispatchauthority.NewRoutingOnlyRuntime(
			dispatchauthority.RoutingOnlyAuthorityOptions{
				NamespaceID: publicNamespaceID, Publications: runtime, Issuer: issuerOptions,
			},
		)
	}
	if err != nil {
		return fmt.Errorf("compose backend dispatch authority: %w", err)
	}
	return nil
}

func composePublicationRuntime(
	runtime *Runtime,
	cfg *config.RouterConfig,
	capabilities runtimecapabilities.RuntimeCapabilities,
	options Options,
	foundation *durableFoundation,
	quotaEngine *quotaruntime.RedisEngine,
) error {
	processor, worker, publicationStore, err := composePublicationPipeline(
		foundation.database, foundation.redisClient, runtimeStoreConfig(cfg),
		foundation.replicaOptions.ReplicaID, foundation.postgresDSN, capabilities.NativeAccess,
	)
	if err != nil {
		return err
	}
	runtime.publisherProcessor, runtime.publisherWorker = processor, worker
	runtime.publicationCoordinator = publicationStore
	if closer, ok := publicationStore.(interface{ Close() error }); ok {
		runtime.publicationCloser = closer
	}
	inferenceProviderResolver, err := composeInferenceProviderCredentialResolver(
		publicationStore, capabilities.NativeAccess, foundation.providerCodec, foundation.credentialAdapters,
	)
	if err != nil {
		return err
	}
	if capabilities.NativeAccess {
		runtime.usageSupervisor, err = composeUsageSupervisor(
			foundation.database, foundation.redisClient, cfg.AccessRuntimeStore.Redis,
			cfg.Access.UsageStorage, foundation.replicaOptions.ReplicaID,
		)
		if err != nil {
			return err
		}
		namespaces, namespaceErr := usageledger.NewPostgresNamespaceSource(foundation.database)
		if namespaceErr != nil {
			return fmt.Errorf("compose quota recovery namespace source: %w", namespaceErr)
		}
		runtime.quotaRecovery, err = quotarecovery.NewSupervisor(quotarecovery.SupervisorOptions{
			Namespaces: namespaces, Runtime: quotaEngine,
		})
		if err != nil {
			return fmt.Errorf("compose quota recovery supervisor: %w", err)
		}
	}
	var journal backendinvoker.Journal = backendinvoker.ProcessLocalJournal{}
	if capabilities.NativeAccess {
		journal, err = backendinvoker.NewAuthoritativeAttemptJournal(quotaEngine)
		if err != nil {
			return fmt.Errorf("compose authoritative attempt journal: %w", err)
		}
	}
	runtime.backendDispatch, err = newBackendDispatchComposition(
		cfg.BackendDispatch, inferenceProviderResolver, foundation.protocolCodecs, journal,
		runtime.responseTerminals, foundation.egressPolicy, options.BackendDialTimeout,
		runtime.Ready,
	)
	return err
}

func composeManagementRuntime(
	ctx context.Context,
	runtime *Runtime,
	cfg *config.RouterConfig,
	capabilities runtimecapabilities.RuntimeCapabilities,
	options Options,
	foundation *durableFoundation,
) error {
	if !capabilities.ManagementAPI {
		return nil
	}
	issuerEgressPolicy := foundation.egressPolicy
	if options.ManagementIssuerEgressPolicy != nil {
		var err error
		issuerEgressPolicy, err = backendegress.Overlay(
			foundation.egressPolicy,
			*options.ManagementIssuerEgressPolicy,
		)
		if err != nil {
			return fmt.Errorf("compose Management issuer egress policy: %w", err)
		}
	}
	factoryKeyrings := runtime.keyrings.clone()
	defer factoryKeyrings.zero()
	management, err := options.ManagementFactory.Build(ctx, ManagementDependencies{
		Database: foundation.database, Redis: foundation.redisClient,
		AccessStore: foundation.accessStore, SessionStore: foundation.sessionStore,
		Catalog: foundation.catalogApplication, EgressPolicy: foundation.egressPolicy,
		IssuerEgressPolicy: issuerEgressPolicy,
		ProtocolCodecs:     foundation.protocolCodecs, CredentialAdapters: foundation.credentialAdapters,
		DiscoveryAdapters: foundation.discoveryAdapters, ProviderCredentialCodec: foundation.providerCodec,
		ProviderCredentialResolver: foundation.providerResolver,
		BootstrapToken:             foundation.bootstrapToken, BootstrapTokenPresent: foundation.bootstrapTokenPresent,
		RecoveryToken: foundation.recoveryToken, ModelProber: foundation.modelProber,
		Keyrings: factoryKeyrings, ReplicaID: foundation.replicaOptions.ReplicaID,
		DelegationAudience: InferenceDelegationAudience,
	})
	if err != nil {
		return fmt.Errorf("compose Management API: %w", err)
	}
	if nilLike(management) {
		return errors.New("management factory returned no application")
	}
	runtime.management = management
	return nil
}

func composeDistributedResponseTerminalStore(
	client redis.UniversalClient,
	keyPrefix string,
) (backendinvoker.ResponseTerminalStore, error) {
	return backendinvoker.NewRedisResponseTerminalStore(
		backendinvoker.RedisResponseTerminalStoreOptions{Client: client, KeyPrefix: keyPrefix},
	)
}

func composeInferenceProviderCredentialResolver(
	publicationStore publicationCoordinator,
	nativeAccess bool,
	codec providercredential.Codec,
	registry backendresolver.StaticRegistry,
) (backendresolver.PublishedResolver, error) {
	if publicationStore == nil {
		return backendresolver.PublishedResolver{}, errors.New("published provider credential store is required")
	}
	loader, ok := publicationStore.(backendresolver.PublishedLoader)
	if !ok {
		if nativeAccess {
			return backendresolver.PublishedResolver{}, errors.New("runtime publication credential projection is unavailable")
		}
		return backendresolver.PublishedResolver{}, errors.New("PostgreSQL publication credential authority is unavailable")
	}
	return backendresolver.PublishedResolver{
		Loader: loader, Codec: codec, Registry: registry,
	}, nil
}

func openPostgreSQL(ctx context.Context, store config.PostgresAccessStoreConfig) (*sql.DB, string, error) {
	dsn, err := readScalarSecret(store.DSNFile, store.DSNEnv)
	if err != nil {
		return nil, "", fmt.Errorf("read durable PostgreSQL DSN: %w", err)
	}
	database, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, "", errors.New("open durable PostgreSQL")
	}
	database.SetMaxOpenConns(store.MaxConnections)
	idle := store.MaxConnections / 2
	if idle < 1 {
		idle = 1
	}
	database.SetMaxIdleConns(idle)
	database.SetConnMaxLifetime(30 * time.Minute)
	database.SetConnMaxIdleTime(5 * time.Minute)
	if err := database.PingContext(ctx); err != nil {
		_ = database.Close()
		return nil, "", errors.New("durable PostgreSQL readiness failed")
	}
	return database, dsn, nil
}

func openValkey(ctx context.Context, store config.RedisAccessRuntimeStoreConfig) (*redis.Client, error) {
	redisURL, err := readScalarSecret(store.URLFile, store.URLEnv)
	if err != nil {
		return nil, fmt.Errorf("read distributed runtime store URL: %w", err)
	}
	options, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, errors.New("parse distributed runtime store URL")
	}
	client := redis.NewClient(options)
	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		return nil, errors.New("distributed runtime store readiness failed")
	}
	return client, nil
}

func buildInferenceAccess(
	client *redis.Client,
	store config.RedisAccessRuntimeStoreConfig,
	keyrings DeploymentKeyrings,
	delegationBarriers managementauth.DelegationRevocationBarrierStore,
) (*accessruntime.Runtime, *quotaruntime.RedisEngine, error) {
	reader, err := accessruntime.NewRedisProjectionReader(accessruntime.RedisProjectionReaderOptions{
		Client: client, KeyPrefix: store.KeyPrefix,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("compose access projection reader: %w", err)
	}
	engine, err := quotaruntime.NewRedisEngine(client, quotaruntime.RedisEngineOptions{KeyPrefix: store.KeyPrefix})
	if err != nil {
		return nil, nil, fmt.Errorf("compose quota runtime: %w", err)
	}
	runtime, err := accessruntime.New(accessruntime.RuntimeOptions{
		Reader: reader, Engine: engine,
		APIKeyPeppers: keyrings.APIKeyPeppers, DelegationPeppers: keyrings.DelegationPeppers,
		DelegationAudience: InferenceDelegationAudience, DelegationBarriers: delegationBarriers,
		KeyPrefix: store.KeyPrefix,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("compose inference access runtime: %w", err)
	}
	return runtime, engine, nil
}

func catalogReplicaOptions(
	options Options,
	capabilities runtimecapabilities.RuntimeCapabilities,
) (catalogapplication.ReplicaOptions, error) {
	replicaID := strings.TrimSpace(options.ReplicaID)
	if replicaID == "" || replicaID != options.ReplicaID {
		return catalogapplication.ReplicaOptions{}, errors.New("provider Catalog replica identity is unavailable")
	}
	groups := append([]providercatalog.RolloutGroup(nil), options.ProviderCatalogGroups...)
	if len(groups) == 0 {
		groups = []providercatalog.RolloutGroup{{Plane: providercatalog.CapabilityPlaneData, ID: "router"}}
		if capabilities.ManagementAPI {
			groups = append(groups, providercatalog.RolloutGroup{
				Plane: providercatalog.CapabilityPlaneControl, ID: "management",
			})
		}
	}
	required := append([]providercatalog.RolloutGroup(nil), options.RequiredCatalogGroups...)
	if len(required) == 0 {
		required = append(required, groups...)
	}
	return catalogapplication.ReplicaOptions{
		ReplicaID: replicaID, Lease: options.ProviderCatalogLease,
		RenewInterval: options.ProviderCatalogRenew,
		RolloutGroups: groups, RequiredRolloutGroups: required,
	}, nil
}

func runtimeStoreConfig(cfg *config.RouterConfig) config.RedisAccessRuntimeStoreConfig {
	if cfg == nil || cfg.AccessRuntimeStore == nil {
		return config.RedisAccessRuntimeStoreConfig{}
	}
	return cfg.AccessRuntimeStore.Redis
}

func resolvePublicRoutingNamespace(ctx context.Context, database *sql.DB) (string, error) {
	if database == nil {
		return "", errors.New("durable routing database is unavailable")
	}
	rows, err := database.QueryContext(ctx, `SELECT id FROM access_namespaces
WHERE status='active' ORDER BY id LIMIT 2`)
	if err != nil {
		return "", fmt.Errorf("resolve public routing namespace: %w", err)
	}
	defer rows.Close()
	var namespaces []string
	for rows.Next() {
		var namespaceID string
		if err := rows.Scan(&namespaceID); err != nil {
			return "", fmt.Errorf("scan public routing namespace: %w", err)
		}
		namespaces = append(namespaces, namespaceID)
	}
	if err := rows.Err(); err != nil {
		return "", fmt.Errorf("iterate public routing namespaces: %w", err)
	}
	if len(namespaces) != 1 {
		return "", fmt.Errorf("routing without native access requires exactly one active Namespace, found %d", len(namespaces))
	}
	return namespaces[0], nil
}

func nilLike(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}
