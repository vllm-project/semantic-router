package managedruntime

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"time"

	_ "github.com/lib/pq"
	"github.com/redis/go-redis/v9"

	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/modelprobe"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	catalogmanaged "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog/managed"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	managementauthpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/postgres"
	managementauthredis "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/redis"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/backendresolver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

const (
	defaultStartupTimeout    = 10 * time.Second
	defaultDiscoveryClaimTTL = 5 * time.Minute
)

// New creates the strict standalone or managed composition. Construction
// opens each durable client once, validates it, and never loads product
// provider metadata in an inference data-plane component.
func New(ctx context.Context, cfg *config.RouterConfig, options Options) (*Runtime, error) {
	if cfg == nil {
		return nil, errors.New("router configuration is required")
	}
	if err := cfg.ValidateControlPlaneBootstrap(); err != nil {
		return nil, fmt.Errorf("validate control-plane bootstrap: %w", err)
	}
	mode := cfg.ControlPlane.Mode
	if mode == "" {
		mode = config.ControlPlaneModeStandalone
	}
	switch mode {
	case config.ControlPlaneModeStandalone:
		if !nilLike(options.ManagementFactory) {
			return nil, errors.New("standalone mode rejects a managed Management factory")
		}
		return newStandalone(cfg, options)
	case config.ControlPlaneModeManaged:
		if nilLike(options.ManagementFactory) {
			return nil, errors.New("managed mode requires a Management factory")
		}
	default:
		return nil, errors.New("control-plane mode is invalid")
	}

	timeout := options.StartupTimeout
	if timeout == 0 {
		timeout = defaultStartupTimeout
	}
	if timeout <= 0 {
		return nil, errors.New("managed startup timeout must be positive")
	}
	startupContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return newManaged(startupContext, cfg, options)
}

func newManaged(ctx context.Context, cfg *config.RouterConfig, options Options) (_ *Runtime, resultErr error) {
	keyrings, newManagedErr := loadDeploymentKeyrings(cfg)
	if newManagedErr != nil {
		return nil, newManagedErr
	}
	runtime := &Runtime{
		mode: config.ControlPlaneModeManaged, keyrings: keyrings,
		accessEnabled: cfg.Access.Enabled,
	}
	defer func() {
		if resultErr != nil {
			_ = runtime.Close()
		}
	}()
	protocolCodecs := protocolcodec.NewBuiltinRegistry()
	runtime.protocolCodecs = protocolCodecs
	credentialAdapters, newManagedErr := backendresolver.BuiltinRegistry()
	if newManagedErr != nil {
		return nil, fmt.Errorf("compose provider credential adapters: %w", newManagedErr)
	}
	discoveryAdapters, newManagedErr := providerdiscovery.BuiltinRegistry()
	if newManagedErr != nil {
		return nil, fmt.Errorf("compose provider discovery adapters: %w", newManagedErr)
	}
	providerRegistry, newManagedErr := composeProviderRegistry(
		options.ProviderIntegrations, options.ProviderBackendCompilers,
		protocolCodecs, credentialAdapters, discoveryAdapters,
	)
	if newManagedErr != nil {
		return nil, newManagedErr
	}
	bootstrapToken, bootstrapTokenPresent, newManagedErr := readBootstrapToken(
		cfg.ManagementAPI.Auth.Bootstrap.TokenFile,
		cfg.ManagementAPI.Auth.Bootstrap.TokenEnv,
	)
	if newManagedErr != nil {
		return nil, fmt.Errorf("read Management bootstrap token: %w", newManagedErr)
	}
	defer zero(bootstrapToken)
	recoveryToken, newManagedErr := readRecoveryToken(
		cfg.ManagementAPI.Auth.Recovery.Enabled,
		cfg.ManagementAPI.Auth.Recovery.TokenFile,
		cfg.ManagementAPI.Auth.Recovery.TokenEnv,
	)
	if newManagedErr != nil {
		return nil, fmt.Errorf("read Management recovery token: %w", newManagedErr)
	}
	defer zero(recoveryToken)

	database, newManagedErr := openPostgreSQL(ctx, cfg.AccessStore.Postgres)
	if newManagedErr != nil {
		return nil, newManagedErr
	}
	runtime.database = database
	redisClient, newManagedErr := openValkey(ctx, cfg.AccessRuntimeStore.Redis)
	if newManagedErr != nil {
		return nil, newManagedErr
	}
	runtime.redis = redisClient
	runtime.redisReady = func(ctx context.Context) error { return redisClient.Ping(ctx).Err() }
	runtime.redisClose = redisClient.Close
	runtime.responseTerminals, newManagedErr = composeManagedResponseTerminalStore(
		redisClient, cfg.AccessRuntimeStore.Redis.KeyPrefix,
	)
	if newManagedErr != nil {
		return nil, fmt.Errorf("compose shared response terminal store: %w", newManagedErr)
	}

	egressPolicy, newManagedErr := backendegress.LoadFile(cfg.BackendEgress.PolicyFile)
	if newManagedErr != nil {
		return nil, fmt.Errorf("load backend egress policy: %w", newManagedErr)
	}
	accessStore, newManagedErr := accesspostgres.New(database)
	if newManagedErr != nil {
		return nil, newManagedErr
	}
	sessionStore, newManagedErr := managementauthpostgres.New(database)
	if newManagedErr != nil {
		return nil, newManagedErr
	}

	providerCodec := providercredential.Codec{Keyring: runtime.keyrings.ProviderKEK}
	providerResolver := backendresolver.Resolver{
		Loader: accessStore, Codec: providerCodec, Registry: credentialAdapters,
	}
	discoveryClaims, newManagedErr := providerdiscovery.NewClaimCodec(providerdiscovery.ClaimKeyset{
		ActiveKeyID: runtime.keyrings.ControlPlane.DiscoveryClaim.ActiveVersion(),
		Keys:        runtime.keyrings.ControlPlane.DiscoveryClaim.Symmetric().Keys,
	})
	if newManagedErr != nil {
		return nil, fmt.Errorf("compose provider discovery claims: %w", newManagedErr)
	}
	replicaOptions, newManagedErr := catalogReplicaOptions(cfg.ControlPlane.ProviderCatalog)
	if newManagedErr != nil {
		return nil, newManagedErr
	}
	runtime.replicaID = replicaOptions.ReplicaID
	runtime.accessKeyPrefix = cfg.AccessRuntimeStore.Redis.KeyPrefix
	discoveryTTL := options.DiscoveryClaimTTL
	if discoveryTTL == 0 {
		discoveryTTL = defaultDiscoveryClaimTTL
	}
	catalogApplication, newManagedErr := catalogmanaged.NewApplication(catalogmanaged.ApplicationOptions{
		DB:                database,
		Registry:          providerRegistry,
		DiscoveryAdapters: discoveryAdapters, CredentialMetadata: accessStore,
		Credentials: providerResolver, EgressPolicy: egressPolicy,
		DialTimeout:       options.BackendDialTimeout,
		CatalogCursorKeys: runtime.keyrings.ControlPlane.CatalogCursor.Symmetric(),
		DiscoveryClaims:   discoveryClaims, DiscoveryClaimTTL: discoveryTTL,
		Replica: replicaOptions,
	})
	if newManagedErr != nil {
		return nil, fmt.Errorf("compose Provider Catalog: %w", newManagedErr)
	}
	runtime.catalog = catalogApplication.Replica
	runtime.catalogClose = catalogApplication.Close
	modelProber, newManagedErr := modelprobe.New(modelprobe.Options{
		Credentials: providerResolver, Codecs: protocolCodecs,
		Transport: catalogApplication.Discovery.Transport,
	})
	if newManagedErr != nil {
		return nil, fmt.Errorf("compose Model probe: %w", newManagedErr)
	}

	var quotaEngine *quotaruntime.RedisEngine
	if cfg.Access.Enabled {
		delegationBarriers, err := managementauthredis.New(managementauthredis.Options{
			Client: redisClient, KeyPrefix: cfg.AccessRuntimeStore.Redis.KeyPrefix, Loader: sessionStore,
		})
		if err != nil {
			return nil, fmt.Errorf("compose delegated inference revocation barriers: %w", err)
		}
		access, engine, err := buildInferenceAccess(
			redisClient, cfg.AccessRuntimeStore.Redis, runtime.keyrings, delegationBarriers,
		)
		if err != nil {
			return nil, err
		}
		runtime.access = access
		quotaEngine = engine
		outcomeRepository, err := outcomefeedback.NewPostgresRepository(database)
		if err != nil {
			return nil, fmt.Errorf("compose outcome feedback repository: %w", err)
		}
		outcomeLimiter, err := outcomefeedback.NewRedisAbuseLimiter(outcomefeedback.RedisAbuseLimiterOptions{
			Client: redisClient, KeyPrefix: cfg.AccessRuntimeStore.Redis.KeyPrefix,
		})
		if err != nil {
			return nil, fmt.Errorf("compose outcome feedback abuse limit: %w", err)
		}
		outcomeProjection, err := outcomefeedback.NewRedisProjectionStore(outcomefeedback.RedisProjectionStoreOptions{
			Client: redisClient, KeyPrefix: cfg.AccessRuntimeStore.Redis.KeyPrefix,
		})
		if err != nil {
			return nil, fmt.Errorf("compose outcome feedback projection store: %w", err)
		}
		runtime.outcomeProjection = outcomeProjection
		runtime.outcomeFeedback, err = outcomefeedback.NewService(outcomefeedback.ServiceOptions{
			Repository: outcomeRepository, Limiter: outcomeLimiter,
		})
		if err != nil {
			return nil, fmt.Errorf("compose outcome feedback service: %w", err)
		}
		runtime.outcomeProjector, err = outcomefeedback.NewProjector(outcomefeedback.ProjectorOptions{
			Repository: outcomeRepository, Publisher: outcomeProjection,
		})
		if err != nil {
			return nil, fmt.Errorf("compose outcome feedback projector: %w", err)
		}
	}
	capabilityLifetime, newManagedErr := cfg.BackendDispatch.CapabilityLifetime()
	if newManagedErr != nil {
		return nil, newManagedErr
	}
	capabilityKeys := runtime.keyrings.ControlPlane.BackendDispatch.Symmetric()
	issuerOptions := backendinvoker.CapabilityIssuerOptions{
		Audience: cfg.BackendDispatch.Audience,
		Keyring: backendinvoker.SigningKeyring{
			ActiveVersion: capabilityKeys.ActiveVersion,
			Keys:          capabilityKeys.Keys,
			MaxLifetime:   capabilityLifetime,
		},
		Lifetime: capabilityLifetime,
	}
	if cfg.Access.Enabled {
		runtime.dispatchCapabilities, newManagedErr = dispatchauthority.NewMeteredRuntime(
			dispatchauthority.MeteredAuthorityOptions{Access: runtime.access, Issuer: issuerOptions},
		)
	} else {
		runtime.dispatchCapabilities, newManagedErr = dispatchauthority.NewRoutingOnlyRuntime(
			dispatchauthority.RoutingOnlyAuthorityOptions{
				NamespaceID:  cfg.ControlPlane.PublicNamespaceID,
				Publications: runtime,
				Issuer:       issuerOptions,
			},
		)
	}
	zeroSymmetric(&capabilityKeys)
	if newManagedErr != nil {
		return nil, fmt.Errorf("compose backend dispatch authority: %w", newManagedErr)
	}

	publicationProcessor, publicationWorker, publicationStore, newManagedErr := composePublicationPipeline(
		database, redisClient, cfg.AccessRuntimeStore.Redis, replicaOptions.ReplicaID,
	)
	if newManagedErr != nil {
		return nil, newManagedErr
	}
	runtime.publisherProcessor = publicationProcessor
	runtime.publisherWorker = publicationWorker
	inferenceProviderResolver, newManagedErr := composeInferenceProviderCredentialResolver(
		publicationStore, providerCodec, credentialAdapters,
	)
	if newManagedErr != nil {
		return nil, newManagedErr
	}
	if cfg.Access.Enabled {
		usageSupervisor, err := composeUsageSupervisor(
			database, redisClient, cfg.AccessRuntimeStore.Redis,
			cfg.Access.UsageStorage, replicaOptions.ReplicaID,
		)
		if err != nil {
			return nil, err
		}
		runtime.usageSupervisor = usageSupervisor
	}

	var dispatchJournal backendinvoker.Journal = backendinvoker.ProcessLocalJournal{}
	if cfg.Access.Enabled {
		authoritativeJournal, err := backendinvoker.NewAuthoritativeAttemptJournal(quotaEngine)
		if err != nil {
			return nil, fmt.Errorf("compose authoritative attempt journal: %w", err)
		}
		dispatchJournal = authoritativeJournal
	}
	dispatch, newManagedErr := newBackendDispatchComposition(
		cfg.BackendDispatch, inferenceProviderResolver, protocolCodecs, dispatchJournal,
		runtime.responseTerminals,
		egressPolicy, options.BackendDialTimeout,
	)
	if newManagedErr != nil {
		return nil, newManagedErr
	}
	runtime.backendDispatch = dispatch

	factoryKeyrings := runtime.keyrings.clone()
	management, newManagedErr := options.ManagementFactory.Build(ctx, ManagementDependencies{
		Database: database, Redis: redisClient, AccessStore: accessStore, SessionStore: sessionStore,
		Catalog: catalogApplication, EgressPolicy: egressPolicy,
		ProtocolCodecs: protocolCodecs, CredentialAdapters: credentialAdapters,
		DiscoveryAdapters: discoveryAdapters, ProviderCredentialCodec: providerCodec,
		ProviderCredentialResolver: providerResolver, BootstrapToken: bootstrapToken,
		BootstrapTokenPresent: bootstrapTokenPresent,
		RecoveryToken:         recoveryToken,
		ModelProber:           modelProber,
		Keyrings:              factoryKeyrings, ReplicaID: replicaOptions.ReplicaID,
		DelegationAudience: InferenceDelegationAudience,
	})
	factoryKeyrings.zero()
	if newManagedErr != nil {
		return nil, fmt.Errorf("compose managed Management API: %w", newManagedErr)
	}
	if nilLike(management) {
		return nil, errors.New("managed Management factory returned no application")
	}
	runtime.management = management
	return runtime, nil
}

func composeManagedResponseTerminalStore(
	client redis.UniversalClient,
	keyPrefix string,
) (backendinvoker.ResponseTerminalStore, error) {
	return backendinvoker.NewRedisResponseTerminalStore(
		backendinvoker.RedisResponseTerminalStoreOptions{Client: client, KeyPrefix: keyPrefix},
	)
}

func composeInferenceProviderCredentialResolver(
	store *accesspublisher.RedisStore,
	codec providercredential.Codec,
	registry backendresolver.StaticRegistry,
) (backendresolver.PublishedResolver, error) {
	if store == nil {
		return backendresolver.PublishedResolver{}, errors.New("published provider credential store is required")
	}
	return backendresolver.PublishedResolver{Loader: store, Codec: codec, Registry: registry}, nil
}

func openPostgreSQL(ctx context.Context, store config.PostgresAccessStoreConfig) (*sql.DB, error) {
	dsn, err := readScalarSecret(store.DSNFile, store.DSNEnv)
	if err != nil {
		return nil, fmt.Errorf("read managed PostgreSQL DSN: %w", err)
	}
	database, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, errors.New("open managed PostgreSQL")
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
		return nil, errors.New("managed PostgreSQL readiness failed")
	}
	return database, nil
}

func openValkey(ctx context.Context, store config.RedisAccessRuntimeStoreConfig) (*redis.Client, error) {
	redisURL, err := readScalarSecret(store.URLFile, store.URLEnv)
	if err != nil {
		return nil, fmt.Errorf("read managed Valkey URL: %w", err)
	}
	options, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, errors.New("parse managed Valkey URL")
	}
	client := redis.NewClient(options)
	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		return nil, errors.New("managed Valkey readiness failed")
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

func catalogReplicaOptions(source config.ProviderCatalogBootstrapConfig) (catalogmanaged.ReplicaOptions, error) {
	replicaID, found := os.LookupEnv(source.ReplicaIDEnv)
	if !found || strings.TrimSpace(replicaID) == "" || replicaID != strings.TrimSpace(replicaID) {
		return catalogmanaged.ReplicaOptions{}, errors.New("provider Catalog replica identity is unavailable")
	}
	lease, err := time.ParseDuration(source.Lease)
	if err != nil {
		return catalogmanaged.ReplicaOptions{}, errors.New("provider Catalog lease is invalid")
	}
	renew, err := time.ParseDuration(source.RenewInterval)
	if err != nil {
		return catalogmanaged.ReplicaOptions{}, errors.New("provider Catalog renewal interval is invalid")
	}
	return catalogmanaged.ReplicaOptions{
		ReplicaID: replicaID, Lease: lease, RenewInterval: renew,
		RolloutGroups:         catalogRolloutGroups(source.RolloutGroups),
		RequiredRolloutGroups: catalogRolloutGroups(source.RequiredRolloutGroups),
	}, nil
}

func catalogRolloutGroups(source []config.ProviderCatalogRolloutGroupConfig) []providercatalog.RolloutGroup {
	groups := make([]providercatalog.RolloutGroup, len(source))
	for index, group := range source {
		groups[index] = providercatalog.RolloutGroup{
			Plane: providercatalog.CapabilityPlane(group.Plane), ID: group.ID,
		}
	}
	return groups
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
