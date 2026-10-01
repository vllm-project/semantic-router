package extproc

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

type routerComponents struct {
	embeddings                  *embedding.Set
	serviceEmbeddings           *embedding.Set
	cacheEmbeddings             *embedding.Set
	modelRuntime                *native.Runtime
	rerankers                   map[config.RecipeName]modelruntime.PairScorer
	cfg                         *config.RouterConfig
	categoryDescriptions        []string
	classifier                  *classification.Classifier
	recipeClassifiers           *classification.RecipeClassifiers
	classificationSvc           *services.ClassificationService
	semanticCache               cache.CacheBackend
	responseCache               *cache.ResponseCacheService
	semanticCacheIdentity       string
	toolsDatabase               *tools.ToolsDatabase
	toolEmbedder                *cachedToolEmbedder
	responseAPIFilter           *ResponseAPIFilter
	replayRecorder              *routerreplay.Recorder
	replayStoreShared           bool
	replayRecorders             map[string]*routerreplay.Recorder
	shadowDispatcher            *shadowDispatcher
	modelSelector               *selection.Registry
	recipeModelSelectors        map[config.RecipeName]*selection.Registry
	lookupTable                 lookuptable.LookupTable
	memoryStore                 memory.Store
	memoryExtractor             *memory.MemoryExtractor
	memoryPersistence           *memory.PersistenceRunner
	protocolCodecs              *protocolcodec.Registry
	looperClient                *looper.Client
	credentialResolver          *authz.CredentialResolver
	rateLimiter                 *ratelimit.RateLimitResolver
	lookupTableCancel           func()
	routerSessionStore          *sessiontelemetry.RouterSessionStateStoreSlot
	workflowStateService        *looper.WorkflowStateService
	fallbackOrchestrator        *fallback.Orchestrator
	recipeFallbackOrchestrators map[config.RecipeName]*fallback.Orchestrator
	fallbackCircuitBreaker      *fallback.BackendCircuitBreaker
	resources                   *resourceScope
}

// NewOpenAIRouter creates a new OpenAI API router instance.
func NewOpenAIRouter(configPath string) (*OpenAIRouter, error) {
	cfg, err := loadRouterConfig(configPath)
	if err != nil {
		return nil, err
	}

	router, err := buildOpenAIRouterFromConfig(cfg)
	if err != nil {
		return nil, err
	}

	config.Replace(cfg)
	publishRouterLearningStateStore(router)
	logLoadedRouterConfig(configPath, cfg)
	return router, nil
}

func newOpenAIRouterForServer(
	configPath string,
	runtimeRegistry *routerruntime.Registry,
	pool *binding.Pool,
) (*OpenAIRouter, error) {
	cfg, publishGlobal, err := resolveInitialRouterConfig(configPath, runtimeRegistry)
	if err != nil {
		return nil, err
	}

	router, err := buildOpenAIRouterFromConfig(cfg, pool)
	if err != nil {
		return nil, err
	}

	if publishGlobal {
		config.Replace(cfg)
	}
	logLoadedRouterConfig(configPath, cfg)
	return router, nil
}

func resolveInitialRouterConfig(
	configPath string,
	runtimeRegistry *routerruntime.Registry,
) (*config.RouterConfig, bool, error) {
	if runtimeRegistry != nil {
		if cfg := runtimeRegistry.CurrentConfig(); cfg != nil {
			logging.ComponentEvent("extproc", "router_config_using_runtime_registry", map[string]interface{}{
				"config_source": cfg.ConfigSource,
			})
			return cfg, false, nil
		}
		cfg, err := parseRouterConfigFile(configPath)
		return cfg, false, err
	}

	cfg, err := loadRouterConfig(configPath)
	return cfg, true, err
}

func loadRouterConfig(configPath string) (*config.RouterConfig, error) {
	globalCfg := config.Get()
	if globalCfg != nil && globalCfg.ConfigSource == config.ConfigSourceKubernetes {
		logging.ComponentEvent("extproc", "router_config_using_kubernetes_source", map[string]interface{}{
			"config_source": globalCfg.ConfigSource,
		})
		return globalCfg, nil
	}

	return parseRouterConfigFile(configPath)
}

func parseRouterConfigFile(configPath string) (*config.RouterConfig, error) {
	cfg, err := config.Parse(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	return cfg, nil
}

func buildOpenAIRouterFromConfig(cfg *config.RouterConfig, pools ...*binding.Pool) (*OpenAIRouter, error) {
	if err := validateResponseCacheScopeSecret(cfg); err != nil {
		return nil, err
	}
	components, err := buildRouterComponents(cfg, pools...)
	if err != nil {
		return nil, err
	}
	return components.buildRouter(), nil
}

func validateResponseCacheScopeSecret(cfg *config.RouterConfig) error {
	if cfg == nil || !cfg.ManagementAPI.RemoteExposure || cache.UserScopeSecretConfigured() {
		return nil
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		plugin := decision.GetResponseCacheConfig()
		if plugin == nil || !plugin.Enabled || plugin.Scope == "global" {
			continue
		}
		return fmt.Errorf(
			"USER_SCOPE_NAMESPACE_SECRET is required for remotely exposed response_cache scope %q",
			plugin.Scope,
		)
	}
	return nil
}

func logLoadedRouterConfig(configPath string, cfg *config.RouterConfig) {
	logging.ComponentDebugEvent("extproc", "router_config_loaded", map[string]interface{}{
		"config_path":    configPath,
		"decision_count": len(cfg.Decisions),
	})
	for i, decision := range cfg.Decisions {
		logging.ComponentDebugEvent("extproc", "router_config_decision_loaded", map[string]interface{}{
			"config_path": configPath,
			"index":       i,
			"name":        decision.Name,
			"model_refs":  len(decision.ModelRefs),
			"priority":    decision.Priority,
		})
	}
}

func buildRouterComponents(cfg *config.RouterConfig, pools ...*binding.Pool) (*routerComponents, error) {
	var pool *binding.Pool
	if len(pools) > 0 {
		pool = pools[0]
	}
	components := &routerComponents{
		modelRuntime:       native.New(pool),
		cfg:                cfg,
		resources:          newResourceScope(),
		routerSessionStore: buildRouterLearningStateStore(cfg),
		protocolCodecs:     protocolcodec.NewBuiltinRegistry(),
	}
	registerRouterSessionStore(components.resources, components.routerSessionStore)
	embeddings, err := modelruntime.PrepareOwnedEmbeddings(context.Background(), cfg, components.modelRuntime)
	if err != nil {
		return nil, rollbackResources(components.resources, err)
	}
	components.embeddings = embeddings
	components.resources.add(embeddings.Close)
	servicesConfig := *cfg
	// Ingestion owns an independent handle for its longer worker lifetime.
	servicesConfig.VectorStore = nil
	components.serviceEmbeddings, err = modelruntime.PrepareOwnedGlobalServiceEmbeddings(context.Background(), &servicesConfig, components.modelRuntime)
	if err != nil {
		return nil, rollbackResources(components.resources, err)
	}
	components.resources.add(components.serviceEmbeddings.Close)
	components.cacheEmbeddings = embeddings
	if cfg.NeedsSemanticResponseCache() {
		components.cacheEmbeddings, err = modelruntime.PrepareOwnedResponseCacheEmbeddings(context.Background(), cfg, components.modelRuntime)
		if err != nil {
			return nil, rollbackResources(components.resources, err)
		}
		components.resources.add(components.cacheEmbeddings.Close)
	}
	components.rerankers, err = modelruntime.PrepareRerankers(context.Background(), cfg, components.modelRuntime)
	if err != nil {
		return nil, rollbackResources(components.resources, err)
	}
	components.resources.add(func() error { return modelruntime.CloseRerankers(components.rerankers) })
	if cfg.Looper.IsEnabled() {
		looperClient, clientErr := looper.NewConnectorClient(&cfg.Looper)
		if clientErr != nil {
			return nil, rollbackResources(components.resources, clientErr)
		}
		components.looperClient = looperClient
		components.resources.add(components.looperClient.Close)
	}

	components.categoryDescriptions = cfg.GetCategoryDescriptions()
	logging.ComponentDebugEvent("extproc", "category_descriptions_loaded", map[string]interface{}{
		"count":        len(components.categoryDescriptions),
		"descriptions": components.categoryDescriptions,
	})

	if buildErr := components.buildEarlyResources(); buildErr != nil {
		return nil, buildErr
	}

	components.responseAPIFilter = createResponseAPIFilter(cfg)
	components.resources.add(components.responseAPIFilter.Close)

	components.replayRecorders, components.replayRecorder, components.replayStoreShared, err = createReplayRuntime(cfg)
	if err != nil {
		return nil, rollbackResources(components.resources, err)
	}
	components.resources.add(func() error {
		return closeReplayRecorders(components.replayRecorder, components.replayRecorders, components.replayStoreShared)
	})
	components.shadowDispatcher = newShadowDispatcher()
	components.resources.add(components.shadowDispatcher.Close)
	fallbackPolicy := fallback.DefaultPolicy()
	if cfg.Fallback != nil {
		fallbackPolicy = cfg.Fallback.WithDefaults()
	}
	components.fallbackCircuitBreaker = fallback.NewBackendCircuitBreaker(fallbackPolicy.CircuitBreaker)
	components.fallbackOrchestrator = fallback.NewOrchestrator(fallbackPolicy, components.fallbackCircuitBreaker)

	breakersByPolicy := map[fallback.CircuitBreakerConfig]*fallback.BackendCircuitBreaker{
		fallbackPolicy.CircuitBreaker: components.fallbackCircuitBreaker,
	}

	components.recipeFallbackOrchestrators = make(map[config.RecipeName]*fallback.Orchestrator, len(cfg.Recipes))
	for _, recipe := range cfg.Recipes {
		recipePolicy := fallbackPolicy
		if recipe.Profile.Fallback != nil {
			recipePolicy = recipe.Profile.Fallback.Inherit(fallbackPolicy).WithDefaults()
		}
		breaker, ok := breakersByPolicy[recipePolicy.CircuitBreaker]
		if !ok {
			breaker = fallback.NewBackendCircuitBreaker(recipePolicy.CircuitBreaker)
			breakersByPolicy[recipePolicy.CircuitBreaker] = breaker
		}
		components.recipeFallbackOrchestrators[recipe.Name] = fallback.NewOrchestrator(recipePolicy, breaker)
	}
	var replayReaderForLookup store.Reader
	if components.replayRecorder != nil {
		replayReaderForLookup = components.replayRecorder.Reader()
	}
	if cfg.ModelSelection.Enabled {
		components.recipeModelSelectors, components.modelSelector, components.lookupTable, components.lookupTableCancel = createModelSelectorRegistries(cfg, replayReaderForLookup, components.recipeClassifiers)
		registerModelSelectorResources(components.resources, components.recipeModelSelectors, components.lookupTableCancel)
	} else {
		logging.ComponentEvent("extproc", "model_selection_disabled", map[string]interface{}{})
	}

	components.memoryStore, components.memoryExtractor = createMemoryRuntime(cfg, components.serviceEmbeddings)
	if components.memoryStore != nil {
		components.resources.add(components.memoryStore.Close)
	}
	// Resources close in reverse order, so retire writes before closing the store.
	components.memoryPersistence = createMemoryPersistenceRunner(cfg, components.memoryExtractor)
	if components.memoryPersistence != nil {
		// RetireAndWait treats a non-positive grace as its own default.
		grace := time.Duration(cfg.Memory.Persistence.ShutdownGraceSeconds) * time.Second
		components.resources.addDraining(func() error {
			return components.memoryPersistence.RetireAndWait(grace)
		}, components.memoryPersistence.Done())
	}

	components.credentialResolver = buildCredentialResolver(cfg)
	components.rateLimiter = buildRateLimitResolver(cfg)
	components.resources.add(components.rateLimiter.Close)

	if components.credentialResolver != nil {
		logging.ComponentEvent("extproc", "credential_resolver_initialized", map[string]interface{}{
			"providers": components.credentialResolver.ProviderNames(),
		})
	}
	if components.rateLimiter != nil {
		logging.ComponentEvent("extproc", "rate_limit_resolver_initialized", map[string]interface{}{
			"providers": components.rateLimiter.ProviderNames(),
		})
	}

	components.workflowStateService = newWorkflowStateServiceIfEnabled(cfg)
	if components.workflowStateService != nil {
		components.resources.add(components.workflowStateService.Close)
	}

	return components, nil
}

func (components *routerComponents) buildEarlyResources() error {
	verifier, err := modelruntime.PrepareOwnedResponseCacheNLI(context.Background(), components.cfg, components.modelRuntime)
	if err != nil {
		return rollbackResources(components.resources, err)
	}
	if verifier != nil {
		// The cache drains and closes before its borrowed verifier is released.
		components.resources.add(verifier.Close)
	}
	components.semanticCache, components.semanticCacheIdentity, err = createSemanticCache(components.cfg, components.cacheEmbeddings)
	if err != nil {
		return rollbackResources(components.resources, err)
	}
	if components.semanticCache != nil {
		components.resources.add(components.semanticCache.Close)
	}

	components.toolsDatabase, components.toolEmbedder, err = buildToolsRuntime(components.cfg, components.serviceEmbeddings)
	if err != nil {
		return rollbackResources(components.resources, err)
	}

	components.recipeClassifiers, components.classifier, components.classificationSvc, err = createRouterClassifier(components.cfg, classification.RecipeRuntimeOptions{Runtime: components.modelRuntime, Embeddings: components.embeddings})
	if err != nil {
		return rollbackResources(components.resources, err)
	}
	components.classificationSvc.SetGlobalEmbeddings(components.serviceEmbeddings)
	components.resources.add(components.recipeClassifiers.Close)
	components.resources.add(components.classificationSvc.Close)
	if target, ok := components.semanticCache.(interface {
		SetPolarityVerifier(cache.PolarityVerifyFunc)
	}); ok {
		if verifier != nil {
			target.SetPolarityVerifier(func(ctx context.Context, cached, incoming string) (float32, error) {
				result, callErr := verifier.Call(ctx, string(config.GlobalModelScope), tasks.TextPairRequest{Premise: cached, Hypothesis: incoming})
				if callErr != nil {
					return 0, callErr
				}
				return result.Probabilities[2], nil
			})
		}
	}
	components.responseCache, err = newResponseCacheService(components.cfg, components.semanticCache, components.semanticCacheIdentity, components.cacheEmbeddings)
	if err != nil {
		return rollbackResources(components.resources, err)
	}

	return nil
}

func registerRouterSessionStore(
	resources *resourceScope,
	store *sessiontelemetry.RouterSessionStateStoreSlot,
) {
	if store == nil {
		return
	}
	resources.add(func() error {
		sessiontelemetry.UnpublishRouterSessionStateStore(store)
		return store.RetireAndClose()
	})
}

func createMemoryPersistenceRunner(cfg *config.RouterConfig, extractor *memory.MemoryExtractor) *memory.PersistenceRunner {
	// Workers and queue storage follow the memory store that was actually built:
	// enablement alone still yields a nil extractor when the backend is
	// unreachable, and every write would then be suppressed as "no_extractor".
	if cfg == nil || extractor == nil || !isMemoryEnabled(cfg) {
		return nil
	}
	persistence := cfg.Memory.Persistence
	return memory.NewPersistenceRunner(
		time.Duration(persistence.TimeoutSeconds)*time.Second,
		persistence.Concurrency,
		persistence.Queue,
	)
}

func registerModelSelectorResources(
	resources *resourceScope,
	registries map[config.RecipeName]*selection.Registry,
	lookupTableCancel func(),
) {
	resources.add(func() error {
		return closeRecipeModelSelectors(registries)
	})
	if lookupTableCancel == nil {
		return
	}
	resources.add(func() error {
		lookupTableCancel()
		return nil
	})
}

func rollbackResources(resources *resourceScope, cause error) error {
	if err := resources.close(); err != nil {
		logging.ComponentWarnEvent("extproc", "router_build_rollback_failed", map[string]interface{}{
			"error": err.Error(),
		})
	}
	return cause
}

func buildToolsRuntime(cfg *config.RouterConfig, sets ...*embedding.Set) (*tools.ToolsDatabase, *cachedToolEmbedder, error) {
	// One provider serves both the tools database and the tool embedder, so a
	// remote endpoint gets a single HTTP client/connection pool.
	provider, providerErr := toolsEmbeddingProvider(cfg, sets...)
	if providerErr != nil && cfg.Tools.Enabled {
		return nil, nil, providerErr
	}
	database, err := createToolsDatabase(cfg, provider)
	if err != nil {
		return nil, nil, err
	}
	if providerErr != nil {
		logging.Warnf("tool_selection: embedding provider unavailable, filter mode will use its fallback: %v", providerErr)
		return database, nil, nil
	}
	return database, newToolEmbedderForConfig(cfg, provider), nil
}

func (components *routerComponents) buildRouter() *OpenAIRouter {
	router := &OpenAIRouter{
		Config:                      components.cfg,
		Embeddings:                  components.embeddings,
		serviceEmbeddings:           components.serviceEmbeddings,
		cacheEmbeddings:             components.cacheEmbeddings,
		rerankers:                   components.rerankers,
		CategoryDescriptions:        components.categoryDescriptions,
		Classifier:                  components.classifier,
		RecipeClassifiers:           components.recipeClassifiers,
		ClassificationService:       components.classificationSvc,
		Cache:                       components.semanticCache,
		ResponseCache:               components.responseCache,
		ToolsDatabase:               components.toolsDatabase,
		toolEmbedder:                components.toolEmbedder,
		ResponseAPIFilter:           components.responseAPIFilter,
		ReplayRecorder:              components.replayRecorder,
		ReplayStoreShared:           components.replayStoreShared,
		ModelSelector:               components.modelSelector,
		RecipeModelSelectors:        components.recipeModelSelectors,
		LookupTable:                 components.lookupTable,
		ReplayRecorders:             components.replayRecorders,
		ShadowDispatcher:            components.shadowDispatcher,
		MemoryStore:                 components.memoryStore,
		MemoryExtractor:             components.memoryExtractor,
		memoryPersistence:           components.memoryPersistence,
		ProtocolCodecs:              components.protocolCodecs,
		looperClient:                components.looperClient,
		CredentialResolver:          components.credentialResolver,
		RateLimiter:                 components.rateLimiter,
		lookupTableCancel:           components.lookupTableCancel,
		routerSessionStateStore:     components.routerSessionStore,
		WorkflowStateService:        components.workflowStateService,
		FallbackOrchestrator:        components.fallbackOrchestrator,
		RecipeFallbackOrchestrators: components.recipeFallbackOrchestrators,
		resources:                   components.resources,
	}
	if components.classificationSvc != nil {
		components.classificationSvc.SetEvalModelSelector(router)
	}

	components.resources.add(func() error {
		if router.CompressionRecovery != nil {
			return router.CompressionRecovery.Close()
		}
		return nil
	})

	return router
}

// newWorkflowStateServiceIfEnabled owns one workflow tool-state store for the
// router generation when any routing profile uses algorithm.type=workflows,
// including recipe-only configs whose decisions are not on the flat list.
func newWorkflowStateServiceIfEnabled(cfg *config.RouterConfig) *looper.WorkflowStateService {
	if cfg == nil || !cfg.HasFlowDecision() {
		return nil
	}
	return looper.NewWorkflowStateService(&cfg.Looper)
}
