package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
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

type classifierMappings struct {
	categoryMapping  *classification.CategoryMapping
	piiMapping       *classification.PIIMapping
	jailbreakMapping *classification.JailbreakMapping
}

type routerComponents struct {
	cfg                  *config.RouterConfig
	categoryDescriptions []string
	classifier           *classification.Classifier
	recipeClassifiers    *classification.RecipeClassifiers
	classificationSvc    *services.ClassificationService
	semanticCache        cache.CacheBackend
	toolsDatabase        *tools.ToolsDatabase
	toolEmbedder         *cachedToolEmbedder
	responseAPIFilter    *ResponseAPIFilter
	replayRecorder       *routerreplay.Recorder
	replayStoreShared    bool
	replayRecorders      map[string]*routerreplay.Recorder
	modelSelector        *selection.Registry
	recipeModelSelectors map[config.RecipeName]*selection.Registry
	lookupTable          lookuptable.LookupTable
	memoryStore          memory.Store
	memoryExtractor      *memory.MemoryExtractor
	protocolCodecs       *protocolcodec.Registry
	credentialResolver   *authz.CredentialResolver
	rateLimiter          *ratelimit.RateLimitResolver
	lookupTableCancel    func()
	routerSessionStore   *sessiontelemetry.RouterSessionStateStoreSlot
	resources            *resourceScope
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
) (*OpenAIRouter, error) {
	cfg, publishGlobal, err := resolveInitialRouterConfig(configPath, runtimeRegistry)
	if err != nil {
		return nil, err
	}

	router, err := buildOpenAIRouterFromConfig(cfg)
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

func buildOpenAIRouterFromConfig(cfg *config.RouterConfig) (*OpenAIRouter, error) {
	if err := validateResponseCacheScopeSecret(cfg); err != nil {
		return nil, err
	}
	components, err := buildRouterComponents(cfg)
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

func buildRouterComponents(cfg *config.RouterConfig) (*routerComponents, error) {
	components := &routerComponents{
		cfg:                cfg,
		resources:          newResourceScope(),
		routerSessionStore: buildRouterLearningStateStore(cfg),
		protocolCodecs:     protocolcodec.NewBuiltinRegistry(),
	}
	registerRouterSessionStore(components.resources, components.routerSessionStore)
	mappings, err := loadClassifierMappings(cfg)
	if err != nil {
		return nil, rollbackResources(components.resources, err)
	}

	components.categoryDescriptions = cfg.GetCategoryDescriptions()
	logging.ComponentDebugEvent("extproc", "category_descriptions_loaded", map[string]interface{}{
		"count":        len(components.categoryDescriptions),
		"descriptions": components.categoryDescriptions,
	})

	if err := components.buildEarlyResources(mappings); err != nil {
		return nil, err
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
	var replayReaderForLookup store.Reader
	if components.replayRecorder != nil {
		replayReaderForLookup = components.replayRecorder.Reader()
	}
	components.recipeModelSelectors, components.modelSelector, components.lookupTable, components.lookupTableCancel = createModelSelectorRegistries(cfg, replayReaderForLookup)
	registerModelSelectorResources(components.resources, components.recipeModelSelectors, components.lookupTableCancel)

	components.memoryStore, components.memoryExtractor = createMemoryRuntime(cfg)
	if components.memoryStore != nil {
		components.resources.add(components.memoryStore.Close)
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

	return components, nil
}

func (components *routerComponents) buildEarlyResources(mappings *classifierMappings) error {
	var err error
	components.semanticCache, err = createSemanticCache(components.cfg)
	if err != nil {
		return rollbackResources(components.resources, err)
	}
	if components.semanticCache != nil {
		components.resources.add(components.semanticCache.Close)
	}

	components.toolsDatabase, components.toolEmbedder, err = buildToolsRuntime(components.cfg)
	if err != nil {
		return rollbackResources(components.resources, err)
	}

	components.recipeClassifiers, components.classifier, components.classificationSvc, err = createRouterClassifier(components.cfg, mappings)
	if err != nil {
		return rollbackResources(components.resources, err)
	}
	components.resources.add(components.recipeClassifiers.Close)
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

func buildToolsRuntime(cfg *config.RouterConfig) (*tools.ToolsDatabase, *cachedToolEmbedder, error) {
	// One provider serves both the tools database and the tool embedder, so a
	// remote endpoint gets a single HTTP client/connection pool.
	provider, providerErr := toolsEmbeddingProvider(cfg)
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
		Config:                  components.cfg,
		CategoryDescriptions:    components.categoryDescriptions,
		Classifier:              components.classifier,
		RecipeClassifiers:       components.recipeClassifiers,
		ClassificationService:   components.classificationSvc,
		Cache:                   components.semanticCache,
		ToolsDatabase:           components.toolsDatabase,
		toolEmbedder:            components.toolEmbedder,
		ResponseAPIFilter:       components.responseAPIFilter,
		ReplayRecorder:          components.replayRecorder,
		ReplayStoreShared:       components.replayStoreShared,
		ModelSelector:           components.modelSelector,
		RecipeModelSelectors:    components.recipeModelSelectors,
		LookupTable:             components.lookupTable,
		ReplayRecorders:         components.replayRecorders,
		MemoryStore:             components.memoryStore,
		MemoryExtractor:         components.memoryExtractor,
		ProtocolCodecs:          components.protocolCodecs,
		CredentialResolver:      components.credentialResolver,
		RateLimiter:             components.rateLimiter,
		lookupTableCancel:       components.lookupTableCancel,
		routerSessionStateStore: components.routerSessionStore,
		resources:               components.resources,
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

	components.resources.add(func() error {
		router.routerLearningMu.Lock()
		learningRuntime := router.routerLearningRuntime
		router.routerLearningMu.Unlock()
		if learningRuntime != nil {
			learningRuntime.RetireAndWait()
		}
		return nil
	})

	return router
}
