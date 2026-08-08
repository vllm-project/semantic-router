package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
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
	responseAPIFilter    *ResponseAPIFilter
	replayRecorder       *routerreplay.Recorder
	replayStoreShared    bool
	replayRecorders      map[string]*routerreplay.Recorder
	modelSelector        *selection.Registry
	recipeModelSelectors map[config.RecipeName]*selection.Registry
	lookupTable          lookuptable.LookupTable
	memoryStore          memory.Store
	memoryExtractor      *memory.MemoryExtractor
	credentialResolver   *authz.CredentialResolver
	rateLimiter          *ratelimit.RateLimitResolver
	lookupTableCancel    func()
	// generation owns every closeable resource above, registered in
	// construction order. It doubles as the rollback stack while
	// buildRouterComponents runs and as the router's teardown driver
	// afterwards, so there is exactly one list of what a router owns.
	generation *routerruntime.Generation
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
	components, err := buildRouterComponents(cfg)
	if err != nil {
		return nil, err
	}
	return components.buildRouter(), nil
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
	mappings, err := loadClassifierMappings(cfg)
	if err != nil {
		return nil, err
	}

	categoryDescriptions := cfg.GetCategoryDescriptions()
	logging.ComponentDebugEvent("extproc", "category_descriptions_loaded", map[string]interface{}{
		"count":        len(categoryDescriptions),
		"descriptions": categoryDescriptions,
	})

	gen := routerruntime.NewGeneration()

	semanticCache, err := createSemanticCache(cfg)
	if err != nil {
		return nil, err
	}
	gen.Defer(semanticCache.Close)

	toolsDatabase, err := createToolsDatabase(cfg)
	if err != nil {
		return nil, rollbackGeneration(gen, err)
	}
	gen.Defer(toolsDatabase.Close)

	recipeClassifiers, classifier, classificationSvc, err := createRouterClassifier(cfg, mappings)
	if err != nil {
		return nil, rollbackGeneration(gen, err)
	}
	// The whole recipe set, not just the default classifier: every recipe gets
	// its own Classifier, so closing only the default one strands the MCP
	// connections the others opened. RecipeClassifiers.Close covers the default
	// too, which is why it is not registered separately.
	gen.Defer(recipeClassifiers.Close)

	responseAPIFilter := createResponseAPIFilter(cfg)
	gen.Defer(responseAPIFilter.Close)

	replayRecorders, replayRecorder, replayStoreShared := createReplayRuntime(cfg)
	gen.Defer(func() error {
		return closeReplayRecorders(replayRecorder, replayRecorders, replayStoreShared)
	})

	var replayReaderForLookup store.Reader
	if replayRecorder != nil {
		replayReaderForLookup = replayRecorder.Reader()
	}
	recipeModelSelectors, modelSelector, lookupTable, lookupTableCancel := createModelSelectorRegistries(cfg, replayReaderForLookup)
	// Same reasoning as the classifiers: there is one registry per recipe, and
	// each holds its own Elo storage and native ML handles. Closing the map
	// covers the default registry, so it is not registered separately.
	gen.Defer(func() error {
		return closeRecipeModelSelectors(recipeModelSelectors)
	})
	// Registered after the selector so the reverse teardown cancels the
	// lookup table's auto-save/re-population goroutines first, before the
	// selectors those goroutines read through are closed.
	if lookupTableCancel != nil {
		gen.Defer(func() error {
			lookupTableCancel()
			return nil
		})
	}

	memoryStore, memoryExtractor := createMemoryRuntime(cfg)
	if memoryStore != nil {
		gen.Defer(memoryStore.Close)
	}

	credentialResolver := buildCredentialResolver(cfg)
	rateLimiter := buildRateLimitResolver(cfg)
	gen.Defer(rateLimiter.Close)

	if credentialResolver != nil {
		logging.ComponentEvent("extproc", "credential_resolver_initialized", map[string]interface{}{
			"providers": credentialResolver.ProviderNames(),
		})
	}
	if rateLimiter != nil {
		logging.ComponentEvent("extproc", "rate_limit_resolver_initialized", map[string]interface{}{
			"providers": rateLimiter.ProviderNames(),
		})
	}

	return &routerComponents{
		cfg:                  cfg,
		categoryDescriptions: categoryDescriptions,
		classifier:           classifier,
		recipeClassifiers:    recipeClassifiers,
		classificationSvc:    classificationSvc,
		semanticCache:        semanticCache,
		toolsDatabase:        toolsDatabase,
		responseAPIFilter:    responseAPIFilter,
		replayRecorder:       replayRecorder,
		replayStoreShared:    replayStoreShared,
		replayRecorders:      replayRecorders,
		modelSelector:        modelSelector,
		recipeModelSelectors: recipeModelSelectors,
		lookupTable:          lookupTable,
		memoryStore:          memoryStore,
		memoryExtractor:      memoryExtractor,
		credentialResolver:   credentialResolver,
		rateLimiter:          rateLimiter,
		lookupTableCancel:    lookupTableCancel,
		generation:           gen,
	}, nil
}

// rollbackGeneration closes every resource gen has accumulated so far and
// returns cause unchanged, so a failed construction step never leaks the
// resources built by the steps before it.
func rollbackGeneration(gen *routerruntime.Generation, cause error) error {
	if err := gen.Close(); err != nil {
		logging.ComponentWarnEvent("extproc", "router_build_rollback_failed", map[string]interface{}{
			"error": err.Error(),
		})
	}
	return cause
}

func (components *routerComponents) buildRouter() *OpenAIRouter {
	router := &OpenAIRouter{
		Config:                components.cfg,
		CategoryDescriptions:  components.categoryDescriptions,
		Classifier:            components.classifier,
		RecipeClassifiers:     components.recipeClassifiers,
		ClassificationService: components.classificationSvc,
		Cache:                 components.semanticCache,
		ToolsDatabase:         components.toolsDatabase,
		ResponseAPIFilter:     components.responseAPIFilter,
		ReplayRecorder:        components.replayRecorder,
		ReplayStoreShared:     components.replayStoreShared,
		ModelSelector:         components.modelSelector,
		RecipeModelSelectors:  components.recipeModelSelectors,
		LookupTable:           components.lookupTable,
		ReplayRecorders:       components.replayRecorders,
		MemoryStore:           components.memoryStore,
		MemoryExtractor:       components.memoryExtractor,
		CredentialResolver:    components.credentialResolver,
		RateLimiter:           components.rateLimiter,
		lookupTableCancel:     components.lookupTableCancel,
		generation:            components.generation,
	}

	// Registered against the router rather than a value the build produced,
	// because the per-path tool databases are loaded lazily at request time —
	// after every other closer here is in place. Registering last means the
	// reverse teardown closes them first, which is also the right order: they
	// own nothing the other resources need.
	components.generation.Defer(router.closeToolSelectionDatabases)

	return router
}
