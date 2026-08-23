package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
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
	responseAPIFilter    *ResponseAPIFilter
	replayRecorder       *routerreplay.Recorder
	replayStoreShared    bool
	replayRecorders      map[string]*routerreplay.Recorder
	modelSelector        *selection.Registry
	recipeModelSelectors map[config.RecipeName]*selection.Registry
	lookupTable          lookuptable.LookupTable
	memoryStore          memory.Store
	memoryExtractor      *memory.MemoryExtractor
	inferenceAccess      InferenceAccessRuntime
	outcomeFeedback      OutcomeFeedbackRuntime
	outcomeProjection    OutcomeLearningProjectionRuntime
	dispatchCapabilities DispatchCapabilityRuntime
	responseTerminals    backendinvoker.ResponseTerminalReader
	protocolCodecs       *protocolcodec.Registry
	lookupTableCancel    func()
	routerSessionStore   *sessiontelemetry.RouterSessionStateStoreSlot
}

func newOpenAIRouterForServer(
	configPath string,
	runtimeRegistry *routerruntime.Registry,
	dependencies RuntimeDependencies,
	parseConfig ConfigParser,
) (*OpenAIRouter, error) {
	cfg, publishGlobal, err := resolveInitialRouterConfig(configPath, runtimeRegistry, parseConfig)
	if err != nil {
		return nil, err
	}

	router, err := buildOpenAIRouterFromConfigWithDependencies(cfg, dependencies)
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
	parseConfig ConfigParser,
) (*config.RouterConfig, bool, error) {
	if runtimeRegistry != nil {
		if cfg := runtimeRegistry.CurrentConfig(); cfg != nil {
			logging.ComponentEvent("extproc", "router_config_using_runtime_registry", nil)
			return cfg, false, nil
		}
		cfg, err := parseRouterConfigFile(configPath, parseConfig)
		return cfg, false, err
	}

	cfg, err := loadRouterConfig(configPath, parseConfig)
	return cfg, true, err
}

func loadRouterConfig(configPath string, parseConfig ConfigParser) (*config.RouterConfig, error) {
	return parseRouterConfigFile(configPath, parseConfig)
}

func parseRouterConfigFile(configPath string, parseConfig ConfigParser) (*config.RouterConfig, error) {
	if parseConfig == nil {
		return nil, fmt.Errorf("failed to load config: router config parser is unavailable")
	}
	cfg, err := parseConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	return cfg, nil
}

func buildOpenAIRouterFromConfigWithDependencies(
	cfg *config.RouterConfig,
	dependencies RuntimeDependencies,
) (*OpenAIRouter, error) {
	if err := validateResponseCacheScopeSecret(cfg); err != nil {
		return nil, err
	}
	components, err := buildRouterComponentsWithDependencies(cfg, dependencies)
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
		"config_path":      configPath,
		"recipe_count":     len(cfg.Recipes),
		"entrypoint_count": len(cfg.Entrypoints),
	})
	for index := range cfg.Recipes {
		recipe := &cfg.Recipes[index]
		logging.ComponentDebugEvent("extproc", "router_config_recipe_loaded", map[string]interface{}{
			"config_path":    configPath,
			"index":          index,
			"name":           recipe.Name,
			"decision_count": len(recipe.Profile.Decisions),
		})
	}
}

func buildRouterComponentsWithDependencies(
	cfg *config.RouterConfig,
	dependencies RuntimeDependencies,
) (*routerComponents, error) {
	if err := dependencies.validate(cfg); err != nil {
		return nil, err
	}
	routerSessionStore := buildRouterLearningStateStore(cfg)
	keepRouterSessionStore := false
	defer func() {
		if !keepRouterSessionStore && routerSessionStore != nil {
			_ = routerSessionStore.RetireAndClose()
		}
	}()
	mappings, err := loadClassifierMappings(cfg)
	if err != nil {
		return nil, err
	}

	categoryDescriptions := cfg.GetCategoryDescriptions()
	logging.ComponentDebugEvent("extproc", "category_descriptions_loaded", map[string]interface{}{
		"count":        len(categoryDescriptions),
		"descriptions": categoryDescriptions,
	})

	semanticCache, err := createSemanticCache(cfg)
	if err != nil {
		return nil, err
	}

	toolsDatabase, err := createToolsDatabase(cfg)
	if err != nil {
		return nil, err
	}
	recipeClassifiers, classifier, classificationSvc, err := createRouterClassifier(cfg, mappings)
	if err != nil {
		return nil, err
	}

	responseAPIFilter := createResponseAPIFilter(cfg)
	replayRecorders, replayRecorder, replayStoreShared, err := createReplayRuntime(cfg)
	if err != nil {
		return nil, err
	}
	var replayReaderForLookup store.Reader
	if replayRecorder != nil {
		replayReaderForLookup = replayRecorder.Reader()
	}
	recipeModelSelectors, modelSelector, lookupTable, lookupTableCancel := createModelSelectorRegistries(cfg, replayReaderForLookup)
	memoryStore, memoryExtractor := createMemoryRuntime(cfg)
	components := &routerComponents{
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
		inferenceAccess:      dependencies.InferenceAccess,
		outcomeFeedback:      dependencies.OutcomeFeedback,
		outcomeProjection:    dependencies.OutcomeProjection,
		dispatchCapabilities: dependencies.DispatchCapabilities,
		responseTerminals:    dependencies.ResponseTerminals,
		protocolCodecs:       dependencies.ProtocolCodecs,
		lookupTableCancel:    lookupTableCancel,
		routerSessionStore:   routerSessionStore,
	}
	keepRouterSessionStore = true
	return components, nil
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
		ResponseAPIFilter:       components.responseAPIFilter,
		ReplayRecorder:          components.replayRecorder,
		ReplayStoreShared:       components.replayStoreShared,
		ModelSelector:           components.modelSelector,
		RecipeModelSelectors:    components.recipeModelSelectors,
		LookupTable:             components.lookupTable,
		ReplayRecorders:         components.replayRecorders,
		MemoryStore:             components.memoryStore,
		MemoryExtractor:         components.memoryExtractor,
		InferenceAccess:         components.inferenceAccess,
		OutcomeFeedback:         components.outcomeFeedback,
		OutcomeProjection:       components.outcomeProjection,
		DispatchCapabilities:    components.dispatchCapabilities,
		ResponseTerminals:       components.responseTerminals,
		ProtocolCodecs:          components.protocolCodecs,
		lookupTableCancel:       components.lookupTableCancel,
		routerSessionStateStore: components.routerSessionStore,
	}
	if components.classificationSvc != nil {
		components.classificationSvc.SetEvalModelSelector(router)
	}
	return router
}
