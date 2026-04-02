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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
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
	semanticCache        cache.CacheBackend
	toolsDatabase        *tools.ToolsDatabase
	responseAPIFilter    *ResponseAPIFilter
	replayRecorder       *routerreplay.Recorder
	replayStoreShared    bool
	replayRecorders      map[string]*routerreplay.Recorder
	modelSelector        *selection.Registry
	memoryStore          memory.Store
	memoryExtractor      *memory.MemoryExtractor
	credentialResolver   *authz.CredentialResolver
	rateLimiter          *ratelimit.RateLimitResolver
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

func loadRouterConfig(configPath string) (*config.RouterConfig, error) {
	globalCfg := config.Get()
	if globalCfg != nil && globalCfg.ConfigSource == config.ConfigSourceKubernetes {
		logging.ComponentEvent("extproc", "router_config_using_kubernetes_source", map[string]interface{}{
			"config_source": globalCfg.ConfigSource,
		})
		return globalCfg, nil
	}

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

	semanticCache, err := createSemanticCache(cfg)
	if err != nil {
		return nil, err
	}

	toolsDatabase := createToolsDatabase(cfg)
	classifier, err := createRouterClassifier(cfg, mappings)
	if err != nil {
		return nil, err
	}

	responseAPIFilter := createResponseAPIFilter(cfg)
	replayRecorders, replayRecorder, replayStoreShared := createReplayRuntime(cfg)
	modelSelector := createModelSelectorRegistry(cfg)
	memoryStore, memoryExtractor := createMemoryRuntime(cfg)
	credentialResolver := buildCredentialResolver(cfg)
	rateLimiter := buildRateLimitResolver(cfg)

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
		semanticCache:        semanticCache,
		toolsDatabase:        toolsDatabase,
		responseAPIFilter:    responseAPIFilter,
		replayRecorder:       replayRecorder,
		replayStoreShared:    replayStoreShared,
		replayRecorders:      replayRecorders,
		modelSelector:        modelSelector,
		memoryStore:          memoryStore,
		memoryExtractor:      memoryExtractor,
		credentialResolver:   credentialResolver,
		rateLimiter:          rateLimiter,
	}, nil
}

func (components *routerComponents) buildRouter() *OpenAIRouter {
	return &OpenAIRouter{
		Config:               components.cfg,
		CategoryDescriptions: components.categoryDescriptions,
		Classifier:           components.classifier,
		Cache:                components.semanticCache,
		ToolsDatabase:        components.toolsDatabase,
		ResponseAPIFilter:    components.responseAPIFilter,
		ReplayRecorder:       components.replayRecorder,
		ReplayStoreShared:    components.replayStoreShared,
		ModelSelector:        components.modelSelector,
		ReplayRecorders:      components.replayRecorders,
		MemoryStore:          components.memoryStore,
		MemoryExtractor:      components.memoryExtractor,
		CredentialResolver:   components.credentialResolver,
		RateLimiter:          components.rateLimiter,
	}
}
