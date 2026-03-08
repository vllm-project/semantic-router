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
	replayRecorders      map[string]*routerreplay.Recorder
	modelSelector        *selection.Registry
	memoryStore          *memory.MilvusStore
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

	components, err := buildRouterComponents(cfg)
	if err != nil {
		return nil, err
	}
	return components.buildRouter(), nil
}

func loadRouterConfig(configPath string) (*config.RouterConfig, error) {
	globalCfg := config.Get()
	if globalCfg != nil && globalCfg.ConfigSource == config.ConfigSourceKubernetes {
		logging.Infof("Using Kubernetes-managed configuration")
		return globalCfg, nil
	}

	cfg, err := config.Parse(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	config.Replace(cfg)
	logging.Debugf("[NewOpenAIRouter] Parsed config from file: %s, decisions=%d", configPath, len(cfg.Decisions))
	for i, decision := range cfg.Decisions {
		logging.Debugf(
			"[NewOpenAIRouter]   decision[%d]: name=%q, modelRefs=%d, priority=%d",
			i,
			decision.Name,
			len(decision.ModelRefs),
			decision.Priority,
		)
	}
	return cfg, nil
}

func buildRouterComponents(cfg *config.RouterConfig) (*routerComponents, error) {
	mappings, err := loadClassifierMappings(cfg)
	if err != nil {
		return nil, err
	}

	categoryDescriptions := cfg.GetCategoryDescriptions()
	logging.Debugf("Category descriptions: %v", categoryDescriptions)

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
	replayRecorders, replayRecorder := createReplayRuntime(cfg)
	modelSelector := createModelSelectorRegistry(cfg)
	memoryStore, memoryExtractor := createMemoryRuntime(cfg)
	credentialResolver := buildCredentialResolver(cfg)
	rateLimiter := buildRateLimitResolver(cfg)

	if credentialResolver != nil {
		logging.Infof("Credential resolver initialized with providers: %v", credentialResolver.ProviderNames())
	}
	if rateLimiter != nil {
		logging.Infof("Rate limit resolver initialized with providers: %v", rateLimiter.ProviderNames())
	}

	return &routerComponents{
		cfg:                  cfg,
		categoryDescriptions: categoryDescriptions,
		classifier:           classifier,
		semanticCache:        semanticCache,
		toolsDatabase:        toolsDatabase,
		responseAPIFilter:    responseAPIFilter,
		replayRecorder:       replayRecorder,
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
		ModelSelector:        components.modelSelector,
		ReplayRecorders:      components.replayRecorders,
		MemoryStore:          components.memoryStore,
		MemoryExtractor:      components.memoryExtractor,
		CredentialResolver:   components.credentialResolver,
		RateLimiter:          components.rateLimiter,
	}
}
