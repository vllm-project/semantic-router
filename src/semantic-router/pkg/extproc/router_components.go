package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func loadClassifierMappings(cfg *config.RouterConfig) (*classifierMappings, error) {
	mappings := &classifierMappings{}
	var err error

	if cfg.CategoryMappingPath != "" {
		mappings.categoryMapping, err = classification.LoadCategoryMapping(cfg.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
		logging.Infof(
			"Loaded category mapping with %d categories",
			mappings.categoryMapping.GetCategoryCount(),
		)
	}

	if cfg.PIIMappingPath != "" {
		mappings.piiMapping, err = classification.LoadPIIMapping(cfg.PIIMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
		logging.Infof("Loaded PII mapping with %d PII types", mappings.piiMapping.GetPIITypeCount())
	}

	if cfg.IsPromptGuardEnabled() {
		mappings.jailbreakMapping, err = classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
		logging.Infof(
			"Loaded jailbreak mapping with %d jailbreak types",
			mappings.jailbreakMapping.GetJailbreakTypeCount(),
		)
	}

	return mappings, nil
}

func createSemanticCache(cfg *config.RouterConfig) (cache.CacheBackend, error) {
	semanticCacheCfg := cfg.SemanticCache
	cacheConfig := cache.CacheConfig{
		BackendType:         cache.CacheBackendType(semanticCacheCfg.BackendType),
		Enabled:             semanticCacheCfg.Enabled,
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          semanticCacheCfg.MaxEntries,
		TTLSeconds:          semanticCacheCfg.TTLSeconds,
		EvictionPolicy:      cache.EvictionPolicyType(semanticCacheCfg.EvictionPolicy),
		Redis:               semanticCacheCfg.Redis,
		Milvus:              semanticCacheCfg.Milvus,
		EmbeddingModel:      detectSemanticCacheEmbeddingModel(cfg),
	}

	if cacheConfig.BackendType == "" {
		cacheConfig.BackendType = cache.InMemoryCacheType
	}

	semanticCache, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}

	if semanticCache.IsEnabled() {
		logging.Infof(
			"Semantic cache enabled with backend: %s with threshold: %.4f, TTL: %d s",
			cacheConfig.BackendType,
			cacheConfig.SimilarityThreshold,
			cacheConfig.TTLSeconds,
		)
		if cacheConfig.BackendType == cache.InMemoryCacheType {
			logging.Infof("In-memory cache max entries: %d", cacheConfig.MaxEntries)
		}
	} else {
		logging.Infof("Semantic cache is disabled")
	}

	return semanticCache, nil
}

func detectSemanticCacheEmbeddingModel(cfg *config.RouterConfig) string {
	semanticCacheCfg := cfg.SemanticCache
	embeddingModels := cfg.EmbeddingModels
	embeddingModel := semanticCacheCfg.EmbeddingModel
	if embeddingModel != "" {
		return embeddingModel
	}

	switch {
	case embeddingModels.MmBertModelPath != "":
		logging.Infof("Auto-selected mmbert for semantic cache (from embedding_models.mmbert_model_path)")
		return "mmbert"
	case embeddingModels.MultiModalModelPath != "":
		logging.Infof("Auto-selected multimodal for semantic cache (from embedding_models.multimodal_model_path)")
		return "multimodal"
	case embeddingModels.Qwen3ModelPath != "":
		logging.Infof("Auto-selected qwen3 for semantic cache (from embedding_models.qwen3_model_path)")
		return "qwen3"
	case embeddingModels.GemmaModelPath != "":
		logging.Infof("Auto-selected gemma for semantic cache (from embedding_models.gemma_model_path)")
		return "gemma"
	default:
		logging.Warnf("No embedding models configured, falling back to bert for semantic cache")
		return "bert"
	}
}

func createToolsDatabase(cfg *config.RouterConfig) *tools.ToolsDatabase {
	embeddingModels := cfg.EmbeddingModels
	toolsThreshold := embeddingModels.MinSimilarityThreshold()
	if cfg.Tools.SimilarityThreshold != nil {
		toolsThreshold = *cfg.Tools.SimilarityThreshold
	}

	toolsDatabase := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsThreshold,
		Enabled:             cfg.Tools.Enabled,
		ModelType:           embeddingModels.EmbeddingConfig.ModelType,
		TargetDimension:     embeddingModels.EmbeddingConfig.TargetDimension,
	})

	if toolsDatabase.IsEnabled() {
		logging.Infof(
			"Tools database enabled with threshold: %.4f, top-k: %d",
			toolsThreshold,
			cfg.Tools.TopK,
		)
	} else {
		logging.Infof("Tools database is disabled")
	}

	return toolsDatabase
}

func createRouterClassifier(
	cfg *config.RouterConfig,
	mappings *classifierMappings,
) (*classification.Classifier, error) {
	classifier, err := classification.NewClassifier(
		cfg,
		mappings.categoryMapping,
		mappings.piiMapping,
		mappings.jailbreakMapping,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier: %w", err)
	}

	services.NewClassificationService(classifier, cfg)
	logging.Infof("Global classification service initialized with legacy classifier")
	return classifier, nil
}

func createResponseAPIFilter(cfg *config.RouterConfig) *ResponseAPIFilter {
	if !cfg.ResponseAPI.Enabled {
		return nil
	}

	responseStore, err := createResponseStore(cfg)
	if err != nil {
		logging.Warnf("Failed to create response store: %v, Response API will be disabled", err)
		return nil
	}

	logging.Infof("Response API enabled with %s backend", cfg.ResponseAPI.StoreBackend)
	return NewResponseAPIFilter(responseStore)
}
