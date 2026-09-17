package extproc

import (
	"fmt"
	"net/url"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func loadClassifierMappings(cfg *config.RouterConfig) (*classifierMappings, error) {
	mappings := &classifierMappings{}
	var err error

	if cfg.NeedsCategoryMappingForRouting() {
		mappings.categoryMapping, err = classification.LoadCategoryMapping(cfg.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
		logging.ComponentEvent("extproc", "category_mapping_loaded", map[string]interface{}{
			"count": mappings.categoryMapping.GetCategoryCount(),
		})
	}

	if cfg.NeedsPIIMappingForRouting() {
		mappings.piiMapping, err = classification.LoadPIIMapping(cfg.PIIMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
		logging.ComponentEvent("extproc", "pii_mapping_loaded", map[string]interface{}{
			"count": mappings.piiMapping.GetPIITypeCount(),
		})
	}

	if cfg.NeedsJailbreakMappingForRouting() {
		mappings.jailbreakMapping, err = classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
		logging.ComponentEvent("extproc", "jailbreak_mapping_loaded", map[string]interface{}{
			"count": mappings.jailbreakMapping.GetJailbreakTypeCount(),
		})
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
		Valkey:              semanticCacheCfg.Valkey,
		Milvus:              semanticCacheCfg.Milvus,
		Qdrant:              semanticCacheCfg.Qdrant,
		EmbeddingModel:      detectSemanticCacheEmbeddingModel(cfg),
		PolarityGuard: cache.PolarityGuardOptions{
			UseNLI:                 semanticCacheCfg.PolarityGuard.UsesNLI(),
			ContradictionThreshold: semanticCacheCfg.PolarityGuard.EffectiveContradictionThreshold(),
		},
	}

	if cacheConfig.BackendType == "" {
		cacheConfig.BackendType = cache.InMemoryCacheType
	}

	semanticCache, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}

	if semanticCache.IsEnabled() {
		logging.ComponentEvent("extproc", "semantic_cache_initialized", map[string]interface{}{
			"backend":              cacheConfig.BackendType,
			"similarity_threshold": cacheConfig.SimilarityThreshold,
			"ttl_seconds":          cacheConfig.TTLSeconds,
			"max_entries":          cacheConfig.MaxEntries,
			"polarity_guard_mode":  semanticCacheCfg.PolarityGuard.NormalizedMode(),
		})
	} else {
		logging.ComponentEvent("extproc", "semantic_cache_disabled", map[string]interface{}{
			"backend": cacheConfig.BackendType,
		})
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
		return "mmbert"
	case embeddingModels.MultiModalModelPath != "":
		return "multimodal"
	case embeddingModels.Qwen3ModelPath != "":
		return "qwen3"
	case embeddingModels.GemmaModelPath != "":
		return "gemma"
	default:
		logging.ComponentWarnEvent("extproc", "semantic_cache_embedding_fallback", map[string]interface{}{
			"fallback_model": "bert",
		})
		return "bert"
	}
}

func createToolsDatabase(cfg *config.RouterConfig, provider embedding.Provider) (*tools.ToolsDatabase, error) {
	embeddingModels := cfg.EmbeddingModels
	toolsThreshold := embeddingModels.MinSimilarityThreshold()
	if cfg.Tools.SimilarityThreshold != nil {
		toolsThreshold = *cfg.Tools.SimilarityThreshold
	}
	if !cfg.Tools.Enabled {
		provider = nil
	}

	toolsDatabase := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsThreshold,
		Enabled:             cfg.Tools.Enabled,
		Backend:             embeddingModels.EmbeddingBackend(),
		ModelType:           embeddingModels.EmbeddingConfig.ModelType,
		TargetDimension:     embeddingModels.EmbeddingConfig.TargetDimension,
		Provider:            provider,
		ProviderIdentity:    toolsEmbeddingProviderIdentity(cfg),
	})

	if toolsDatabase.IsEnabled() {
		logging.ComponentEvent("extproc", "tools_database_initialized", map[string]interface{}{
			"similarity_threshold": toolsThreshold,
			"top_k":                cfg.Tools.TopK,
		})
	} else {
		logging.ComponentEvent("extproc", "tools_database_disabled", map[string]interface{}{})
	}

	return toolsDatabase, nil
}

func toolsEmbeddingProviderIdentity(cfg *config.RouterConfig) string {
	if cfg == nil {
		return ""
	}
	models := cfg.EmbeddingModels
	modelType := strings.ToLower(strings.TrimSpace(models.EmbeddingConfig.ModelType))
	backend := strings.ToLower(strings.TrimSpace(models.EmbeddingBackend()))
	parts := []string{
		backend,
		modelType,
		strconv.Itoa(models.EmbeddingConfig.TargetDimension),
		strconv.FormatBool(models.UseCPU),
	}
	if models.UsesRemoteEmbeddingBackend() {
		parts = append(parts,
			normalizeEmbeddingProviderURL(models.Endpoint.BaseURL),
			strings.TrimSpace(models.Endpoint.Model),
			strconv.Itoa(models.Endpoint.Dimensions),
		)
		return strings.Join(parts, "\x00")
	}

	modelPath := ""
	switch modelType {
	case config.EmbeddingModelTypeQwen3:
		modelPath = models.Qwen3ModelPath
	case "gemma":
		modelPath = models.GemmaModelPath
	case "mmbert":
		modelPath = models.MmBertModelPath
	case "multimodal":
		modelPath = models.MultiModalModelPath
	case "bert":
		modelPath = models.BertModelPath
	default:
		modelPath = strings.Join([]string{
			models.Qwen3ModelPath,
			models.GemmaModelPath,
			models.MmBertModelPath,
			models.MultiModalModelPath,
			models.BertModelPath,
		}, "\x00")
	}
	parts = append(parts, config.ResolveModelPath(strings.TrimSpace(modelPath)))
	return strings.Join(parts, "\x00")
}

func normalizeEmbeddingProviderURL(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}

	parsed, err := url.Parse(trimmed)
	if err == nil && parsed.Scheme != "" && parsed.Host != "" {
		return canonicalEmbeddingProviderURL(parsed)
	}

	// A malformed endpoint should not make credentials part of the provider
	// identity. Keep a useful, deterministic fallback while removing the
	// portions that commonly carry secrets (userinfo, query, and fragment).
	return sanitizeMalformedEmbeddingProviderURL(trimmed)
}

func canonicalEmbeddingProviderURL(parsed *url.URL) string {
	if parsed == nil {
		return ""
	}

	canonical := *parsed
	canonical.Scheme = strings.ToLower(canonical.Scheme)
	canonical.Host = strings.ToLower(canonical.Host)
	canonical.User = nil
	canonical.RawQuery = ""
	canonical.ForceQuery = false
	canonical.Fragment = ""
	canonical.RawFragment = ""
	canonical.Path = strings.TrimRight(canonical.Path, "/")
	if canonical.RawPath != "" {
		canonical.RawPath = strings.TrimRight(canonical.RawPath, "/")
		if canonical.RawPath == "" || canonical.RawPath == canonical.Path {
			canonical.RawPath = ""
		}
	}
	return strings.TrimRight(canonical.String(), "/")
}

func sanitizeMalformedEmbeddingProviderURL(raw string) string {
	safe := raw
	if queryStart := strings.IndexAny(safe, "?#"); queryStart >= 0 {
		safe = safe[:queryStart]
	}

	// url.Parse can reject an invalid escape or host while the raw value still
	// contains userinfo. Remove the segment before the last at-sign even when a
	// malformed or schemeless endpoint has no recognizable authority boundary.
	if userInfoEnd := strings.LastIndexByte(safe, '@'); userInfoEnd >= 0 {
		userInfoStart := strings.LastIndexAny(safe[:userInfoEnd], "/\\") + 1
		safe = safe[:userInfoStart] + safe[userInfoEnd+1:]
	}
	return strings.TrimSpace(safe)
}

func toolsEmbeddingProvider(cfg *config.RouterConfig) (embedding.Provider, error) {
	if cfg == nil || !cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return nil, nil
	}
	provider, err := embedding.NewProvider(cfg.EmbeddingModels, embedding.ProviderOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create tools embedding provider: %w", err)
	}
	return provider, nil
}

func createRouterClassifier(
	cfg *config.RouterConfig,
	mappings *classifierMappings,
) (*classification.RecipeClassifiers, *classification.Classifier, *services.ClassificationService, error) {
	classifiers, err := classification.BuildRecipeClassifiers(
		cfg,
		mappings.categoryMapping,
		mappings.piiMapping,
		mappings.jailbreakMapping,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to build recipe classifiers: %w", err)
	}

	if err := classifiers.InitializeRuntime(); err != nil {
		return nil, nil, nil, fmt.Errorf("failed to initialize recipe classifiers: %w", err)
	}

	defaultClassifier := classifiers.Default()
	if defaultClassifier == nil {
		return nil, nil, nil, fmt.Errorf("default routing recipe classifier is unavailable")
	}
	classificationService := services.NewRecipeClassificationService(classifiers, cfg)
	return classifiers, defaultClassifier, classificationService, nil
}

func createResponseAPIFilter(cfg *config.RouterConfig) *ResponseAPIFilter {
	if !cfg.ResponseAPI.Enabled {
		return nil
	}

	responseStore, err := createResponseStore(cfg)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "response_api_store_init_failed", map[string]interface{}{
			"backend":              cfg.ResponseAPI.StoreBackend,
			"error":                err.Error(),
			"response_api_enabled": false,
		})
		return nil
	}

	logging.ComponentEvent("extproc", "response_api_initialized", map[string]interface{}{
		"backend": cfg.ResponseAPI.StoreBackend,
	})
	return NewResponseAPIFilter(responseStore)
}
