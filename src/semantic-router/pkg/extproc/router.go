package extproc

import (
	"encoding/json"
	"fmt"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/pii"
)

// OpenAIRouter is an Envoy ExtProc server that routes OpenAI API requests
type OpenAIRouter struct {
	Config               *config.RouterConfig
	CategoryDescriptions []string
	Classifier           *classification.Classifier
	ClassificationSvc    *services.ClassificationService // NEW: Use service with UnifiedClassifier
	PIIChecker           *pii.PolicyChecker
	Cache                cache.CacheBackend
	ToolsDatabase        *tools.ToolsDatabase
}

// Ensure OpenAIRouter implements the ext_proc calls
var _ ext_proc.ExternalProcessorServer = (*OpenAIRouter)(nil)

// NewOpenAIRouter creates a new OpenAI API router instance
func NewOpenAIRouter(configPath string) (*OpenAIRouter, error) {
	var cfg *config.RouterConfig
	var err error

	// Check if we should use the global config (Kubernetes mode) or parse from file
	globalCfg := config.Get()
	if globalCfg != nil && globalCfg.ConfigSource == config.ConfigSourceKubernetes {
		// Use the global config that's managed by the Kubernetes controller
		cfg = globalCfg
		logging.Infof("Using Kubernetes-managed configuration")
	} else {
		// Parse fresh config from file for file-based configuration (supports live reload)
		cfg, err = config.Parse(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load config: %w", err)
		}
		// Update global config reference for packages that rely on config.GetConfig()
		config.Replace(cfg)
		logging.Debugf("Parsed configuration from file: %s", configPath)
	}

	// Load category mapping if classifier is enabled
	var categoryMapping *classification.CategoryMapping
	if cfg.CategoryMappingPath != "" {
		categoryMapping, err = classification.LoadCategoryMapping(cfg.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
		logging.Infof("Loaded category mapping with %d categories", categoryMapping.GetCategoryCount())
	}

	// Load PII mapping if PII classifier is enabled
	var piiMapping *classification.PIIMapping
	if cfg.PIIMappingPath != "" {
		piiMapping, err = classification.LoadPIIMapping(cfg.PIIMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
		logging.Infof("Loaded PII mapping with %d PII types", piiMapping.GetPIITypeCount())
	}

	// Load jailbreak mapping if prompt guard is enabled
	var jailbreakMapping *classification.JailbreakMapping
	if cfg.IsPromptGuardEnabled() {
		jailbreakMapping, err = classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
		logging.Infof("Loaded jailbreak mapping with %d jailbreak types", jailbreakMapping.GetJailbreakTypeCount())
	}

	// Initialize the BERT model for similarity search
	if initErr := candle_binding.InitModel(cfg.BertModel.ModelID, cfg.BertModel.UseCPU); initErr != nil {
		return nil, fmt.Errorf("failed to initialize BERT model: %w", initErr)
	}

	categoryDescriptions := cfg.GetCategoryDescriptions()
	logging.Infof("Category descriptions: %v", categoryDescriptions)

	// Create semantic cache with config options
	cacheConfig := cache.CacheConfig{
		BackendType:         cache.CacheBackendType(cfg.SemanticCache.BackendType),
		Enabled:             cfg.SemanticCache.Enabled,
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          cfg.SemanticCache.MaxEntries,
		TTLSeconds:          cfg.SemanticCache.TTLSeconds,
		EvictionPolicy:      cache.EvictionPolicyType(cfg.SemanticCache.EvictionPolicy),
		BackendConfigPath:   cfg.SemanticCache.BackendConfigPath,
		EmbeddingModel:      cfg.SemanticCache.EmbeddingModel,
	}

	// Use default backend type if not specified
	if cacheConfig.BackendType == "" {
		cacheConfig.BackendType = cache.InMemoryCacheType
	}

	semanticCache, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}

	if semanticCache.IsEnabled() {
		logging.Infof("Semantic cache enabled (backend: %s) with threshold: %.4f, TTL: %d seconds",
			cacheConfig.BackendType, cacheConfig.SimilarityThreshold, cacheConfig.TTLSeconds)
		if cacheConfig.BackendType == cache.InMemoryCacheType {
			logging.Infof("In-memory cache max entries: %d", cacheConfig.MaxEntries)
		}
	} else {
		logging.Infof("Semantic cache is disabled")
	}

	// Create tools database with config options
	toolsThreshold := cfg.BertModel.Threshold // Default to BERT threshold
	if cfg.Tools.SimilarityThreshold != nil {
		toolsThreshold = *cfg.Tools.SimilarityThreshold
	}
	toolsOptions := tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsThreshold,
		Enabled:             cfg.Tools.Enabled,
	}
	toolsDatabase := tools.NewToolsDatabase(toolsOptions)

	// Load tools from file if enabled and path is provided
	if toolsDatabase.IsEnabled() && cfg.Tools.ToolsDBPath != "" {
		if loadErr := toolsDatabase.LoadToolsFromFile(cfg.Tools.ToolsDBPath); loadErr != nil {
			logging.Warnf("Failed to load tools from file %s: %v", cfg.Tools.ToolsDBPath, loadErr)
		}
		logging.Infof("Tools database enabled with threshold: %.4f, top-k: %d",
			toolsThreshold, cfg.Tools.TopK)
	} else {
		logging.Infof("Tools database is disabled")
	}

	// Create utility components
	piiChecker := pii.NewPolicyChecker(cfg)

	classifier, err := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier: %w", err)
	}

	// Create global classification service for API access with auto-discovery
	// This will prioritize LoRA models over legacy ModernBERT
	var classificationSvc *services.ClassificationService
	autoSvc, err := services.NewClassificationServiceWithAutoDiscovery(cfg)
	if err != nil {
		logging.Warnf("Auto-discovery failed during router initialization: %v, using legacy classifier", err)
		classificationSvc = services.NewClassificationService(classifier, cfg)
	} else {
		classificationSvc = autoSvc
		if classificationSvc.HasUnifiedClassifier() {
			// Wire the UnifiedClassifier from the service to the legacy Classifier for delegation
			unifiedClassifier := classificationSvc.GetUnifiedClassifier()
			if unifiedClassifier != nil {
				classifier.UnifiedClassifier = unifiedClassifier
				logging.Infof("Router using UnifiedClassifier (LoRA models) for category classification")
			}
		}
	}

	router := &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: categoryDescriptions,
		Classifier:           classifier,
		ClassificationSvc:    classificationSvc, // NEW: Store the service
		PIIChecker:           piiChecker,
		Cache:                semanticCache,
		ToolsDatabase:        toolsDatabase,
	}

	return router, nil
}

// createJSONResponseWithBody creates a direct response with pre-marshaled JSON body
func (r *OpenAIRouter) createJSONResponseWithBody(statusCode int, jsonBody []byte) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: statusCodeToEnum(statusCode),
				},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: []*core.HeaderValueOption{
						{
							Header: &core.HeaderValue{
								Key:      "content-type",
								RawValue: []byte("application/json"),
							},
						},
					},
				},
				Body: jsonBody,
			},
		},
	}
}

// createJSONResponse creates a direct response with JSON content
func (r *OpenAIRouter) createJSONResponse(statusCode int, data interface{}) *ext_proc.ProcessingResponse {
	jsonData, err := json.Marshal(data)
	if err != nil {
		logging.Errorf("Failed to marshal JSON response: %v", err)
		return r.createErrorResponse(500, "Internal server error")
	}

	return r.createJSONResponseWithBody(statusCode, jsonData)
}

// createErrorResponse creates a direct error response
func (r *OpenAIRouter) createErrorResponse(statusCode int, message string) *ext_proc.ProcessingResponse {
	errorResp := map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "invalid_request_error",
			"code":    statusCode,
		},
	}

	jsonData, err := json.Marshal(errorResp)
	if err != nil {
		logging.Errorf("Failed to marshal error response: %v", err)
		jsonData = []byte(`{"error":{"message":"Internal server error","type":"internal_error","code":500}}`)
		// Use 500 status code for fallback error
		statusCode = 500
	}

	return r.createJSONResponseWithBody(statusCode, jsonData)
}

// shouldClearRouteCache checks if route cache should be cleared
func (r *OpenAIRouter) shouldClearRouteCache() bool {
	// Check if feature is enabled
	return r.Config.ClearRouteCache
}
