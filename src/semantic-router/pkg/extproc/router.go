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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/pii"
)

// OpenAIRouter is an Envoy ExtProc server that routes OpenAI API requests
type OpenAIRouter struct {
	Config               *config.RouterConfig
	CategoryDescriptions []string
	Classifier           *classification.Classifier
	PIIChecker           *pii.PolicyChecker
	Cache                cache.CacheBackend
	ToolsDatabase        *tools.ToolsDatabase
	ResponseAPIFilter    *ResponseAPIFilter
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

	// Initialize the BERT model for similarity search (only if configured)
	if cfg.BertModel.ModelID != "" {
		if initErr := candle_binding.InitModel(cfg.BertModel.ModelID, cfg.BertModel.UseCPU); initErr != nil {
			return nil, fmt.Errorf("failed to initialize BERT model: %w", initErr)
		}
		logging.Infof("BERT similarity model initialized: %s", cfg.BertModel.ModelID)
	} else {
		logging.Infof("BERT model not configured, skipping initialization")
	}

	categoryDescriptions := cfg.GetCategoryDescriptions()
	logging.Debugf("Category descriptions: %v", categoryDescriptions)

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
		logging.Infof("Semantic cache enabled with backend: %s with threshold: %.4f, TTL: %d s",
			cacheConfig.BackendType, cacheConfig.SimilarityThreshold, cacheConfig.TTLSeconds)
		if cacheConfig.BackendType == cache.InMemoryCacheType {
			logging.Infof("In-memory cache max entries: %d", cacheConfig.MaxEntries)
		}
	} else {
		logging.Infof("Semantic cache is disabled")
	}

	// Create tools database with config options (but don't load tools yet)
	// Tools will be loaded after embedding models are initialized to avoid
	// "ModelFactory not initialized" errors
	toolsThreshold := cfg.BertModel.Threshold // Default to BERT threshold
	if cfg.Tools.SimilarityThreshold != nil {
		toolsThreshold = *cfg.Tools.SimilarityThreshold
	}
	toolsOptions := tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsThreshold,
		Enabled:             cfg.Tools.Enabled,
	}
	toolsDatabase := tools.NewToolsDatabase(toolsOptions)

	// Note: Tools will be loaded later via LoadToolsDatabase() after embedding models init
	if toolsDatabase.IsEnabled() {
		logging.Infof("Tools database enabled with threshold: %.4f, top-k: %d (will load after embedding models init)",
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
	autoSvc, err := services.NewClassificationServiceWithAutoDiscovery(cfg)
	if err != nil {
		logging.Warnf("Auto-discovery failed during router initialization: %v, using legacy classifier", err)
		services.NewClassificationService(classifier, cfg)
	} else {
		logging.Infof("Router initialization: Using auto-discovered unified classifier")
		// The service is already set as global in NewUnifiedClassificationService
		_ = autoSvc
	}

	// Create Response API filter if enabled
	var responseAPIFilter *ResponseAPIFilter
	if cfg.ResponseAPI.Enabled {
		responseStore, err := createResponseStore(cfg)
		if err != nil {
			logging.Warnf("Failed to create response store: %v, Response API will be disabled", err)
		} else {
			responseAPIFilter = NewResponseAPIFilter(responseStore)
			logging.Infof("Response API enabled with %s backend", cfg.ResponseAPI.StoreBackend)
		}
	}

	router := &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: categoryDescriptions,
		Classifier:           classifier,
		PIIChecker:           piiChecker,
		Cache:                semanticCache,
		ToolsDatabase:        toolsDatabase,
		ResponseAPIFilter:    responseAPIFilter,
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

// createResponseStore creates a response store based on configuration.
func createResponseStore(cfg *config.RouterConfig) (responsestore.ResponseStore, error) {
	storeConfig := responsestore.StoreConfig{
		Enabled:     true,
		TTLSeconds:  cfg.ResponseAPI.TTLSeconds,
		BackendType: responsestore.StoreBackendType(cfg.ResponseAPI.StoreBackend),
		Memory: responsestore.MemoryStoreConfig{
			MaxResponses: cfg.ResponseAPI.MaxResponses,
		},
		Milvus: responsestore.MilvusStoreConfig{
			Address:            cfg.ResponseAPI.Milvus.Address,
			Database:           cfg.ResponseAPI.Milvus.Database,
			ResponseCollection: cfg.ResponseAPI.Milvus.Collection,
		},
	}
	return responsestore.NewStore(storeConfig)
}

// LoadToolsDatabase loads tools from file after embedding models are initialized
func (r *OpenAIRouter) LoadToolsDatabase() error {
	if !r.ToolsDatabase.IsEnabled() {
		return nil
	}

	if r.Config.Tools.ToolsDBPath == "" {
		logging.Warnf("Tools database enabled but no tools file path configured")
		return nil
	}

	if err := r.ToolsDatabase.LoadToolsFromFile(r.Config.Tools.ToolsDBPath); err != nil {
		return fmt.Errorf("failed to load tools from file %s: %w", r.Config.Tools.ToolsDBPath, err)
	}

	logging.Infof("Tools database loaded successfully from: %s", r.Config.Tools.ToolsDBPath)
	return nil
}
