package extproc

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
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
	ReplayRecorder       *routerreplay.Recorder
	// ModelSelector is the registry of advanced model selection algorithms
	// Initialized from config.IntelligentRouting.ModelSelection
	ModelSelector *selection.Registry
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
		responseStore, storeErr := createResponseStore(cfg)
		if storeErr != nil {
			logging.Warnf("Failed to create response store: %v, Response API will be disabled", storeErr)
		} else {
			responseAPIFilter = NewResponseAPIFilter(responseStore)
			logging.Infof("Response API enabled with %s backend", cfg.ResponseAPI.StoreBackend)
		}
	}

	// Create replay recorder with store support
	replayRecorder, err := createReplayRecorder(cfg)
	if err != nil {
		logging.Warnf("Failed to create replay recorder with store: %v, using in-memory fallback", err)
		replayRecorder = routerreplay.NewRecorder(routerreplay.DefaultMaxRecords)
	}

	// Initialize model selection registry with default configs
	// Actual selection method is determined per-decision via algorithm config (aligned with looper)
	modelSelectionCfg := &selection.ModelSelectionConfig{
		Method: "static", // Default; per-decision algorithm overrides this
	}
	// Copy Elo config from config package to selection package format
	eloCfg := cfg.IntelligentRouting.ModelSelection.Elo
	modelSelectionCfg.Elo = &selection.EloConfig{
		InitialRating:     eloCfg.InitialRating,
		KFactor:           eloCfg.KFactor,
		CategoryWeighted:  eloCfg.CategoryWeighted,
		DecayFactor:       eloCfg.DecayFactor,
		MinComparisons:    eloCfg.MinComparisons,
		CostScalingFactor: eloCfg.CostScalingFactor,
	}

	// Copy RouterDC config
	routerDCCfg := cfg.IntelligentRouting.ModelSelection.RouterDC
	modelSelectionCfg.RouterDC = &selection.RouterDCConfig{
		Temperature:         routerDCCfg.Temperature,
		DimensionSize:       routerDCCfg.DimensionSize,
		MinSimilarity:       routerDCCfg.MinSimilarity,
		UseQueryContrastive: routerDCCfg.UseQueryContrastive,
		UseModelContrastive: routerDCCfg.UseModelContrastive,
	}

	// Copy AutoMix config
	autoMixCfg := cfg.IntelligentRouting.ModelSelection.AutoMix
	modelSelectionCfg.AutoMix = &selection.AutoMixConfig{
		VerificationThreshold:  autoMixCfg.VerificationThreshold,
		MaxEscalations:         autoMixCfg.MaxEscalations,
		CostAwareRouting:       autoMixCfg.CostAwareRouting,
		CostQualityTradeoff:    autoMixCfg.CostQualityTradeoff,
		DiscountFactor:         autoMixCfg.DiscountFactor,
		UseLogprobVerification: autoMixCfg.UseLogprobVerification,
	}

	// Copy Hybrid config
	hybridCfg := cfg.IntelligentRouting.ModelSelection.Hybrid
	modelSelectionCfg.Hybrid = &selection.HybridConfig{
		EloWeight:           hybridCfg.EloWeight,
		RouterDCWeight:      hybridCfg.RouterDCWeight,
		AutoMixWeight:       hybridCfg.AutoMixWeight,
		CostWeight:          hybridCfg.CostWeight,
		QualityGapThreshold: hybridCfg.QualityGapThreshold,
		NormalizeScores:     hybridCfg.NormalizeScores,
	}

	// Create selection factory and initialize all selectors
	selectionFactory := selection.NewFactory(modelSelectionCfg)
	if cfg.BackendModels.ModelConfig != nil {
		selectionFactory = selectionFactory.WithModelConfig(cfg.BackendModels.ModelConfig)
	}
	if len(cfg.Categories) > 0 {
		selectionFactory = selectionFactory.WithCategories(cfg.Categories)
	}
	modelSelectorRegistry := selectionFactory.CreateAll()

	logging.Infof("[Router] Initialized model selection registry (per-decision algorithm config)")

	router := &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: categoryDescriptions,
		Classifier:           classifier,
		PIIChecker:           piiChecker,
		Cache:                semanticCache,
		ToolsDatabase:        toolsDatabase,
		ResponseAPIFilter:    responseAPIFilter,
		ReplayRecorder:       replayRecorder,
		ModelSelector:        modelSelectorRegistry,
	}

	return router, nil
}

// handleRouterReplayAPI serves read-only endpoints for router replay records.
// Supports pagination and filtering via query parameters:
//   - limit: max records to return (default 20, max 100)
//   - after: cursor for forward pagination (record ID)
//   - order: sort order "asc" or "desc" (default "desc")
//   - decision: filter by decision name
//   - category: filter by category
//   - model: filter by selected model
//   - request_id: filter by request ID
//   - start_time: filter records after this time (RFC3339)
//   - end_time: filter records before this time (RFC3339)
//   - from_cache: filter by cache hit status (true/false)
func (r *OpenAIRouter) handleRouterReplayAPI(method string, path string) *ext_proc.ProcessingResponse {
	// If recorder is not initialized, the feature is effectively disabled.
	if r.ReplayRecorder == nil {
		return nil
	}

	// Parse query string
	var queryParams url.Values
	cleanPath := path
	if idx := strings.Index(path, "?"); idx != -1 {
		cleanPath = path[:idx]
		var err error
		queryParams, err = url.ParseQuery(path[idx+1:])
		if err != nil {
			queryParams = url.Values{}
		}
	} else {
		queryParams = url.Values{}
	}

	base := "/v1/router_replay"
	if cleanPath == base || cleanPath == base+"/" {
		if method != "GET" {
			return r.createErrorResponse(405, "method not allowed")
		}

		// Build list options from query parameters
		opts := store.ListOptions{
			Limit:        parseIntQueryParam(queryParams, "limit", store.DefaultListLimit),
			After:        queryParams.Get("after"),
			Before:       queryParams.Get("before"),
			Order:        queryParams.Get("order"),
			DecisionName: queryParams.Get("decision"),
			Category:     queryParams.Get("category"),
			Model:        queryParams.Get("model"),
			RequestID:    queryParams.Get("request_id"),
		}

		// Parse time filters
		if startStr := queryParams.Get("start_time"); startStr != "" {
			if t, err := time.Parse(time.RFC3339, startStr); err == nil {
				opts.StartTime = &t
			}
		}
		if endStr := queryParams.Get("end_time"); endStr != "" {
			if t, err := time.Parse(time.RFC3339, endStr); err == nil {
				opts.EndTime = &t
			}
		}

		// Parse from_cache filter
		if fromCacheStr := queryParams.Get("from_cache"); fromCacheStr != "" {
			fromCache := fromCacheStr == "true"
			opts.FromCache = &fromCache
		}

		// List records with pagination
		result, err := r.ReplayRecorder.ListRecords(opts)
		if err != nil {
			logging.Warnf("Failed to list replay records: %v", err)
			return r.createErrorResponse(500, "failed to list records")
		}

		payload := map[string]interface{}{
			"object":   "router_replay.list",
			"count":    len(result.Records),
			"data":     result.Records,
			"has_more": result.HasMore,
		}
		if result.FirstID != "" {
			payload["first_id"] = result.FirstID
		}
		if result.LastID != "" {
			payload["last_id"] = result.LastID
		}

		return r.createJSONResponse(200, payload)
	}

	if strings.HasPrefix(cleanPath, base+"/") {
		if method != "GET" {
			return r.createErrorResponse(405, "method not allowed")
		}
		replayID := strings.TrimPrefix(cleanPath, base+"/")
		if replayID == "" {
			return r.createErrorResponse(400, "replay id is required")
		}

		if rec, ok := r.ReplayRecorder.GetRecord(replayID); ok {
			return r.createJSONResponse(200, rec)
		}
		return r.createErrorResponse(404, "replay record not found")
	}

	return nil
}

// parseIntQueryParam parses an integer query parameter with a default value.
func parseIntQueryParam(params url.Values, key string, defaultVal int) int {
	if val := params.Get(key); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			return i
		}
	}
	return defaultVal
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

// createReplayRecorder creates a replay recorder based on configuration.
// It examines all decisions to find router_replay plugin configurations
// and uses the first enabled one to configure the store backend.
func createReplayRecorder(cfg *config.RouterConfig) (*routerreplay.Recorder, error) {
	// Find the first enabled router_replay configuration
	var replayCfg *config.RouterReplayPluginConfig
	maxRecords := routerreplay.DefaultMaxRecords

	for _, d := range cfg.Decisions {
		if rc := d.GetRouterReplayConfig(); rc != nil && rc.Enabled {
			if replayCfg == nil {
				replayCfg = rc
			}
			if rc.MaxRecords > maxRecords {
				maxRecords = rc.MaxRecords
			}
		}
	}

	// If no router_replay config found, use legacy in-memory recorder
	if replayCfg == nil {
		logging.Debugf("No router_replay configuration found, using in-memory recorder")
		return routerreplay.NewRecorder(maxRecords), nil
	}

	// Determine store backend
	backendType := store.MemoryStoreType
	if replayCfg.StoreBackend != "" {
		backendType = store.StoreBackendType(replayCfg.StoreBackend)
	}

	// Build store config
	storeConfig := store.StoreConfig{
		BackendType: backendType,
		Enabled:     true,
		TTLSeconds:  replayCfg.TTLSeconds,
		MaxRecords:  maxRecords,
		Memory: store.MemoryStoreConfig{
			MaxRecords: maxRecords,
		},
		Redis: store.RedisStoreConfig{
			Address:    replayCfg.Redis.Address,
			Database:   replayCfg.Redis.DB,
			Password:   replayCfg.Redis.Password,
			KeyPrefix:  replayCfg.Redis.KeyPrefix,
			TTLSeconds: replayCfg.TTLSeconds,
		},
	}

	// Validate configuration
	if err := store.ValidateConfig(storeConfig); err != nil {
		return nil, fmt.Errorf("invalid replay store configuration: %w", err)
	}

	// Build async writer config
	asyncConfig := store.DefaultAsyncWriterConfig()
	if replayCfg.WriteBufferSize > 0 {
		asyncConfig.BufferSize = replayCfg.WriteBufferSize
	}
	if replayCfg.WriteWorkers > 0 {
		asyncConfig.Workers = replayCfg.WriteWorkers
	}

	// Build recorder config
	recorderConfig := routerreplay.RecorderConfig{
		StoreConfig:         storeConfig,
		AsyncConfig:         asyncConfig,
		UseAsyncWrites:      replayCfg.AsyncWrites,
		MaxBodyBytes:        replayCfg.MaxBodyBytes,
		CaptureRequestBody:  replayCfg.CaptureRequestBody,
		CaptureResponseBody: replayCfg.CaptureResponseBody,
	}

	recorder, err := routerreplay.NewRecorderWithStore(recorderConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create replay recorder: %w", err)
	}

	logging.Infof("Replay recorder created with %s backend, async_writes=%v, max_records=%d",
		backendType, replayCfg.AsyncWrites, maxRecords)

	return recorder, nil
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
