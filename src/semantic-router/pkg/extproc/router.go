package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

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
	ModelSelector   *selection.Registry
	ReplayRecorders map[string]*routerreplay.Recorder
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
		Redis:               cfg.SemanticCache.Redis,
		Milvus:              cfg.SemanticCache.Milvus,
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
		responseStore, err := createResponseStore(cfg)
		if err != nil {
			logging.Warnf("Failed to create response store: %v, Response API will be disabled", err)
		} else {
			responseAPIFilter = NewResponseAPIFilter(responseStore)
			logging.Infof("Response API enabled with %s backend", cfg.ResponseAPI.StoreBackend)
		}
	}

	// Initialize router replay recorders (one per decision with replay enabled)
	replayRecorders := initializeReplayRecorders(cfg)

	// Keep first recorder for backward compatibility
	var replayRecorder *routerreplay.Recorder
	for _, recorder := range replayRecorders {
		replayRecorder = recorder
		break
	}

	// Initialize model selection registry with default configs
	// Actual selection method is determined per-decision via algorithm config (aligned with looper)
	modelSelectionCfg := &selection.ModelSelectionConfig{
		Method: "static", // Default; per-decision algorithm overrides this
	}

	// Scan decisions for per-decision algorithm configs (aligned with PR #1089)
	// Per-decision config takes precedence over global config
	var eloFromDecision *config.EloSelectionConfig
	var routerDCFromDecision *config.RouterDCSelectionConfig
	for _, decision := range cfg.IntelligentRouting.Decisions {
		if decision.Algorithm != nil {
			if decision.Algorithm.Type == "elo" && decision.Algorithm.Elo != nil && eloFromDecision == nil {
				eloFromDecision = decision.Algorithm.Elo
			}
			if decision.Algorithm.Type == "router_dc" && decision.Algorithm.RouterDC != nil && routerDCFromDecision == nil {
				routerDCFromDecision = decision.Algorithm.RouterDC
			}
		}
	}

	// Build Elo config: per-decision takes precedence, then global, then defaults
	eloCfg := cfg.IntelligentRouting.ModelSelection.Elo
	modelSelectionCfg.Elo = &selection.EloConfig{
		InitialRating:     eloCfg.InitialRating,
		KFactor:           eloCfg.KFactor,
		CategoryWeighted:  eloCfg.CategoryWeighted,
		DecayFactor:       eloCfg.DecayFactor,
		MinComparisons:    eloCfg.MinComparisons,
		CostScalingFactor: eloCfg.CostScalingFactor,
		StoragePath:       eloCfg.StoragePath,
		AutoSaveInterval:  eloCfg.AutoSaveInterval,
	}
	// Override with per-decision config if present
	if eloFromDecision != nil {
		if eloFromDecision.StoragePath != "" {
			modelSelectionCfg.Elo.StoragePath = eloFromDecision.StoragePath
		}
		if eloFromDecision.AutoSaveInterval != "" {
			modelSelectionCfg.Elo.AutoSaveInterval = eloFromDecision.AutoSaveInterval
		}
		if eloFromDecision.KFactor != 0 {
			modelSelectionCfg.Elo.KFactor = eloFromDecision.KFactor
		}
		if eloFromDecision.InitialRating != 0 {
			modelSelectionCfg.Elo.InitialRating = eloFromDecision.InitialRating
		}
		modelSelectionCfg.Elo.CategoryWeighted = eloFromDecision.CategoryWeighted
	}

	// Build RouterDC config: per-decision takes precedence
	routerDCCfg := cfg.IntelligentRouting.ModelSelection.RouterDC
	modelSelectionCfg.RouterDC = &selection.RouterDCConfig{
		Temperature:         routerDCCfg.Temperature,
		DimensionSize:       routerDCCfg.DimensionSize,
		MinSimilarity:       routerDCCfg.MinSimilarity,
		UseQueryContrastive: routerDCCfg.UseQueryContrastive,
		UseModelContrastive: routerDCCfg.UseModelContrastive,
		RequireDescriptions: routerDCCfg.RequireDescriptions,
		UseCapabilities:     routerDCCfg.UseCapabilities,
	}
	// Override with per-decision config if present
	if routerDCFromDecision != nil {
		if routerDCFromDecision.Temperature != 0 {
			modelSelectionCfg.RouterDC.Temperature = routerDCFromDecision.Temperature
		}
		modelSelectionCfg.RouterDC.RequireDescriptions = routerDCFromDecision.RequireDescriptions
		modelSelectionCfg.RouterDC.UseCapabilities = routerDCFromDecision.UseCapabilities
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
	// Wire embedding function for RouterDC to convert model descriptions to vectors
	selectionFactory = selectionFactory.WithEmbeddingFunc(func(text string) ([]float32, error) {
		return candle_binding.GetEmbedding(text, 0)
	})
	modelSelectorRegistry := selectionFactory.CreateAll()

	// Set as global registry so feedback API can access it
	selection.GlobalRegistry = modelSelectorRegistry

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
		ReplayRecorders:      replayRecorders,
	}

	return router, nil
}

// initializeReplayRecorders creates replay recorders for all decisions with router_replay enabled.
func initializeReplayRecorders(cfg *config.RouterConfig) map[string]*routerreplay.Recorder {
	recorders := make(map[string]*routerreplay.Recorder)

	// Check if router replay
	if !cfg.RouterReplay.Enabled {
		return recorders
	}

	// Create a recorder for each decision
	// Each decision gets its own storage isolation via collection/table/keyspace naming
	for _, d := range cfg.Decisions {
		recorder, err := createReplayRecorder(d.Name, &cfg.RouterReplay)
		if err != nil {
			logging.Errorf("Failed to initialize replay recorder for decision %s: %v", d.Name, err)
			continue
		}

		recorders[d.Name] = recorder
	}

	return recorders
}

// createReplayRecorder creates a single replay recorder with the appropriate storage backend.
func createReplayRecorder(decisionName string, replayCfg *config.RouterReplayConfig) (*routerreplay.Recorder, error) {
	backend := replayCfg.StoreBackend
	if backend == "" {
		backend = "memory"
	}

	maxBodyBytes := replayCfg.MaxBodyBytes
	if maxBodyBytes <= 0 {
		maxBodyBytes = routerreplay.DefaultMaxBodyBytes
	}

	var storage store.Storage
	var err error

	switch backend {
	case "memory":
		maxRecords := replayCfg.MaxRecords
		if maxRecords <= 0 {
			maxRecords = routerreplay.DefaultMaxRecords
		}
		storage = store.NewMemoryStore(maxRecords, replayCfg.TTLSeconds)
		logging.Infof("Router replay for %s using memory backend (max_records=%d)", decisionName, maxRecords)

	case "redis":
		if replayCfg.Redis == nil {
			return nil, fmt.Errorf("redis config required when store_backend is 'redis'")
		}
		// Use decision name as key prefix for Redis isolation
		keyPrefix := decisionName + ":"
		if replayCfg.Redis.KeyPrefix != "" {
			keyPrefix = replayCfg.Redis.KeyPrefix + ":" + decisionName + ":"
		}
		redisConfig := &store.RedisConfig{
			Address:       replayCfg.Redis.Address,
			DB:            replayCfg.Redis.DB,
			Password:      replayCfg.Redis.Password,
			UseTLS:        replayCfg.Redis.UseTLS,
			TLSSkipVerify: replayCfg.Redis.TLSSkipVerify,
			MaxRetries:    replayCfg.Redis.MaxRetries,
			PoolSize:      replayCfg.Redis.PoolSize,
			KeyPrefix:     keyPrefix,
		}
		storage, err = store.NewRedisStore(redisConfig, replayCfg.TTLSeconds, replayCfg.AsyncWrites)
		if err != nil {
			return nil, fmt.Errorf("failed to create redis store: %w", err)
		}
		logging.Infof("Router replay for %s using redis backend (address=%s, key_prefix=%s, ttl=%ds, async=%v)",
			decisionName, redisConfig.Address, keyPrefix, replayCfg.TTLSeconds, replayCfg.AsyncWrites)

	case "postgres":
		if replayCfg.Postgres == nil {
			return nil, fmt.Errorf("postgres config required when store_backend is 'postgres'")
		}
		// Use decision name as table name for PostgreSQL isolation
		tableName := decisionName + "_replay_records"
		if replayCfg.Postgres.TableName != "" {
			tableName = replayCfg.Postgres.TableName + "_" + decisionName
		}
		pgConfig := &store.PostgresConfig{
			Host:            replayCfg.Postgres.Host,
			Port:            replayCfg.Postgres.Port,
			Database:        replayCfg.Postgres.Database,
			User:            replayCfg.Postgres.User,
			Password:        replayCfg.Postgres.Password,
			SSLMode:         replayCfg.Postgres.SSLMode,
			MaxOpenConns:    replayCfg.Postgres.MaxOpenConns,
			MaxIdleConns:    replayCfg.Postgres.MaxIdleConns,
			ConnMaxLifetime: replayCfg.Postgres.ConnMaxLifetime,
			TableName:       tableName,
		}
		storage, err = store.NewPostgresStore(pgConfig, replayCfg.TTLSeconds, replayCfg.AsyncWrites)
		if err != nil {
			return nil, fmt.Errorf("failed to create postgres store: %w", err)
		}
		logging.Infof("Router replay for %s using postgres backend (host=%s, db=%s, table=%s, ttl=%ds, async=%v)",
			decisionName, pgConfig.Host, pgConfig.Database, pgConfig.TableName, replayCfg.TTLSeconds, replayCfg.AsyncWrites)

	case "milvus":
		if replayCfg.Milvus == nil {
			return nil, fmt.Errorf("milvus config required when store_backend is 'milvus'")
		}
		// Use decision name as collection name for Milvus isolation
		collectionName := decisionName + "_replay_records"
		if replayCfg.Milvus.CollectionName != "" {
			collectionName = replayCfg.Milvus.CollectionName + "_" + decisionName
		}
		milvusConfig := &store.MilvusConfig{
			Address:          replayCfg.Milvus.Address,
			Username:         replayCfg.Milvus.Username,
			Password:         replayCfg.Milvus.Password,
			CollectionName:   collectionName,
			ConsistencyLevel: replayCfg.Milvus.ConsistencyLevel,
			ShardNum:         replayCfg.Milvus.ShardNum,
		}
		storage, err = store.NewMilvusStore(milvusConfig, replayCfg.TTLSeconds, replayCfg.AsyncWrites)
		if err != nil {
			return nil, fmt.Errorf("failed to create milvus store: %w", err)
		}
		logging.Infof("Router replay for %s using milvus backend (address=%s, collection=%s, ttl=%ds, async=%v)",
			decisionName, milvusConfig.Address, milvusConfig.CollectionName, replayCfg.TTLSeconds, replayCfg.AsyncWrites)

	default:
		return nil, fmt.Errorf("unknown store_backend: %s (supported: memory, redis, postgres, milvus)", backend)
	}

	recorder := routerreplay.NewRecorder(storage)
	recorder.SetCapturePolicy(replayCfg.CaptureRequestBody, replayCfg.CaptureResponseBody, maxBodyBytes)
	return recorder, nil
}

// handleRouterReplayAPI serves read-only endpoints for router replay records.
func (r *OpenAIRouter) handleRouterReplayAPI(method string, path string) *ext_proc.ProcessingResponse {
	// Check if any recorders are initialized
	hasRecorders := len(r.ReplayRecorders) > 0 || r.ReplayRecorder != nil
	if !hasRecorders {
		return nil
	}

	// Strip query string
	if idx := strings.Index(path, "?"); idx != -1 {
		path = path[:idx]
	}

	base := "/v1/router_replay"
	if path == base || path == base+"/" {
		if method != "GET" {
			return r.createErrorResponse(405, "method not allowed")
		}

		// Aggregate records from all recorders
		var allRecords []routerreplay.RoutingRecord
		for _, recorder := range r.ReplayRecorders {
			records := recorder.ListAllRecords()
			allRecords = append(allRecords, records...)
		}

		// Fallback to legacy single recorder if no multi-recorders
		if len(allRecords) == 0 && r.ReplayRecorder != nil {
			allRecords = r.ReplayRecorder.ListAllRecords()
		}

		payload := map[string]interface{}{
			"object": "router_replay.list",
			"count":  len(allRecords),
			"data":   allRecords,
		}
		return r.createJSONResponse(200, payload)
	}

	if strings.HasPrefix(path, base+"/") {
		if method != "GET" {
			return r.createErrorResponse(405, "method not allowed")
		}
		replayID := strings.TrimPrefix(path, base+"/")
		if replayID == "" {
			return r.createErrorResponse(400, "replay id is required")
		}

		// Search in all recorders
		for _, recorder := range r.ReplayRecorders {
			if rec, ok := recorder.GetRecord(replayID); ok {
				return r.createJSONResponse(200, rec)
			}
		}

		// Fallback to legacy single recorder
		if r.ReplayRecorder != nil {
			if rec, ok := r.ReplayRecorder.GetRecord(replayID); ok {
				return r.createJSONResponse(200, rec)
			}
		}

		return r.createErrorResponse(404, "replay record not found")
	}

	return nil
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
		Redis: responsestore.RedisStoreConfig{
			Address:          cfg.ResponseAPI.Redis.Address,
			Password:         cfg.ResponseAPI.Redis.Password,
			DB:               cfg.ResponseAPI.Redis.DB,
			KeyPrefix:        cfg.ResponseAPI.Redis.KeyPrefix,
			ClusterMode:      cfg.ResponseAPI.Redis.ClusterMode,
			ClusterAddresses: cfg.ResponseAPI.Redis.ClusterAddresses,
			PoolSize:         cfg.ResponseAPI.Redis.PoolSize,
			MinIdleConns:     cfg.ResponseAPI.Redis.MinIdleConns,
			MaxRetries:       cfg.ResponseAPI.Redis.MaxRetries,
			DialTimeout:      cfg.ResponseAPI.Redis.DialTimeout,
			ReadTimeout:      cfg.ResponseAPI.Redis.ReadTimeout,
			WriteTimeout:     cfg.ResponseAPI.Redis.WriteTimeout,
			TLSEnabled:       cfg.ResponseAPI.Redis.TLSEnabled,
			TLSCertPath:      cfg.ResponseAPI.Redis.TLSCertPath,
			TLSKeyPath:       cfg.ResponseAPI.Redis.TLSKeyPath,
			TLSCAPath:        cfg.ResponseAPI.Redis.TLSCAPath,
			ConfigPath:       cfg.ResponseAPI.Redis.ConfigPath,
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
