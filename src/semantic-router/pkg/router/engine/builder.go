package engine

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/pii"
)

func NewRouterEngine(configPath string) (*RouterEngine, error) {
	var cfg *config.RouterConfig
	var err error

	// Check if we should use the global config (Kubernetes mode) or parse from file
	globalCfg := config.Get()
	if globalCfg != nil && globalCfg.ConfigSource == config.ConfigSourceKubernetes {
		cfg = globalCfg
		logging.Infof("Using Kubernetes-managed configuration")
	} else {
		cfg, err = config.Parse(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load config: %w", err)
		}
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

	categoryDescriptions := cfg.GetCategoryDescriptions()
	logging.Debugf("Category descriptions: %v", categoryDescriptions)

	// Auto-detect embedding model for semantic cache
	embeddingModel := cfg.SemanticCache.EmbeddingModel
	if embeddingModel == "" {
		if cfg.EmbeddingModels.MmBertModelPath != "" {
			embeddingModel = "mmbert"
		} else if cfg.EmbeddingModels.Qwen3ModelPath != "" {
			embeddingModel = "qwen3"
		} else if cfg.EmbeddingModels.GemmaModelPath != "" {
			embeddingModel = "gemma"
		} else {
			embeddingModel = "bert"
		}
		logging.Infof("Auto-selected %s for semantic cache", embeddingModel)
	}

	// Create semantic cache
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
		EmbeddingModel:      embeddingModel,
	}

	if cacheConfig.BackendType == "" {
		cacheConfig.BackendType = cache.InMemoryCacheType
	}

	semanticCache, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}

	if semanticCache.IsEnabled() {
		logging.Infof("Semantic cache enabled with backend: %s", cacheConfig.BackendType)
	}

	// Create tools database
	toolsThreshold := cfg.BertModel.Threshold
	if cfg.Tools.SimilarityThreshold != nil {
		toolsThreshold = *cfg.Tools.SimilarityThreshold
	}
	toolsOptions := tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsThreshold,
		Enabled:             cfg.Tools.Enabled,
		ModelType:           cfg.EmbeddingModels.HNSWConfig.ModelType,
		TargetDimension:     cfg.EmbeddingModels.HNSWConfig.TargetDimension,
	}
	toolsDatabase := tools.NewToolsDatabase(toolsOptions)

	if toolsDatabase.IsEnabled() {
		logging.Infof("Tools database enabled with threshold: %.4f", toolsThreshold)
	}

	// Create PII checker and classifier
	piiChecker := pii.NewPolicyChecker(cfg)
	classifier, err := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier: %w", err)
	}

	// Set global classification service
	services.NewClassificationService(classifier, cfg)
	logging.Infof("Global classification service initialized")

	// Initialize replay recorders
	replayRecorders := initializeReplayRecorders(cfg)

	// Initialize model selection registry
	modelSelector, err := initializeModelSelector(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize model selector: %w", err)
	}

	return &RouterEngine{
		Config:               cfg,
		CategoryDescriptions: categoryDescriptions,
		Classifier:           classifier,
		PIIChecker:           piiChecker,
		Cache:                semanticCache,
		ToolsDatabase:        toolsDatabase,
		ModelSelector:        modelSelector,
		ReplayRecorders:      replayRecorders,
	}, nil
}

func initializeReplayRecorders(cfg *config.RouterConfig) map[string]*routerreplay.Recorder {
	recorders := make(map[string]*routerreplay.Recorder)

	for _, d := range cfg.Decisions {
		pluginCfg := d.GetRouterReplayConfig()
		if pluginCfg == nil || !pluginCfg.Enabled {
			continue
		}

		// Create recorder with plugin config (per-decision) and global config (system-level)
		recorder, err := createReplayRecorder(d.Name, pluginCfg, &cfg.RouterReplay)
		if err != nil {
			logging.Errorf("Failed to initialize replay recorder for decision %s: %v", d.Name, err)
			continue
		}

		recorders[d.Name] = recorder
	}

	return recorders
}

func initializeModelSelector(cfg *config.RouterConfig) (*selection.Registry, error) {
	// Initialize model selection registry with default configs
	// Actual selection method is determined per-decision via algorithm config
	modelSelectionCfg := &selection.ModelSelectionConfig{
		Method: "static",
	}

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
	if routerDCFromDecision != nil {
		if routerDCFromDecision.Temperature != 0 {
			modelSelectionCfg.RouterDC.Temperature = routerDCFromDecision.Temperature
		}
		modelSelectionCfg.RouterDC.RequireDescriptions = routerDCFromDecision.RequireDescriptions
		modelSelectionCfg.RouterDC.UseCapabilities = routerDCFromDecision.UseCapabilities
	}

	autoMixCfg := cfg.IntelligentRouting.ModelSelection.AutoMix
	modelSelectionCfg.AutoMix = &selection.AutoMixConfig{
		VerificationThreshold:  autoMixCfg.VerificationThreshold,
		MaxEscalations:         autoMixCfg.MaxEscalations,
		CostAwareRouting:       autoMixCfg.CostAwareRouting,
		CostQualityTradeoff:    autoMixCfg.CostQualityTradeoff,
		DiscountFactor:         autoMixCfg.DiscountFactor,
		UseLogprobVerification: autoMixCfg.UseLogprobVerification,
	}

	hybridCfg := cfg.IntelligentRouting.ModelSelection.Hybrid
	modelSelectionCfg.Hybrid = &selection.HybridConfig{
		EloWeight:           hybridCfg.EloWeight,
		RouterDCWeight:      hybridCfg.RouterDCWeight,
		AutoMixWeight:       hybridCfg.AutoMixWeight,
		CostWeight:          hybridCfg.CostWeight,
		QualityGapThreshold: hybridCfg.QualityGapThreshold,
		NormalizeScores:     hybridCfg.NormalizeScores,
	}

	mlCfg := cfg.IntelligentRouting.ModelSelection.ML
	if mlCfg.ModelsPath != "" || mlCfg.KNN.PretrainedPath != "" || mlCfg.KMeans.PretrainedPath != "" || mlCfg.SVM.PretrainedPath != "" {
		modelSelectionCfg.ML = &selection.MLSelectorConfig{
			ModelsPath:   mlCfg.ModelsPath,
			EmbeddingDim: mlCfg.EmbeddingDim,
			KNN: &selection.KNNConfig{
				K:              mlCfg.KNN.K,
				PretrainedPath: mlCfg.KNN.PretrainedPath,
			},
			KMeans: &selection.KMeansConfig{
				NumClusters:      mlCfg.KMeans.NumClusters,
				EfficiencyWeight: mlCfg.KMeans.EfficiencyWeight,
				PretrainedPath:   mlCfg.KMeans.PretrainedPath,
			},
			SVM: &selection.SVMConfig{
				Kernel:         mlCfg.SVM.Kernel,
				Gamma:          mlCfg.SVM.Gamma,
				PretrainedPath: mlCfg.SVM.PretrainedPath,
			},
		}
		logging.Infof("[Router] ML model selection enabled with models_path=%s", mlCfg.ModelsPath)
	}

	// Create selection factory and initialize all selectors
	selectionFactory := selection.NewFactory(modelSelectionCfg)
	if cfg.BackendModels.ModelConfig != nil {
		selectionFactory = selectionFactory.WithModelConfig(cfg.BackendModels.ModelConfig)
	}
	if len(cfg.Categories) > 0 {
		selectionFactory = selectionFactory.WithCategories(cfg.Categories)
	}
	selectionFactory = selectionFactory.WithEmbeddingFunc(func(text string) ([]float32, error) {
		output, err := candle_binding.GetEmbeddingBatched(text, "qwen3", 1024)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	})
	modelSelectorRegistry := selectionFactory.CreateAll()

	selection.GlobalRegistry = modelSelectorRegistry

	logging.Infof("[Router] Initialized model selection registry (per-decision algorithm config)")

	return modelSelectorRegistry, nil
}

// LoadToolsDatabase loads tools from the configured path
func (e *RouterEngine) LoadToolsDatabase(toolsDBPath string) error {
	if !e.ToolsDatabase.IsEnabled() {
		return nil
	}

	if toolsDBPath == "" {
		logging.Warnf("Tools database enabled but no tools file path configured")
		return nil
	}

	if err := e.ToolsDatabase.LoadToolsFromFile(toolsDBPath); err != nil {
		return fmt.Errorf("failed to load tools from file %s: %w", toolsDBPath, err)
	}

	logging.Infof("Tools database loaded successfully from: %s", toolsDBPath)
	return nil
}

func createReplayRecorder(decisionName string, pluginCfg *config.RouterReplayPluginConfig, globalCfg *config.RouterReplayConfig) (*routerreplay.Recorder, error) {
	backend := globalCfg.StoreBackend
	if backend == "" {
		backend = "memory"
	}

	maxBodyBytes := pluginCfg.MaxBodyBytes
	if maxBodyBytes <= 0 {
		maxBodyBytes = routerreplay.DefaultMaxBodyBytes
	}

	var storage store.Storage
	var err error

	switch backend {
	case "memory":
		maxRecords := pluginCfg.MaxRecords
		if maxRecords <= 0 {
			maxRecords = routerreplay.DefaultMaxRecords
		}
		storage = store.NewMemoryStore(maxRecords, globalCfg.TTLSeconds)
		logging.Infof("Router replay for %s using memory backend (max_records=%d)", decisionName, maxRecords)

	case "redis":
		if globalCfg.Redis == nil {
			return nil, fmt.Errorf("redis config required when store_backend is 'redis'")
		}
		// Use decision name as key prefix for Redis isolation
		keyPrefix := decisionName + ":"
		if globalCfg.Redis.KeyPrefix != "" {
			keyPrefix = globalCfg.Redis.KeyPrefix + ":" + decisionName + ":"
		}
		redisConfig := &store.RedisConfig{
			Address:       globalCfg.Redis.Address,
			DB:            globalCfg.Redis.DB,
			Password:      globalCfg.Redis.Password,
			UseTLS:        globalCfg.Redis.UseTLS,
			TLSSkipVerify: globalCfg.Redis.TLSSkipVerify,
			MaxRetries:    globalCfg.Redis.MaxRetries,
			PoolSize:      globalCfg.Redis.PoolSize,
			KeyPrefix:     keyPrefix,
		}
		storage, err = store.NewRedisStore(redisConfig, globalCfg.TTLSeconds, globalCfg.AsyncWrites)
		if err != nil {
			return nil, fmt.Errorf("failed to create redis store: %w", err)
		}
		logging.Infof("Router replay for %s using redis backend (address=%s, key_prefix=%s, ttl=%ds, async=%v)",
			decisionName, redisConfig.Address, keyPrefix, globalCfg.TTLSeconds, globalCfg.AsyncWrites)

	case "postgres":
		if globalCfg.Postgres == nil {
			return nil, fmt.Errorf("postgres config required when store_backend is 'postgres'")
		}
		// Use decision name as table name for PostgreSQL isolation
		tableName := decisionName + "_replay_records"
		if globalCfg.Postgres.TableName != "" {
			tableName = globalCfg.Postgres.TableName + "_" + decisionName
		}
		pgConfig := &store.PostgresConfig{
			Host:            globalCfg.Postgres.Host,
			Port:            globalCfg.Postgres.Port,
			Database:        globalCfg.Postgres.Database,
			User:            globalCfg.Postgres.User,
			Password:        globalCfg.Postgres.Password,
			SSLMode:         globalCfg.Postgres.SSLMode,
			MaxOpenConns:    globalCfg.Postgres.MaxOpenConns,
			MaxIdleConns:    globalCfg.Postgres.MaxIdleConns,
			ConnMaxLifetime: globalCfg.Postgres.ConnMaxLifetime,
			TableName:       tableName,
		}
		storage, err = store.NewPostgresStore(pgConfig, globalCfg.TTLSeconds, globalCfg.AsyncWrites)
		if err != nil {
			return nil, fmt.Errorf("failed to create postgres store: %w", err)
		}
		logging.Infof("Router replay for %s using postgres backend (host=%s, db=%s, table=%s, ttl=%ds, async=%v)",
			decisionName, pgConfig.Host, pgConfig.Database, pgConfig.TableName, globalCfg.TTLSeconds, globalCfg.AsyncWrites)

	case "milvus":
		if globalCfg.Milvus == nil {
			return nil, fmt.Errorf("milvus config required when store_backend is 'milvus'")
		}
		// Use decision name as collection name for Milvus isolation
		collectionName := decisionName + "_replay_records"
		if globalCfg.Milvus.CollectionName != "" {
			collectionName = globalCfg.Milvus.CollectionName + "_" + decisionName
		}
		milvusConfig := &store.MilvusConfig{
			Address:          globalCfg.Milvus.Address,
			Username:         globalCfg.Milvus.Username,
			Password:         globalCfg.Milvus.Password,
			CollectionName:   collectionName,
			ConsistencyLevel: globalCfg.Milvus.ConsistencyLevel,
			ShardNum:         globalCfg.Milvus.ShardNum,
		}
		storage, err = store.NewMilvusStore(milvusConfig, globalCfg.TTLSeconds, globalCfg.AsyncWrites)
		if err != nil {
			return nil, fmt.Errorf("failed to create milvus store: %w", err)
		}
		logging.Infof("Router replay for %s using milvus backend (address=%s, collection=%s, ttl=%ds, async=%v)",
			decisionName, milvusConfig.Address, milvusConfig.CollectionName, globalCfg.TTLSeconds, globalCfg.AsyncWrites)

	default:
		return nil, fmt.Errorf("unknown store_backend: %s (supported: memory, redis, postgres, milvus)", backend)
	}

	recorder := routerreplay.NewRecorder(storage)
	recorder.SetCapturePolicy(pluginCfg.CaptureRequestBody, pluginCfg.CaptureResponseBody, maxBodyBytes)
	return recorder, nil
}
