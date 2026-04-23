package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func createReplayRuntime(cfg *config.RouterConfig) (map[string]*routerreplay.Recorder, *routerreplay.Recorder, bool) {
	backend := resolveReplayStoreBackend(cfg.RouterReplay.StoreBackend)
	if usesSharedReplayStorage(backend) {
		recorders, replayRecorder := initializeSharedReplayRecorders(cfg, backend)
		return recorders, replayRecorder, replayRecorder != nil
	}

	replayRecorders := initializeReplayRecorders(cfg)

	var replayRecorder *routerreplay.Recorder
	for _, recorder := range replayRecorders {
		replayRecorder = recorder
		break
	}

	return replayRecorders, replayRecorder, false
}

// initializeReplayRecorders creates replay recorders for decisions with router_replay plugin configured.
func initializeReplayRecorders(cfg *config.RouterConfig) map[string]*routerreplay.Recorder {
	backend := resolveReplayStoreBackend(cfg.RouterReplay.StoreBackend)
	if usesSharedReplayStorage(backend) {
		recorders, _ := initializeSharedReplayRecorders(cfg, backend)
		return recorders
	}
	return initializeIsolatedReplayRecorders(cfg, backend)
}

func initializeIsolatedReplayRecorders(
	cfg *config.RouterConfig,
	backend string,
) map[string]*routerreplay.Recorder {
	recorders := make(map[string]*routerreplay.Recorder)

	for _, decision := range cfg.Decisions {
		pluginCfg := cfg.EffectiveRouterReplayConfigForDecision(decision.Name)
		if pluginCfg == nil {
			continue
		}

		recorder, err := createReplayRecorder(decision.Name, backend, pluginCfg, &cfg.RouterReplay)
		if err != nil {
			logging.Errorf("Failed to initialize replay recorder for decision %s: %v", decision.Name, err)
			continue
		}

		recorders[decision.Name] = recorder
	}

	return recorders
}

func initializeSharedReplayRecorders(
	cfg *config.RouterConfig,
	backend string,
) (map[string]*routerreplay.Recorder, *routerreplay.Recorder) {
	recorders := make(map[string]*routerreplay.Recorder)
	var (
		sharedStore    store.Storage
		replayRecorder *routerreplay.Recorder
	)

	for _, decision := range cfg.Decisions {
		pluginCfg := cfg.EffectiveRouterReplayConfigForDecision(decision.Name)
		if pluginCfg == nil {
			continue
		}

		if sharedStore == nil {
			storage, err := createSharedReplayStore(backend, &cfg.RouterReplay)
			if err != nil {
				logging.Errorf("Failed to initialize shared replay store for backend %s: %v", backend, err)
				return map[string]*routerreplay.Recorder{}, nil
			}
			sharedStore = storage
		}

		recorder := createSharedReplayRecorder(sharedStore, pluginCfg)
		recorders[decision.Name] = recorder
		if replayRecorder == nil {
			replayRecorder = recorder
		}
	}

	return recorders, replayRecorder
}

// createReplayRecorder creates a single replay recorder with the appropriate storage backend.
func createReplayRecorder(
	decisionName string,
	backend string,
	pluginCfg *config.RouterReplayPluginConfig,
	globalCfg *config.RouterReplayConfig,
) (*routerreplay.Recorder, error) {
	maxBodyBytes := resolveReplayMaxBodyBytes(pluginCfg.MaxBodyBytes)

	storage, err := createReplayStore(decisionName, backend, pluginCfg, globalCfg)
	if err != nil {
		return nil, err
	}

	recorder := routerreplay.NewRecorder(storage)
	recorder.SetCapturePolicy(pluginCfg.CaptureRequestBody, pluginCfg.CaptureResponseBody, maxBodyBytes)
	recorder.SetMaxToolTraceBytes(pluginCfg.MaxToolTraceBytes)
	return recorder, nil
}

func createSharedReplayRecorder(
	storage store.Storage,
	pluginCfg *config.RouterReplayPluginConfig,
) *routerreplay.Recorder {
	recorder := routerreplay.NewRecorder(storage)
	recorder.SetCapturePolicy(
		pluginCfg.CaptureRequestBody,
		pluginCfg.CaptureResponseBody,
		resolveReplayMaxBodyBytes(pluginCfg.MaxBodyBytes),
	)
	recorder.SetMaxToolTraceBytes(pluginCfg.MaxToolTraceBytes)
	return recorder
}

func usesSharedReplayStorage(backend string) bool {
	switch backend {
	case "postgres", "redis", "milvus":
		return true
	default:
		return false
	}
}

func resolveReplayStoreBackend(backend string) string {
	if backend == "" {
		return "memory"
	}
	return backend
}

func resolveReplayMaxBodyBytes(maxBodyBytes int) int {
	if maxBodyBytes <= 0 {
		return routerreplay.DefaultMaxBodyBytes
	}
	return maxBodyBytes
}

func createReplayStore(
	decisionName string,
	backend string,
	pluginCfg *config.RouterReplayPluginConfig,
	globalCfg *config.RouterReplayConfig,
) (store.Storage, error) {
	switch backend {
	case "memory":
		logging.Warnf("Router replay store_backend is set to %q — all replay records "+
			"will be lost on router restart. Use \"postgres\" or \"redis\" for durable storage in production.",
			"memory")
		return createReplayMemoryStore(decisionName, pluginCfg, globalCfg), nil
	case "redis":
		return createReplayRedisStore(globalCfg)
	case "postgres":
		return createReplayPostgresStore(globalCfg)
	case "milvus":
		return createReplayMilvusStore(globalCfg)
	default:
		return nil, fmt.Errorf(
			"unknown store_backend: %s (supported: memory, redis, postgres, milvus)",
			backend,
		)
	}
}

func createSharedReplayStore(
	backend string,
	globalCfg *config.RouterReplayConfig,
) (store.Storage, error) {
	switch backend {
	case "redis":
		return createReplayRedisStore(globalCfg)
	case "postgres":
		return createReplayPostgresStore(globalCfg)
	case "milvus":
		return createReplayMilvusStore(globalCfg)
	default:
		return nil, fmt.Errorf("shared replay storage not supported for backend %s", backend)
	}
}

func createReplayMemoryStore(
	decisionName string,
	pluginCfg *config.RouterReplayPluginConfig,
	globalCfg *config.RouterReplayConfig,
) store.Storage {
	maxRecords := pluginCfg.MaxRecords
	if maxRecords <= 0 {
		maxRecords = routerreplay.DefaultMaxRecords
	}
	logging.Debugf("Router replay for %s using memory backend (max_records=%d)", decisionName, maxRecords)
	return store.NewMemoryStore(maxRecords, globalCfg.TTLSeconds)
}

func createReplayRedisStore(globalCfg *config.RouterReplayConfig) (store.Storage, error) {
	if globalCfg.Redis == nil {
		return nil, fmt.Errorf("redis config required when store_backend is 'redis'")
	}

	redisConfig := buildReplayRedisConfig(globalCfg.Redis)
	storage, err := store.NewRedisStore(redisConfig, globalCfg.TTLSeconds, globalCfg.AsyncWrites)
	if err != nil {
		return nil, fmt.Errorf("failed to create redis store: %w", err)
	}
	logging.Debugf(
		"Router replay using redis backend (address=%s, key_prefix=%s, ttl=%ds, async=%v)",
		redisConfig.Address,
		redisConfig.KeyPrefix,
		globalCfg.TTLSeconds,
		globalCfg.AsyncWrites,
	)
	return storage, nil
}

func buildReplayRedisConfig(redisCfg *config.RouterReplayRedisConfig) *store.RedisConfig {
	return &store.RedisConfig{
		Address:       redisCfg.Address,
		DB:            redisCfg.DB,
		Password:      redisCfg.Password,
		UseTLS:        redisCfg.UseTLS,
		TLSSkipVerify: redisCfg.TLSSkipVerify,
		MaxRetries:    redisCfg.MaxRetries,
		PoolSize:      redisCfg.PoolSize,
		KeyPrefix:     redisCfg.KeyPrefix,
	}
}

func createReplayPostgresStore(globalCfg *config.RouterReplayConfig) (store.Storage, error) {
	if globalCfg.Postgres == nil {
		return nil, fmt.Errorf("postgres config required when store_backend is 'postgres'")
	}

	pgConfig := buildReplayPostgresConfig(globalCfg.Postgres)
	storage, err := store.NewPostgresStore(pgConfig, globalCfg.TTLSeconds, globalCfg.AsyncWrites)
	if err != nil {
		return nil, fmt.Errorf("failed to create postgres store: %w", err)
	}
	logging.Debugf(
		"Router replay using postgres backend (host=%s, db=%s, table=%s, ttl=%ds, async=%v)",
		pgConfig.Host,
		pgConfig.Database,
		pgConfig.TableName,
		globalCfg.TTLSeconds,
		globalCfg.AsyncWrites,
	)
	return storage, nil
}

func buildReplayPostgresConfig(postgresCfg *config.RouterReplayPostgresConfig) *store.PostgresConfig {
	return &store.PostgresConfig{
		Host:            postgresCfg.Host,
		Port:            postgresCfg.Port,
		Database:        postgresCfg.Database,
		User:            postgresCfg.User,
		Password:        postgresCfg.Password,
		SSLMode:         postgresCfg.SSLMode,
		MaxOpenConns:    postgresCfg.MaxOpenConns,
		MaxIdleConns:    postgresCfg.MaxIdleConns,
		ConnMaxLifetime: postgresCfg.ConnMaxLifetime,
		TableName:       postgresCfg.TableName,
	}
}

func createReplayMilvusStore(globalCfg *config.RouterReplayConfig) (store.Storage, error) {
	if globalCfg.Milvus == nil {
		return nil, fmt.Errorf("milvus config required when store_backend is 'milvus'")
	}

	milvusConfig := buildReplayMilvusConfig(globalCfg.Milvus)
	storage, err := store.NewMilvusStore(milvusConfig, globalCfg.TTLSeconds, globalCfg.AsyncWrites)
	if err != nil {
		return nil, fmt.Errorf("failed to create milvus store: %w", err)
	}
	logging.Debugf(
		"Router replay using milvus backend (address=%s, collection=%s, ttl=%ds, async=%v)",
		milvusConfig.Address,
		milvusConfig.CollectionName,
		globalCfg.TTLSeconds,
		globalCfg.AsyncWrites,
	)
	return storage, nil
}

func buildReplayMilvusConfig(milvusCfg *config.RouterReplayMilvusConfig) *store.MilvusConfig {
	return &store.MilvusConfig{
		Address:          milvusCfg.Address,
		Username:         milvusCfg.Username,
		Password:         milvusCfg.Password,
		CollectionName:   milvusCfg.CollectionName,
		ConsistencyLevel: milvusCfg.ConsistencyLevel,
		ShardNum:         milvusCfg.ShardNum,
	}
}
