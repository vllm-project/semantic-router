package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func createReplayRuntime(cfg *config.RouterConfig) (map[string]*routerreplay.Recorder, *routerreplay.Recorder) {
	replayRecorders := initializeReplayRecorders(cfg)

	var replayRecorder *routerreplay.Recorder
	for _, recorder := range replayRecorders {
		replayRecorder = recorder
		break
	}

	return replayRecorders, replayRecorder
}

// initializeReplayRecorders creates replay recorders for decisions with router_replay plugin configured.
func initializeReplayRecorders(cfg *config.RouterConfig) map[string]*routerreplay.Recorder {
	recorders := make(map[string]*routerreplay.Recorder)

	for _, decision := range cfg.Decisions {
		pluginCfg := decision.GetRouterReplayConfig()
		if pluginCfg == nil || !pluginCfg.Enabled {
			continue
		}

		recorder, err := createReplayRecorder(decision.Name, pluginCfg, &cfg.RouterReplay)
		if err != nil {
			logging.Errorf("Failed to initialize replay recorder for decision %s: %v", decision.Name, err)
			continue
		}

		recorders[decision.Name] = recorder
	}

	return recorders
}

// createReplayRecorder creates a single replay recorder with the appropriate storage backend.
func createReplayRecorder(
	decisionName string,
	pluginCfg *config.RouterReplayPluginConfig,
	globalCfg *config.RouterReplayConfig,
) (*routerreplay.Recorder, error) {
	backend := resolveReplayStoreBackend(globalCfg.StoreBackend)
	maxBodyBytes := resolveReplayMaxBodyBytes(pluginCfg.MaxBodyBytes)

	storage, err := createReplayStore(decisionName, backend, pluginCfg, globalCfg)
	if err != nil {
		return nil, err
	}

	recorder := routerreplay.NewRecorder(storage)
	recorder.SetCapturePolicy(pluginCfg.CaptureRequestBody, pluginCfg.CaptureResponseBody, maxBodyBytes)
	return recorder, nil
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
		return createReplayRedisStore(decisionName, globalCfg)
	case "postgres":
		return createReplayPostgresStore(decisionName, globalCfg)
	case "milvus":
		return createReplayMilvusStore(decisionName, globalCfg)
	default:
		return nil, fmt.Errorf(
			"unknown store_backend: %s (supported: memory, redis, postgres, milvus)",
			backend,
		)
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

func createReplayRedisStore(
	decisionName string,
	globalCfg *config.RouterReplayConfig,
) (store.Storage, error) {
	if globalCfg.Redis == nil {
		return nil, fmt.Errorf("redis config required when store_backend is 'redis'")
	}

	redisConfig := buildReplayRedisConfig(decisionName, globalCfg.Redis)
	storage, err := store.NewRedisStore(redisConfig, globalCfg.TTLSeconds, globalCfg.AsyncWrites)
	if err != nil {
		return nil, fmt.Errorf("failed to create redis store: %w", err)
	}
	logging.Debugf(
		"Router replay for %s using redis backend (address=%s, key_prefix=%s, ttl=%ds, async=%v)",
		decisionName,
		redisConfig.Address,
		redisConfig.KeyPrefix,
		globalCfg.TTLSeconds,
		globalCfg.AsyncWrites,
	)
	return storage, nil
}

func buildReplayRedisConfig(
	decisionName string,
	redisCfg *config.RouterReplayRedisConfig,
) *store.RedisConfig {
	keyPrefix := decisionName + ":"
	if redisCfg.KeyPrefix != "" {
		keyPrefix = redisCfg.KeyPrefix + ":" + decisionName + ":"
	}

	return &store.RedisConfig{
		Address:       redisCfg.Address,
		DB:            redisCfg.DB,
		Password:      redisCfg.Password,
		UseTLS:        redisCfg.UseTLS,
		TLSSkipVerify: redisCfg.TLSSkipVerify,
		MaxRetries:    redisCfg.MaxRetries,
		PoolSize:      redisCfg.PoolSize,
		KeyPrefix:     keyPrefix,
	}
}

func createReplayPostgresStore(
	decisionName string,
	globalCfg *config.RouterReplayConfig,
) (store.Storage, error) {
	if globalCfg.Postgres == nil {
		return nil, fmt.Errorf("postgres config required when store_backend is 'postgres'")
	}

	pgConfig := buildReplayPostgresConfig(decisionName, globalCfg.Postgres)
	storage, err := store.NewPostgresStore(pgConfig, globalCfg.TTLSeconds, globalCfg.AsyncWrites)
	if err != nil {
		return nil, fmt.Errorf("failed to create postgres store: %w", err)
	}
	logging.Debugf(
		"Router replay for %s using postgres backend (host=%s, db=%s, table=%s, ttl=%ds, async=%v)",
		decisionName,
		pgConfig.Host,
		pgConfig.Database,
		pgConfig.TableName,
		globalCfg.TTLSeconds,
		globalCfg.AsyncWrites,
	)
	return storage, nil
}

func buildReplayPostgresConfig(
	decisionName string,
	postgresCfg *config.RouterReplayPostgresConfig,
) *store.PostgresConfig {
	tableName := decisionName + "_replay_records"
	if postgresCfg.TableName != "" {
		tableName = postgresCfg.TableName + "_" + decisionName
	}

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
		TableName:       tableName,
	}
}

func createReplayMilvusStore(
	decisionName string,
	globalCfg *config.RouterReplayConfig,
) (store.Storage, error) {
	if globalCfg.Milvus == nil {
		return nil, fmt.Errorf("milvus config required when store_backend is 'milvus'")
	}

	milvusConfig := buildReplayMilvusConfig(decisionName, globalCfg.Milvus)
	storage, err := store.NewMilvusStore(milvusConfig, globalCfg.TTLSeconds, globalCfg.AsyncWrites)
	if err != nil {
		return nil, fmt.Errorf("failed to create milvus store: %w", err)
	}
	logging.Debugf(
		"Router replay for %s using milvus backend (address=%s, collection=%s, ttl=%ds, async=%v)",
		decisionName,
		milvusConfig.Address,
		milvusConfig.CollectionName,
		globalCfg.TTLSeconds,
		globalCfg.AsyncWrites,
	)
	return storage, nil
}

func buildReplayMilvusConfig(
	decisionName string,
	milvusCfg *config.RouterReplayMilvusConfig,
) *store.MilvusConfig {
	collectionName := decisionName + "_replay_records"
	if milvusCfg.CollectionName != "" {
		collectionName = milvusCfg.CollectionName + "_" + decisionName
	}

	return &store.MilvusConfig{
		Address:          milvusCfg.Address,
		Username:         milvusCfg.Username,
		Password:         milvusCfg.Password,
		CollectionName:   collectionName,
		ConsistencyLevel: milvusCfg.ConsistencyLevel,
		ShardNum:         milvusCfg.ShardNum,
	}
}
