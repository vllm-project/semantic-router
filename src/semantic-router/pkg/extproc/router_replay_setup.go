package extproc

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func createReplayRuntime(cfg *config.RouterConfig) (map[string]*routerreplay.Recorder, *routerreplay.Recorder, bool, error) {
	backend := replayBackend(cfg)
	if usesSharedReplayStorage(backend) {
		recorders, replayRecorder, err := initializeSharedReplayRecorders(cfg, backend)
		shareReplayOutcomeQueueWithFallback(recorders, replayRecorder)
		return recorders, replayRecorder, replayRecorder != nil, err
	}

	replayRecorders, globalRecorder, err := initializeMemoryReplayRecorders(cfg, backend)
	if err != nil {
		return nil, nil, false, err
	}
	shareReplayOutcomeQueueWithFallback(replayRecorders, globalRecorder)
	// The memory store a configuration chose; one that keeps text in memory
	// because the file store holds none has already said so.
	if cfg.RouterReplay.StoreBackendName() == "memory" && (len(replayRecorders) > 0 || globalRecorder != nil) {
		logging.Warnf("Router replay store_backend is set to %q — all replay records "+
			"will be lost on router restart. Use \"postgres\" or \"redis\" for durable storage in production.",
			backend)
	}
	return replayRecorders, globalRecorder, false, nil
}

func shareReplayOutcomeQueueWithFallback(recorders map[string]*routerreplay.Recorder, fallback *routerreplay.Recorder) {
	// A failed initialization and a config with no replay decisions both arrive
	// here empty; neither should allocate a queue and its writer context.
	if len(recorders) == 0 && fallback == nil {
		return
	}
	group := make([]*routerreplay.Recorder, 0, len(recorders))
	for _, recorder := range recorders {
		group = append(group, recorder)
	}
	if fallback != nil {
		group = append(group, fallback)
	}
	routerreplay.ShareOutcomeQueue(group...)
}

// initializeReplayRecorders creates replay recorders for the decisions that record replay.
func initializeReplayRecorders(cfg *config.RouterConfig) map[string]*routerreplay.Recorder {
	recorders, _, _, _ := createReplayRuntime(cfg)
	return recorders
}

// initializeMemoryReplayRecorders opens the stores cfg names for records kept
// in memory, and maps each decision to the recorder of its store.
func initializeMemoryReplayRecorders(
	cfg *config.RouterConfig,
	backend string,
) (map[string]*routerreplay.Recorder, *routerreplay.Recorder, error) {
	recorders := make(map[string]*routerreplay.Recorder)
	var globalRecorder *routerreplay.Recorder
	for _, plan := range cfg.RouterReplayMemoryStores() {
		recorder, err := createReplayRecorder(plan.Name, backend, plan.Policy, cfg)
		if err != nil {
			_ = closeReplayRecorders(globalRecorder, recorders, false)
			if plan.Global {
				return nil, nil, fmt.Errorf("failed to initialize global replay recorder: %w", err)
			}
			return nil, nil, fmt.Errorf("failed to initialize replay recorder for decision %s: %w", plan.Name, err)
		}
		if plan.Global {
			globalRecorder = recorder
		}
		for _, decision := range plan.Decisions {
			recorders[decision] = recorder
		}
	}
	return recorders, globalRecorder, nil
}

func initializeSharedReplayRecorders(
	cfg *config.RouterConfig,
	backend string,
) (map[string]*routerreplay.Recorder, *routerreplay.Recorder, error) {
	recorders := make(map[string]*routerreplay.Recorder)
	var (
		sharedStore    store.Storage
		replayRecorder *routerreplay.Recorder
	)

	// Global capture also covers direct requests and failures before a decision.
	if policy := cfg.RouterReplayPreDecisionPolicy(); policy != nil {
		storage, err := createSharedReplayStore(backend, cfg, policy)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to initialize global replay store: %w", err)
		}
		sharedStore = storage
		replayRecorder = createSharedReplayRecorder(sharedStore, policy)
	}

	for _, ref := range cfg.RoutingDecisionRefs() {
		decision := ref.Decision
		pluginCfg := replayConfigForDecisionRef(cfg, ref)
		if pluginCfg == nil {
			continue
		}

		if sharedStore == nil {
			storage, err := createSharedReplayStore(backend, cfg, pluginCfg)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to initialize shared replay store for backend %s: %w", backend, err)
			}
			sharedStore = storage
		}

		recorder := createSharedReplayRecorder(sharedStore, pluginCfg)
		recorders[config.RoutingDecisionKey(ref.Recipe, decision.Name)] = recorder
		if replayRecorder == nil {
			replayRecorder = recorder
		}
	}

	return recorders, replayRecorder, nil
}

// createReplayRecorder creates a single replay recorder with the appropriate storage backend.
func createReplayRecorder(
	decisionName string,
	backend string,
	pluginCfg *config.RouterReplayPluginConfig,
	cfg *config.RouterConfig,
) (*routerreplay.Recorder, error) {
	maxBodyBytes := resolveReplayMaxBodyBytes(pluginCfg.MaxBodyBytes)

	storage, err := createReplayStore(decisionName, backend, pluginCfg, cfg)
	if err != nil {
		return nil, err
	}

	recorder := routerreplay.NewRecorder(storage)
	recorder.SetCapturePolicy(pluginCfg.CapturesRequestText(), pluginCfg.CapturesResponseText(), maxBodyBytes)
	setReplayTextLimits(recorder, pluginCfg)
	return recorder, nil
}

// setReplayTextLimits gives a recorder the plugin's limits on the prompt and
// the tool trace.
func setReplayTextLimits(recorder *routerreplay.Recorder, pluginCfg *config.RouterReplayPluginConfig) {
	recorder.SetMaxPromptBytes(pluginCfg.PromptByteLimit())
	recorder.SetMaxToolTraceBytes(pluginCfg.ToolTraceByteLimit())
	recorder.SetMaxRawToolTraceBytes(pluginCfg.RawToolTraceByteLimit())
	recorder.SetMaxToolTraceTotalBytes(pluginCfg.ToolTraceTotalByteLimit())
	recorder.SetMaxToolTraceSteps(pluginCfg.MaxToolTraceSteps)
}

func createSharedReplayRecorder(
	storage store.Storage,
	pluginCfg *config.RouterReplayPluginConfig,
) *routerreplay.Recorder {
	recorder := routerreplay.NewRecorder(storage)
	recorder.SetCapturePolicy(
		pluginCfg.CapturesRequestText(),
		pluginCfg.CapturesResponseText(),
		resolveReplayMaxBodyBytes(pluginCfg.MaxBodyBytes),
	)
	setReplayTextLimits(recorder, pluginCfg)
	return recorder
}

func usesSharedReplayStorage(backend string) bool {
	switch backend {
	case "file", "postgres", "redis", "milvus", "qdrant":
		return true
	default:
		return false
	}
}

// replayBackend is the backend that keeps the configuration's records, and
// logs when the records stay in memory because a plugin keeps text.
func replayBackend(cfg *config.RouterConfig) string {
	backend := cfg.RouterReplayStoreBackend()
	if backend != cfg.RouterReplay.StoreBackendName() {
		logging.Warnf("Router replay keeps prompt or body text, which the file store does not hold; " +
			"its records stay in memory. Configure postgres or redis to keep them across restarts.")
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
	cfg *config.RouterConfig,
) (store.Storage, error) {
	globalCfg := &cfg.RouterReplay
	switch backend {
	case "memory":
		return createReplayMemoryStore(decisionName, pluginCfg, cfg), nil
	case "redis":
		return createReplayRedisStore(globalCfg)
	case "postgres":
		return createReplayPostgresStore(globalCfg)
	case "milvus":
		return createReplayMilvusStore(globalCfg)
	case "qdrant":
		return createReplayQdrantStore(globalCfg)
	default:
		return nil, fmt.Errorf(
			"unknown store_backend: %s (supported: %s)",
			backend, strings.Join(config.RouterReplayStoreBackends(), ", "),
		)
	}
}

func createSharedReplayStore(
	backend string,
	cfg *config.RouterConfig,
	pluginCfg *config.RouterReplayPluginConfig,
) (store.Storage, error) {
	globalCfg := &cfg.RouterReplay
	switch backend {
	case "file":
		return createReplayFileStore(cfg, pluginCfg), nil
	case "redis":
		return createReplayRedisStore(globalCfg)
	case "postgres":
		return createReplayPostgresStore(globalCfg)
	case "milvus":
		return createReplayMilvusStore(globalCfg)
	case "qdrant":
		return createReplayQdrantStore(globalCfg)
	default:
		return nil, fmt.Errorf("shared replay storage not supported for backend %s", backend)
	}
}

// replayFileMaxRecords caps the records the file store keeps: at about 8 KB a
// record, 1,000 records make a file of about 8 MB, and 16 MB just before it is
// rewritten.
const replayFileMaxRecords = 1000

// createReplayFileStore keeps the records in .vllm-sr/router-replay.jsonl in
// the configuration file's directory, at most replayFileMaxRecords of them,
// or in memory, as before the file store existed, where there is no such
// file, the directory cannot be written or another process holds the file.
func createReplayFileStore(cfg *config.RouterConfig, pluginCfg *config.RouterReplayPluginConfig) store.Storage {
	path := filepath.Join(replayStateDir(cfg.ConfigBaseDir), "router-replay.jsonl")
	err := errors.New("the configuration was not read from a file")
	if cfg.ConfigBaseDir != "" {
		maxRecords := min(pluginCfg.MaxRecords, replayFileMaxRecords)
		var storage *store.FileStore
		if storage, err = store.OpenFileStore(path, maxRecords, cfg.RouterReplay.TTLSeconds); err == nil {
			logging.Infof("Router replay keeps records in %s (max_records=%d, ttl=%ds)",
				path, maxRecords, cfg.RouterReplay.TTLSeconds)
			return storage
		}
	}
	logging.Warnf("Router replay cannot keep records in %s (%v); they stay in memory and are lost on restart. "+
		"Make that directory writable, or configure postgres or redis.", path, err)
	return store.NewMemoryStore(pluginCfg.MaxRecords, cfg.RouterReplay.TTLSeconds)
}

// replayStateDir is the .vllm-sr directory of the project whose configuration
// lives in configDir: configDir itself for a generated
// .vllm-sr/runtime-config*.yaml, as configsnapshot reads it, else the
// .vllm-sr directory in configDir.
func replayStateDir(configDir string) string {
	if filepath.Base(configDir) == ".vllm-sr" {
		return configDir
	}
	return filepath.Join(configDir, ".vllm-sr")
}

// createReplayMemoryStore keeps a recorder's records in memory. A reload of a
// configuration read from a file shares the store of the generation it
// replaces, keyed by that file's directory and the recorder, so the records
// are lost only on restart.
func createReplayMemoryStore(
	decisionName string,
	pluginCfg *config.RouterReplayPluginConfig,
	cfg *config.RouterConfig,
) store.Storage {
	maxRecords := pluginCfg.MaxRecords
	if maxRecords <= 0 {
		maxRecords = routerreplay.DefaultMaxRecords
	}
	logging.Debugf("Router replay for %s using memory backend (max_records=%d)", decisionName, maxRecords)
	if cfg.ConfigBaseDir == "" {
		return store.NewMemoryStore(maxRecords, cfg.RouterReplay.TTLSeconds)
	}
	key := filepath.Join(replayStateDir(cfg.ConfigBaseDir), "router-replay") + " " + decisionName
	return store.OpenMemoryStore(key, maxRecords, cfg.RouterReplay.TTLSeconds)
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
	tableName := pgConfig.TableName
	if tableName == "" {
		tableName = store.DefaultPostgresTableName
	}
	logging.Debugf(
		"Router replay using postgres backend (host=%s, db=%s, table=%s, ttl=%ds, async=%v)",
		pgConfig.Host,
		pgConfig.Database,
		tableName,
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

func createReplayQdrantStore(globalCfg *config.RouterReplayConfig) (store.Storage, error) {
	if globalCfg.Qdrant == nil {
		return nil, fmt.Errorf("qdrant config required when store_backend is 'qdrant'")
	}

	qdrantConfig := buildReplayQdrantConfig(globalCfg.Qdrant)
	storage, err := store.NewQdrantStore(qdrantConfig, globalCfg.TTLSeconds, globalCfg.AsyncWrites)
	if err != nil {
		return nil, fmt.Errorf("failed to create qdrant store: %w", err)
	}
	logging.Debugf(
		"Router replay using qdrant backend (host=%s, port=%d, collection=%s, ttl=%ds, async=%v)",
		qdrantConfig.Host,
		qdrantConfig.Port,
		qdrantConfig.CollectionName,
		globalCfg.TTLSeconds,
		globalCfg.AsyncWrites,
	)
	return storage, nil
}

func buildReplayQdrantConfig(qdrantCfg *config.RouterReplayQdrantConfig) *store.QdrantConfig {
	return &store.QdrantConfig{
		Host:           qdrantCfg.Host,
		Port:           qdrantCfg.Port,
		APIKey:         qdrantCfg.APIKey,
		UseTLS:         qdrantCfg.UseTLS,
		CollectionName: qdrantCfg.CollectionName,
	}
}
