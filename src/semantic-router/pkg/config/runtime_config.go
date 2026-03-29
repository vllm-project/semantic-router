package config

// LooperConfig defines configuration for multi-model execution.
type LooperConfig struct {
	Endpoint         string            `yaml:"endpoint"`
	ModelEndpoints   map[string]string `yaml:"model_endpoints,omitempty"`
	GRPCMaxMsgSizeMB int               `yaml:"grpc_max_msg_size_mb,omitempty"`
	TimeoutSeconds   int               `yaml:"timeout_seconds,omitempty"`
	RetryCount       int               `yaml:"retry_count,omitempty"`
	Headers          map[string]string `yaml:"headers,omitempty"`
}

func (l *LooperConfig) IsEnabled() bool {
	return l.Endpoint != ""
}

func (l *LooperConfig) GetTimeout() int {
	if l.TimeoutSeconds <= 0 {
		return 30
	}
	return l.TimeoutSeconds
}

func (l *LooperConfig) GetGRPCMaxMsgSize() int {
	if l.GRPCMaxMsgSizeMB <= 0 {
		return 4 * 1024 * 1024
	}
	return l.GRPCMaxMsgSizeMB * 1024 * 1024
}

// RedisConfig defines the complete configuration structure for Redis cache backend.
type RedisConfig struct {
	Connection struct {
		Host     string `json:"host" yaml:"host"`
		Port     int    `json:"port" yaml:"port"`
		Database int    `json:"database" yaml:"database"`
		Password string `json:"password" yaml:"password"`
		Timeout  int    `json:"timeout" yaml:"timeout"`
		TLS      struct {
			Enabled  bool   `json:"enabled" yaml:"enabled"`
			CertFile string `json:"cert_file" yaml:"cert_file"`
			KeyFile  string `json:"key_file" yaml:"key_file"`
			CAFile   string `json:"ca_file" yaml:"ca_file"`
		} `json:"tls" yaml:"tls"`
	} `json:"connection" yaml:"connection"`
	Index struct {
		Name        string `json:"name" yaml:"name"`
		Prefix      string `json:"prefix" yaml:"prefix"`
		VectorField struct {
			Name       string `json:"name" yaml:"name"`
			Dimension  int    `json:"dimension" yaml:"dimension"`
			MetricType string `json:"metric_type" yaml:"metric_type"`
		} `json:"vector_field" yaml:"vector_field"`
		IndexType string `json:"index_type" yaml:"index_type"`
		Params    struct {
			M              int `json:"M" yaml:"M"`
			EfConstruction int `json:"efConstruction" yaml:"efConstruction"`
		} `json:"params" yaml:"params"`
	} `json:"index" yaml:"index"`
	Search struct {
		TopK int `json:"topk" yaml:"topk"`
	} `json:"search" yaml:"search"`
	Development struct {
		DropIndexOnStartup bool `json:"drop_index_on_startup" yaml:"drop_index_on_startup"`
		AutoCreateIndex    bool `json:"auto_create_index" yaml:"auto_create_index"`
		VerboseErrors      bool `json:"verbose_errors" yaml:"verbose_errors"`
	} `json:"development" yaml:"development"`
	Logging struct {
		Level          string `json:"level" yaml:"level"`
		EnableQueryLog bool   `json:"enable_query_log" yaml:"enable_query_log"`
		EnableMetrics  bool   `json:"enable_metrics" yaml:"enable_metrics"`
	} `json:"logging" yaml:"logging"`
}

// ValkeyConfig defines the complete configuration structure for Valkey cache backend.
type ValkeyConfig struct {
	Connection struct {
		Host     string `json:"host" yaml:"host"`
		Port     int    `json:"port" yaml:"port"`
		Database int    `json:"database" yaml:"database"`
		Password string `json:"password" yaml:"password"`
		Timeout  int    `json:"timeout" yaml:"timeout"`
		TLS      struct {
			Enabled  bool   `json:"enabled" yaml:"enabled"`
			CertFile string `json:"cert_file" yaml:"cert_file"`
			KeyFile  string `json:"key_file" yaml:"key_file"`
			CAFile   string `json:"ca_file" yaml:"ca_file"`
		} `json:"tls" yaml:"tls"`
	} `json:"connection" yaml:"connection"`
	Index struct {
		Name        string `json:"name" yaml:"name"`
		Prefix      string `json:"prefix" yaml:"prefix"`
		VectorField struct {
			Name       string `json:"name" yaml:"name"`
			Dimension  int    `json:"dimension" yaml:"dimension"`
			MetricType string `json:"metric_type" yaml:"metric_type"`
		} `json:"vector_field" yaml:"vector_field"`
		IndexType string `json:"index_type" yaml:"index_type"`
		Params    struct {
			M              int `json:"M" yaml:"M"`
			EfConstruction int `json:"efConstruction" yaml:"efConstruction"`
		} `json:"params" yaml:"params"`
	} `json:"index" yaml:"index"`
	Search struct {
		TopK int `json:"topk" yaml:"topk"`
	} `json:"search" yaml:"search"`
	Development struct {
		DropIndexOnStartup bool `json:"drop_index_on_startup" yaml:"drop_index_on_startup"`
		AutoCreateIndex    bool `json:"auto_create_index" yaml:"auto_create_index"`
		VerboseErrors      bool `json:"verbose_errors" yaml:"verbose_errors"`
	} `json:"development" yaml:"development"`
	Logging struct {
		Level          string `json:"level" yaml:"level"`
		EnableQueryLog bool   `json:"enable_query_log" yaml:"enable_query_log"`
		EnableMetrics  bool   `json:"enable_metrics" yaml:"enable_metrics"`
	} `json:"logging" yaml:"logging"`
}

// MilvusConfig defines the complete configuration structure for Milvus cache backend.
type MilvusConfig struct {
	Connection struct {
		Host     string `json:"host" yaml:"host"`
		Port     int    `json:"port" yaml:"port"`
		Database string `json:"database" yaml:"database"`
		Timeout  int    `json:"timeout" yaml:"timeout"`
		Auth     struct {
			Enabled  bool   `json:"enabled" yaml:"enabled"`
			Username string `json:"username" yaml:"username"`
			Password string `json:"password" yaml:"password"`
		} `json:"auth" yaml:"auth"`
		TLS struct {
			Enabled  bool   `json:"enabled" yaml:"enabled"`
			CertFile string `json:"cert_file" yaml:"cert_file"`
			KeyFile  string `json:"key_file" yaml:"key_file"`
			CAFile   string `json:"ca_file" yaml:"ca_file"`
		} `json:"tls" yaml:"tls"`
	} `json:"connection" yaml:"connection"`
	Collection struct {
		Name        string `json:"name" yaml:"name"`
		Description string `json:"description" yaml:"description"`
		VectorField struct {
			Name       string `json:"name" yaml:"name"`
			Dimension  int    `json:"dimension" yaml:"dimension"`
			MetricType string `json:"metric_type" yaml:"metric_type"`
		} `json:"vector_field" yaml:"vector_field"`
		Index struct {
			Type   string `json:"type" yaml:"type"`
			Params struct {
				M              int `json:"M" yaml:"M"`
				EfConstruction int `json:"efConstruction" yaml:"efConstruction"`
			} `json:"params" yaml:"params"`
		} `json:"index" yaml:"index"`
	} `json:"collection" yaml:"collection"`
	Search struct {
		Params struct {
			Ef int `json:"ef" yaml:"ef"`
		} `json:"params" yaml:"params"`
		TopK             int    `json:"topk" yaml:"topk"`
		ConsistencyLevel string `json:"consistency_level" yaml:"consistency_level"`
	} `json:"search" yaml:"search"`
	Performance struct {
		ConnectionPool struct {
			MaxConnections     int `json:"max_connections" yaml:"max_connections"`
			MaxIdleConnections int `json:"max_idle_connections" yaml:"max_idle_connections"`
			AcquireTimeout     int `json:"acquire_timeout" yaml:"acquire_timeout"`
		} `json:"connection_pool" yaml:"connection_pool"`
		Batch struct {
			InsertBatchSize int `json:"insert_batch_size" yaml:"insert_batch_size"`
			Timeout         int `json:"timeout" yaml:"timeout"`
		} `json:"batch" yaml:"batch"`
	} `json:"performance" yaml:"performance"`
	DataManagement struct {
		TTL struct {
			Enabled         bool   `json:"enabled" yaml:"enabled"`
			TimestampField  string `json:"timestamp_field" yaml:"timestamp_field"`
			CleanupInterval int    `json:"cleanup_interval" yaml:"cleanup_interval"`
		} `json:"ttl" yaml:"ttl"`
		Compaction struct {
			Enabled  bool `json:"enabled" yaml:"enabled"`
			Interval int  `json:"interval" yaml:"interval"`
		} `json:"compaction" yaml:"compaction"`
	} `json:"data_management" yaml:"data_management"`
	Logging struct {
		Level          string `json:"level" yaml:"level"`
		EnableQueryLog bool   `json:"enable_query_log" yaml:"enable_query_log"`
		EnableMetrics  bool   `json:"enable_metrics" yaml:"enable_metrics"`
	} `json:"logging" yaml:"logging"`
	Development struct {
		DropCollectionOnStartup bool `json:"drop_collection_on_startup" yaml:"drop_collection_on_startup"`
		AutoCreateCollection    bool `json:"auto_create_collection" yaml:"auto_create_collection"`
		VerboseErrors           bool `json:"verbose_errors" yaml:"verbose_errors"`
	} `json:"development" yaml:"development"`
}

type SemanticCache struct {
	BackendType         string        `yaml:"backend_type,omitempty"`
	Enabled             bool          `yaml:"enabled"`
	SimilarityThreshold *float32      `yaml:"similarity_threshold,omitempty"`
	MaxEntries          int           `yaml:"max_entries,omitempty"`
	TTLSeconds          int           `yaml:"ttl_seconds,omitempty"`
	EvictionPolicy      string        `yaml:"eviction_policy,omitempty"`
	Redis               *RedisConfig  `yaml:"redis,omitempty"`
	Valkey              *ValkeyConfig `yaml:"valkey,omitempty"`
	Milvus              *MilvusConfig `yaml:"milvus,omitempty"`
	EmbeddingModel      string        `yaml:"embedding_model,omitempty"`
}

type MemoryConfig struct {
	Enabled                    bool                       `yaml:"enabled,omitempty"`
	AutoStore                  bool                       `yaml:"auto_store,omitempty"`
	DisabledRoutes             []string                   `yaml:"disabled_routes,omitempty"`
	DisabledModels             []string                   `yaml:"disabled_models,omitempty"`
	Milvus                     MemoryMilvusConfig         `yaml:"milvus,omitempty"`
	RedisCache                 *MemoryRedisCacheConfig    `yaml:"redis_cache,omitempty"`
	EmbeddingModel             string                     `yaml:"embedding_model,omitempty"`
	ExtractionBatchSize        int                        `yaml:"extraction_batch_size,omitempty"`
	DefaultRetrievalLimit      int                        `yaml:"default_retrieval_limit,omitempty"`
	DefaultSimilarityThreshold float32                    `yaml:"default_similarity_threshold,omitempty"`
	HybridSearch               bool                       `yaml:"hybrid_search,omitempty"`
	HybridMode                 string                     `yaml:"hybrid_mode,omitempty"`
	AdaptiveThreshold          bool                       `yaml:"adaptive_threshold,omitempty"`
	QualityScoring             MemoryQualityScoringConfig `yaml:"quality_scoring,omitempty"`
	Reflection                 MemoryReflectionConfig     `yaml:"reflection,omitempty"`
}

// MemoryRedisCacheConfig configures an optional Redis hot cache in front of Milvus retrieval.
type MemoryRedisCacheConfig struct {
	Enabled    bool   `yaml:"enabled,omitempty"`
	Address    string `yaml:"address,omitempty"`
	TTLSeconds int    `yaml:"ttl_seconds,omitempty"`
	DB         int    `yaml:"db,omitempty"`
	KeyPrefix  string `yaml:"key_prefix,omitempty"`
	Password   string `yaml:"password,omitempty"`
}

type MemoryQualityScoringConfig struct {
	InitialStrengthDays int     `yaml:"initial_strength_days,omitempty"`
	PruneThreshold      float64 `yaml:"prune_threshold,omitempty"`
	MaxMemoriesPerUser  int     `yaml:"max_memories_per_user,omitempty"`
}

type MemoryReflectionConfig struct {
	Enabled          *bool    `yaml:"enabled,omitempty"`
	Algorithm        string   `yaml:"algorithm,omitempty"`
	MaxInjectTokens  int      `yaml:"max_inject_tokens,omitempty"`
	RecencyDecayDays int      `yaml:"recency_decay_days,omitempty"`
	DedupThreshold   float32  `yaml:"dedup_threshold,omitempty"`
	BlockPatterns    []string `yaml:"block_patterns,omitempty"`
}

func (c MemoryReflectionConfig) ReflectionEnabled() bool {
	if c.Enabled != nil {
		return *c.Enabled
	}
	return true
}

type MemoryMilvusConfig struct {
	Address       string `yaml:"address"`
	Collection    string `yaml:"collection,omitempty"`
	Dimension     int    `yaml:"dimension,omitempty"`
	NumPartitions int    `yaml:"num_partitions,omitempty"`
}

// ResponseAPIConfig controls response and conversation history storage.
// StoreBackend defaults to "redis" for durable storage that survives router
// restarts. Set to "memory" only for local development — all history is lost
// when the router process exits.
type ResponseAPIConfig struct {
	Enabled      bool                   `yaml:"enabled"`
	StoreBackend string                 `yaml:"store_backend,omitempty"`
	TTLSeconds   int                    `yaml:"ttl_seconds,omitempty"`
	MaxResponses int                    `yaml:"max_responses,omitempty"`
	Redis        ResponseAPIRedisConfig `yaml:"redis,omitempty"`
}

type ResponseAPIRedisConfig struct {
	Address          string   `yaml:"address,omitempty" json:"address,omitempty"`
	Password         string   `yaml:"password,omitempty" json:"password,omitempty"`
	DB               int      `yaml:"db" json:"db"`
	KeyPrefix        string   `yaml:"key_prefix,omitempty" json:"key_prefix,omitempty"`
	ClusterMode      bool     `yaml:"cluster_mode,omitempty" json:"cluster_mode,omitempty"`
	ClusterAddresses []string `yaml:"cluster_addresses,omitempty" json:"cluster_addresses,omitempty"`
	PoolSize         int      `yaml:"pool_size,omitempty" json:"pool_size,omitempty"`
	MinIdleConns     int      `yaml:"min_idle_conns,omitempty" json:"min_idle_conns,omitempty"`
	MaxRetries       int      `yaml:"max_retries,omitempty" json:"max_retries,omitempty"`
	DialTimeout      int      `yaml:"dial_timeout,omitempty" json:"dial_timeout,omitempty"`
	ReadTimeout      int      `yaml:"read_timeout,omitempty" json:"read_timeout,omitempty"`
	WriteTimeout     int      `yaml:"write_timeout,omitempty" json:"write_timeout,omitempty"`
	TLSEnabled       bool     `yaml:"tls_enabled,omitempty" json:"tls_enabled,omitempty"`
	TLSCertPath      string   `yaml:"tls_cert_path,omitempty" json:"tls_cert_path,omitempty"`
	TLSKeyPath       string   `yaml:"tls_key_path,omitempty" json:"tls_key_path,omitempty"`
	TLSCAPath        string   `yaml:"tls_ca_path,omitempty" json:"tls_ca_path,omitempty"`
	ConfigPath       string   `yaml:"config_path,omitempty" json:"config_path,omitempty"`
}

// RouterReplayConfig controls routing-decision replay record storage.
// StoreBackend defaults to "postgres" for durable, SQL-queryable storage
// that survives router restarts. Supported backends: "postgres", "redis",
// "milvus", "memory". Use "redis" for lightweight deployments that already
// run Redis. Set to "memory" only for local development — all replay
// records are lost when the router process exits.
type RouterReplayConfig struct {
	StoreBackend string                      `json:"store_backend,omitempty" yaml:"store_backend,omitempty"`
	TTLSeconds   int                         `json:"ttl_seconds,omitempty" yaml:"ttl_seconds,omitempty"`
	AsyncWrites  bool                        `json:"async_writes,omitempty" yaml:"async_writes,omitempty"`
	Redis        *RouterReplayRedisConfig    `json:"redis,omitempty" yaml:"redis,omitempty"`
	Postgres     *RouterReplayPostgresConfig `json:"postgres,omitempty" yaml:"postgres,omitempty"`
	Milvus       *RouterReplayMilvusConfig   `json:"milvus,omitempty" yaml:"milvus,omitempty"`
}

type RouterReplayRedisConfig struct {
	Address       string `json:"address" yaml:"address"`
	DB            int    `json:"db,omitempty" yaml:"db,omitempty"`
	Password      string `json:"password,omitempty" yaml:"password,omitempty"`
	UseTLS        bool   `json:"use_tls,omitempty" yaml:"use_tls,omitempty"`
	TLSSkipVerify bool   `json:"tls_skip_verify,omitempty" yaml:"tls_skip_verify,omitempty"`
	MaxRetries    int    `json:"max_retries,omitempty" yaml:"max_retries,omitempty"`
	PoolSize      int    `json:"pool_size,omitempty" yaml:"pool_size,omitempty"`
	KeyPrefix     string `json:"key_prefix,omitempty" yaml:"key_prefix,omitempty"`
}

type RouterReplayPostgresConfig struct {
	Host            string `json:"host" yaml:"host"`
	Port            int    `json:"port,omitempty" yaml:"port,omitempty"`
	Database        string `json:"database" yaml:"database"`
	User            string `json:"user" yaml:"user"`
	Password        string `json:"password,omitempty" yaml:"password,omitempty"`
	SSLMode         string `json:"ssl_mode,omitempty" yaml:"ssl_mode,omitempty"`
	MaxOpenConns    int    `json:"max_open_conns,omitempty" yaml:"max_open_conns,omitempty"`
	MaxIdleConns    int    `json:"max_idle_conns,omitempty" yaml:"max_idle_conns,omitempty"`
	ConnMaxLifetime int    `json:"conn_max_lifetime,omitempty" yaml:"conn_max_lifetime,omitempty"`
	TableName       string `json:"table_name,omitempty" yaml:"table_name,omitempty"`
}

type RouterReplayMilvusConfig struct {
	Address          string `json:"address" yaml:"address"`
	Username         string `json:"username,omitempty" yaml:"username,omitempty"`
	Password         string `json:"password,omitempty" yaml:"password,omitempty"`
	CollectionName   string `json:"collection_name,omitempty" yaml:"collection_name,omitempty"`
	ConsistencyLevel string `json:"consistency_level,omitempty" yaml:"consistency_level,omitempty"`
	ShardNum         int    `json:"shard_num,omitempty" yaml:"shard_num,omitempty"`
}
