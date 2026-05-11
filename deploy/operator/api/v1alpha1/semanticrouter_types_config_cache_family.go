/*
Copyright 2026 vLLM Semantic Router Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
)

// SemanticCacheConfig defines semantic cache configuration
type SemanticCacheConfig struct {
	// Enabled controls whether semantic caching is active
	// +kubebuilder:default=true
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// BackendType specifies the cache backend to use
	// Options: "memory" (default), "redis", "valkey", "milvus", "hybrid"
	// +kubebuilder:default="memory"
	// +kubebuilder:validation:Enum=memory;redis;valkey;milvus;hybrid
	// +optional
	BackendType string `json:"backend_type,omitempty"`

	// Similarity threshold for cache hits (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="0.8"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	SimilarityThreshold string `json:"similarity_threshold,omitempty"`

	// MaxEntries is the maximum number of cache entries (for memory/hybrid backends)
	// +kubebuilder:default=1000
	// +optional
	MaxEntries int `json:"max_entries,omitempty"`

	// TTLSeconds is the time-to-live for cache entries in seconds
	// +kubebuilder:default=3600
	// +optional
	TTLSeconds int `json:"ttl_seconds,omitempty"`

	// EvictionPolicy for in-memory cache ("fifo", "lru", "lfu")
	// +kubebuilder:default="fifo"
	// +kubebuilder:validation:Enum=fifo;lru;lfu
	// +optional
	EvictionPolicy string `json:"eviction_policy,omitempty"`

	// Redis configuration (required when backend_type is "redis")
	// +optional
	Redis *RedisCacheConfig `json:"redis,omitempty"`

	// Valkey configuration (required when backend_type is "valkey")
	// +optional
	Valkey *ValkeyCacheConfig `json:"valkey,omitempty"`

	// Milvus configuration (required when backend_type is "milvus")
	// +optional
	Milvus *MilvusCacheConfig `json:"milvus,omitempty"`

	// EmbeddingModel specifies which embedding model to use for semantic similarity
	// Options: "mmbert" (default), "bert", "qwen3", "gemma"
	// +kubebuilder:default="mmbert"
	// +kubebuilder:validation:Enum=bert;qwen3;gemma;mmbert
	// +optional
	EmbeddingModel string `json:"embedding_model,omitempty"`

	// HNSW configuration for hybrid/in-memory backends
	// +optional
	HNSW *HNSWCacheConfig `json:"hnsw,omitempty"`
}

// RedisCacheConfig defines Redis cache backend configuration.
// Configure these settings when using Redis as the semantic cache backend.
type RedisCacheConfig struct {
	// Connection settings for Redis server
	// +optional
	Connection RedisCacheConnection `json:"connection,omitempty"`

	// Index settings for Redis vector search
	// +optional
	Index RedisCacheIndex `json:"index,omitempty"`

	// Search settings for Redis queries
	// +optional
	Search RedisCacheSearch `json:"search,omitempty"`

	// Development settings for Redis cache
	// +optional
	Development RedisCacheDevelopment `json:"development,omitempty"`
}

// RedisCacheConnection defines Redis connection parameters.
type RedisCacheConnection struct {
	// Host is the Redis server hostname or IP address
	// Example: "redis.default.svc.cluster.local"
	// +optional
	Host string `json:"host,omitempty"`

	// Port is the Redis server port
	// +kubebuilder:default=6379
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +optional
	Port int `json:"port,omitempty"`

	// Database is the Redis database number to use
	// +kubebuilder:default=0
	// +kubebuilder:validation:Minimum=0
	// +optional
	Database int `json:"database,omitempty"`

	// Password for Redis authentication (plaintext - consider using PasswordSecretRef instead)
	// +optional
	Password string `json:"password,omitempty"`

	// PasswordSecretRef references a Secret containing the Redis password
	// Preferred over plaintext Password field for security
	// +optional
	PasswordSecretRef *corev1.SecretKeySelector `json:"password_secret_ref,omitempty"`

	// Timeout for Redis operations in seconds
	// +kubebuilder:default=30
	// +kubebuilder:validation:Minimum=0
	// +optional
	Timeout int `json:"timeout,omitempty"`

	// TLS configuration for secure Redis connections
	// +optional
	TLS RedisCacheTLS `json:"tls,omitempty"`
}

// RedisCacheTLS defines TLS settings for Redis connections.
type RedisCacheTLS struct {
	// Enabled controls whether to use TLS for Redis connection
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// CertFile is the path to client certificate file
	// +optional
	CertFile string `json:"cert_file,omitempty"`

	// KeyFile is the path to client key file
	// +optional
	KeyFile string `json:"key_file,omitempty"`

	// CAFile is the path to CA certificate file
	// +optional
	CAFile string `json:"ca_file,omitempty"`
}

// RedisCacheIndex defines Redis vector index configuration.
type RedisCacheIndex struct {
	// Name of the Redis index
	// +kubebuilder:default="semantic_cache_idx"
	// +optional
	Name string `json:"name,omitempty"`

	// Prefix for Redis keys
	// +kubebuilder:default="doc:"
	// +optional
	Prefix string `json:"prefix,omitempty"`

	// VectorField configuration for embeddings
	// +optional
	VectorField RedisCacheVectorField `json:"vector_field,omitempty"`

	// IndexType specifies the index algorithm
	// Options: "HNSW" (recommended), "FLAT"
	// +kubebuilder:default="HNSW"
	// +kubebuilder:validation:Enum=HNSW;FLAT
	// +optional
	IndexType string `json:"index_type,omitempty"`

	// Params for HNSW index
	// +optional
	Params RedisCacheIndexParams `json:"params,omitempty"`
}

// RedisCacheVectorField defines vector field configuration.
type RedisCacheVectorField struct {
	// Name of the vector field
	// +kubebuilder:default="embedding"
	// +optional
	Name string `json:"name,omitempty"`

	// Dimension of the embedding vectors
	// For BERT: 384, for Qwen3: 1024, for Gemma: 768
	// +kubebuilder:validation:Minimum=1
	// +optional
	Dimension int `json:"dimension,omitempty"`

	// MetricType for vector similarity
	// Options: "COSINE", "IP" (inner product), "L2" (Euclidean)
	// +kubebuilder:default="COSINE"
	// +kubebuilder:validation:Enum=COSINE;IP;L2
	// +optional
	MetricType string `json:"metric_type,omitempty"`
}

// RedisCacheIndexParams defines HNSW index parameters.
type RedisCacheIndexParams struct {
	// M is the number of bi-directional links per node
	// Higher values = better recall, more memory
	// +kubebuilder:default=16
	// +kubebuilder:validation:Minimum=2
	// +optional
	M int `json:"M,omitempty"`

	// EfConstruction is the size of dynamic candidate list during construction
	// Higher values = better quality, slower indexing
	// +kubebuilder:default=64
	// +kubebuilder:validation:Minimum=1
	// +optional
	EfConstruction int `json:"efConstruction,omitempty"`
}

// RedisCacheSearch defines Redis search parameters.
type RedisCacheSearch struct {
	// TopK is the number of results to return from vector search
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	// +optional
	TopK int `json:"topk,omitempty"`
}

// RedisCacheDevelopment defines development-mode settings.
type RedisCacheDevelopment struct {
	// DropIndexOnStartup clears the index when router starts (for testing)
	// +kubebuilder:default=false
	// +optional
	DropIndexOnStartup bool `json:"drop_index_on_startup,omitempty"`

	// AutoCreateIndex automatically creates the index if it doesn't exist
	// +kubebuilder:default=true
	// +optional
	AutoCreateIndex bool `json:"auto_create_index,omitempty"`

	// VerboseErrors includes detailed error messages in logs
	// +kubebuilder:default=true
	// +optional
	VerboseErrors bool `json:"verbose_errors,omitempty"`
}

// ValkeyCacheConfig defines Valkey cache backend configuration.
// Configure these settings when using Valkey as the semantic cache backend.
type ValkeyCacheConfig struct {
	// Connection settings for Valkey server
	// +optional
	Connection ValkeyCacheConnection `json:"connection,omitempty"`

	// Index settings for Valkey vector search
	// +optional
	Index ValkeyCacheIndex `json:"index,omitempty"`

	// Search settings for Valkey queries
	// +optional
	Search ValkeyCacheSearch `json:"search,omitempty"`

	// Development settings for Valkey cache
	// +optional
	Development ValkeyCacheDevelopment `json:"development,omitempty"`
}

// ValkeyCacheConnection defines Valkey connection parameters.
type ValkeyCacheConnection struct {
	// Host is the Valkey server hostname or IP address
	// Example: "valkey.default.svc.cluster.local"
	// +optional
	Host string `json:"host,omitempty"`

	// Port is the Valkey server port
	// +kubebuilder:default=6379
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +optional
	Port int `json:"port,omitempty"`

	// Database is the Valkey database number to use
	// +kubebuilder:default=0
	// +kubebuilder:validation:Minimum=0
	// +optional
	Database int `json:"database,omitempty"`

	// Password for Valkey authentication (plaintext - consider using PasswordSecretRef instead)
	// +optional
	Password string `json:"password,omitempty"`

	// PasswordSecretRef references a Secret containing the Valkey password
	// Preferred over plaintext Password field for security
	// +optional
	PasswordSecretRef *corev1.SecretKeySelector `json:"password_secret_ref,omitempty"`

	// Timeout for Valkey operations in seconds
	// +kubebuilder:default=30
	// +kubebuilder:validation:Minimum=0
	// +optional
	Timeout int `json:"timeout,omitempty"`

	// TLS configuration for secure Valkey connections
	// +optional
	TLS ValkeyCacheTLS `json:"tls,omitempty"`
}

// ValkeyCacheTLS defines TLS settings for Valkey connections.
type ValkeyCacheTLS struct {
	// Enabled controls whether to use TLS for Valkey connection
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// CertFile is the path to client certificate file
	// +optional
	CertFile string `json:"cert_file,omitempty"`

	// KeyFile is the path to client key file
	// +optional
	KeyFile string `json:"key_file,omitempty"`

	// CAFile is the path to CA certificate file
	// +optional
	CAFile string `json:"ca_file,omitempty"`
}

// ValkeyCacheIndex defines Valkey vector index configuration.
type ValkeyCacheIndex struct {
	// Name of the Valkey index
	// +kubebuilder:default="semantic_cache_idx"
	// +optional
	Name string `json:"name,omitempty"`

	// Prefix for Valkey keys
	// +kubebuilder:default="doc:"
	// +optional
	Prefix string `json:"prefix,omitempty"`

	// VectorField configuration for embeddings
	// +optional
	VectorField ValkeyCacheVectorField `json:"vector_field,omitempty"`

	// IndexType specifies the index algorithm
	// Options: "HNSW" (recommended), "FLAT"
	// +kubebuilder:default="HNSW"
	// +kubebuilder:validation:Enum=HNSW;FLAT
	// +optional
	IndexType string `json:"index_type,omitempty"`

	// Params for HNSW index
	// +optional
	Params ValkeyCacheIndexParams `json:"params,omitempty"`
}

// ValkeyCacheVectorField defines vector field configuration.
type ValkeyCacheVectorField struct {
	// Name of the vector field
	// +kubebuilder:default="embedding"
	// +optional
	Name string `json:"name,omitempty"`

	// Dimension of the embedding vectors
	// For BERT: 384, for Qwen3: 1024, for Gemma: 768
	// +kubebuilder:validation:Minimum=1
	// +optional
	Dimension int `json:"dimension,omitempty"`

	// MetricType for vector similarity
	// Options: "COSINE", "IP" (inner product), "L2" (Euclidean)
	// +kubebuilder:default="COSINE"
	// +kubebuilder:validation:Enum=COSINE;IP;L2
	// +optional
	MetricType string `json:"metric_type,omitempty"`
}

// ValkeyCacheIndexParams defines HNSW index parameters.
type ValkeyCacheIndexParams struct {
	// M is the number of bi-directional links per node
	// Higher values = better recall, more memory
	// +kubebuilder:default=16
	// +kubebuilder:validation:Minimum=2
	// +optional
	M int `json:"M,omitempty"`

	// EfConstruction is the size of dynamic candidate list during construction
	// Higher values = better quality, slower indexing
	// +kubebuilder:default=64
	// +kubebuilder:validation:Minimum=1
	// +optional
	EfConstruction int `json:"efConstruction,omitempty"`
}

// ValkeyCacheSearch defines Valkey search parameters.
type ValkeyCacheSearch struct {
	// TopK is the number of results to return from vector search
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	// +optional
	TopK int `json:"topk,omitempty"`
}

// ValkeyCacheDevelopment defines development-mode settings.
type ValkeyCacheDevelopment struct {
	// DropIndexOnStartup clears the index when router starts (for testing)
	// +kubebuilder:default=false
	// +optional
	DropIndexOnStartup bool `json:"drop_index_on_startup,omitempty"`

	// AutoCreateIndex automatically creates the index if it doesn't exist
	// +kubebuilder:default=true
	// +optional
	AutoCreateIndex bool `json:"auto_create_index,omitempty"`

	// VerboseErrors includes detailed error messages in logs
	// +kubebuilder:default=true
	// +optional
	VerboseErrors bool `json:"verbose_errors,omitempty"`
}

// MilvusCacheConfig defines Milvus cache backend configuration.
// Configure these settings when using Milvus as the semantic cache backend.
type MilvusCacheConfig struct {
	// Connection settings for Milvus server
	// +optional
	Connection MilvusCacheConnection `json:"connection,omitempty"`

	// Collection settings for Milvus
	// +optional
	Collection MilvusCacheCollection `json:"collection,omitempty"`

	// Search settings for Milvus queries
	// +optional
	Search MilvusCacheSearch `json:"search,omitempty"`

	// Performance tuning for Milvus
	// +optional
	Performance MilvusCachePerformance `json:"performance,omitempty"`

	// DataManagement settings for TTL and compaction
	// +optional
	DataManagement MilvusCacheDataManagement `json:"data_management,omitempty"`

	// Development settings for Milvus cache
	// +optional
	Development MilvusCacheDevelopment `json:"development,omitempty"`
}

// MilvusCacheConnection defines Milvus connection parameters.
type MilvusCacheConnection struct {
	// Host is the Milvus server hostname or IP address
	// +optional
	Host string `json:"host,omitempty"`

	// Port is the Milvus server port
	// +kubebuilder:default=19530
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +optional
	Port int `json:"port,omitempty"`

	// Database name in Milvus
	// +kubebuilder:default="semantic_router_cache"
	// +optional
	Database string `json:"database,omitempty"`

	// Timeout for Milvus operations in seconds
	// +kubebuilder:default=30
	// +kubebuilder:validation:Minimum=0
	// +optional
	Timeout int `json:"timeout,omitempty"`

	// Auth configuration for Milvus authentication
	// +optional
	Auth MilvusCacheAuth `json:"auth,omitempty"`

	// TLS configuration for secure Milvus connections
	// +optional
	TLS MilvusCacheTLS `json:"tls,omitempty"`
}

// MilvusCacheAuth defines Milvus authentication.
type MilvusCacheAuth struct {
	// Enabled controls whether to use authentication
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// Username for Milvus authentication
	// +optional
	Username string `json:"username,omitempty"`

	// Password for Milvus authentication (plaintext - consider using PasswordSecretRef instead)
	// +optional
	Password string `json:"password,omitempty"`

	// PasswordSecretRef references a Secret containing the Milvus password
	// Preferred over plaintext Password field for security
	// +optional
	PasswordSecretRef *corev1.SecretKeySelector `json:"password_secret_ref,omitempty"`
}

// MilvusCacheTLS defines TLS settings for Milvus connections.
type MilvusCacheTLS struct {
	// Enabled controls whether to use TLS
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// CertFile is the path to client certificate file
	// +optional
	CertFile string `json:"cert_file,omitempty"`

	// KeyFile is the path to client key file
	// +optional
	KeyFile string `json:"key_file,omitempty"`

	// CAFile is the path to CA certificate file
	// +optional
	CAFile string `json:"ca_file,omitempty"`
}

// MilvusCacheCollection defines Milvus collection configuration.
type MilvusCacheCollection struct {
	// Name of the Milvus collection
	// +kubebuilder:default="semantic_cache"
	// +optional
	Name string `json:"name,omitempty"`

	// Description of the collection
	// +kubebuilder:default="Semantic cache for LLM request-response pairs"
	// +optional
	Description string `json:"description,omitempty"`

	// VectorField configuration for embeddings
	// +optional
	VectorField MilvusCacheVectorField `json:"vector_field,omitempty"`

	// Index configuration for the collection
	// +optional
	Index MilvusCacheCollectionIndex `json:"index,omitempty"`
}

// MilvusCacheVectorField defines vector field configuration.
type MilvusCacheVectorField struct {
	// Name of the vector field
	// +kubebuilder:default="embedding"
	// +optional
	Name string `json:"name,omitempty"`

	// Dimension of the embedding vectors
	// +kubebuilder:validation:Minimum=1
	// +optional
	Dimension int `json:"dimension,omitempty"`

	// MetricType for vector similarity
	// Options: "IP" (inner product), "L2", "COSINE"
	// +kubebuilder:default="IP"
	// +kubebuilder:validation:Enum=IP;L2;COSINE
	// +optional
	MetricType string `json:"metric_type,omitempty"`
}

// MilvusCacheCollectionIndex defines collection index settings.
type MilvusCacheCollectionIndex struct {
	// Type of index algorithm
	// +kubebuilder:default="HNSW"
	// +kubebuilder:validation:Enum=HNSW;IVF_FLAT;IVF_SQ8;IVF_PQ
	// +optional
	Type string `json:"type,omitempty"`

	// Params for the index
	// +optional
	Params MilvusCacheIndexParams `json:"params,omitempty"`
}

// MilvusCacheIndexParams defines index parameters.
type MilvusCacheIndexParams struct {
	// M is the number of bi-directional links for HNSW
	// +kubebuilder:default=16
	// +kubebuilder:validation:Minimum=2
	// +optional
	M int `json:"M,omitempty"`

	// EfConstruction for HNSW index building
	// +kubebuilder:default=64
	// +kubebuilder:validation:Minimum=1
	// +optional
	EfConstruction int `json:"efConstruction,omitempty"`
}

// MilvusCacheSearch defines Milvus search parameters.
type MilvusCacheSearch struct {
	// Params for search operations
	// +optional
	Params MilvusCacheSearchParams `json:"params,omitempty"`

	// TopK is the number of results to return
	// +kubebuilder:default=10
	// +kubebuilder:validation:Minimum=1
	// +optional
	TopK int `json:"topk,omitempty"`

	// ConsistencyLevel for search operations
	// Options: "Strong", "Session", "Bounded", "Eventually"
	// +kubebuilder:default="Session"
	// +kubebuilder:validation:Enum=Strong;Session;Bounded;Eventually
	// +optional
	ConsistencyLevel string `json:"consistency_level,omitempty"`
}

// MilvusCacheSearchParams defines search-time parameters.
type MilvusCacheSearchParams struct {
	// Ef is the search-time HNSW parameter
	// +kubebuilder:default=64
	// +kubebuilder:validation:Minimum=1
	// +optional
	Ef int `json:"ef,omitempty"`
}

// MilvusCachePerformance defines performance tuning.
type MilvusCachePerformance struct {
	// ConnectionPool settings
	// +optional
	ConnectionPool MilvusCacheConnectionPool `json:"connection_pool,omitempty"`

	// Batch settings for operations
	// +optional
	Batch MilvusCacheBatch `json:"batch,omitempty"`
}

// MilvusCacheConnectionPool defines connection pool settings.
type MilvusCacheConnectionPool struct {
	// MaxConnections in the pool
	// +kubebuilder:default=10
	// +kubebuilder:validation:Minimum=1
	// +optional
	MaxConnections int `json:"max_connections,omitempty"`

	// MaxIdleConnections to keep
	// +kubebuilder:default=5
	// +kubebuilder:validation:Minimum=0
	// +optional
	MaxIdleConnections int `json:"max_idle_connections,omitempty"`

	// AcquireTimeout in seconds
	// +kubebuilder:default=30
	// +kubebuilder:validation:Minimum=0
	// +optional
	AcquireTimeout int `json:"acquire_timeout,omitempty"`
}

// MilvusCacheBatch defines batch operation settings.
type MilvusCacheBatch struct {
	// InsertBatchSize for bulk inserts
	// +kubebuilder:default=100
	// +kubebuilder:validation:Minimum=1
	// +optional
	InsertBatchSize int `json:"insert_batch_size,omitempty"`

	// Timeout for batch operations in seconds
	// +kubebuilder:default=60
	// +kubebuilder:validation:Minimum=0
	// +optional
	Timeout int `json:"timeout,omitempty"`
}

// MilvusCacheDataManagement defines data lifecycle settings.
type MilvusCacheDataManagement struct {
	// TTL settings for automatic expiration
	// +optional
	TTL MilvusCacheTTL `json:"ttl,omitempty"`

	// Compaction settings
	// +optional
	Compaction MilvusCacheCompaction `json:"compaction,omitempty"`
}

// MilvusCacheTTL defines time-to-live settings.
type MilvusCacheTTL struct {
	// Enabled controls whether TTL is active
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// TimestampField is the field used for TTL calculation
	// +kubebuilder:default="created_at"
	// +optional
	TimestampField string `json:"timestamp_field,omitempty"`

	// CleanupInterval in seconds between cleanup runs
	// +kubebuilder:default=3600
	// +kubebuilder:validation:Minimum=0
	// +optional
	CleanupInterval int `json:"cleanup_interval,omitempty"`
}

// MilvusCacheCompaction defines compaction settings.
type MilvusCacheCompaction struct {
	// Enabled controls whether auto-compaction is active
	// +kubebuilder:default=false
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// Interval in seconds between compaction runs
	// +kubebuilder:default=86400
	// +kubebuilder:validation:Minimum=0
	// +optional
	Interval int `json:"interval,omitempty"`
}

// MilvusCacheDevelopment defines development-mode settings.
type MilvusCacheDevelopment struct {
	// DropCollectionOnStartup clears the collection when router starts (for testing)
	// +kubebuilder:default=false
	// +optional
	DropCollectionOnStartup bool `json:"drop_collection_on_startup,omitempty"`

	// AutoCreateCollection automatically creates the collection if it doesn't exist
	// +kubebuilder:default=true
	// +optional
	AutoCreateCollection bool `json:"auto_create_collection,omitempty"`

	// VerboseErrors includes detailed error messages in logs
	// +kubebuilder:default=true
	// +optional
	VerboseErrors bool `json:"verbose_errors,omitempty"`
}

// HNSWCacheConfig defines HNSW index configuration for hybrid/in-memory backends.
type HNSWCacheConfig struct {
	// UseHNSW enables HNSW indexing for faster similarity search
	// +kubebuilder:default=false
	// +optional
	UseHNSW bool `json:"use_hnsw,omitempty"`

	// M is the number of bi-directional links per node
	// +kubebuilder:default=16
	// +kubebuilder:validation:Minimum=2
	// +optional
	M int `json:"hnsw_m,omitempty"`

	// EfConstruction is the size of dynamic candidate list during construction
	// +kubebuilder:default=200
	// +kubebuilder:validation:Minimum=1
	// +optional
	EfConstruction int `json:"hnsw_ef_construction,omitempty"`

	// MaxMemoryEntries limits in-memory entries for hybrid backend
	// +kubebuilder:default=1000
	// +kubebuilder:validation:Minimum=0
	// +optional
	MaxMemoryEntries int `json:"max_memory_entries,omitempty"`
}
