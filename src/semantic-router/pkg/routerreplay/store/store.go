package store

import (
	"context"
	"time"
)

// Signal represents various routing signals captured during a request.
type Signal struct {
	Keyword      []string `json:"keyword,omitempty"`
	Embedding    []string `json:"embedding,omitempty"`
	Domain       []string `json:"domain,omitempty"`
	FactCheck    []string `json:"fact_check,omitempty"`
	UserFeedback []string `json:"user_feedback,omitempty"`
	Preference   []string `json:"preference,omitempty"`
	Language     []string `json:"language,omitempty"`
	Latency      []string `json:"latency,omitempty"`
	Context      []string `json:"context,omitempty"`
}

// Record represents a routing decision record with metadata and captured payloads.
type Record struct {
	ID                    string    `json:"id"`
	Timestamp             time.Time `json:"timestamp"`
	RequestID             string    `json:"request_id,omitempty"`
	Decision              string    `json:"decision,omitempty"`
	Category              string    `json:"category,omitempty"`
	OriginalModel         string    `json:"original_model,omitempty"`
	SelectedModel         string    `json:"selected_model,omitempty"`
	ReasoningMode         string    `json:"reasoning_mode,omitempty"`
	ConfidenceScore       float64   `json:"confidence_score,omitempty"`
	SelectionMethod       string    `json:"selection_method,omitempty"`
	Signals               Signal    `json:"signals"`
	RequestBody           string    `json:"request_body,omitempty"`
	ResponseBody          string    `json:"response_body,omitempty"`
	ResponseStatus        int       `json:"response_status,omitempty"`
	FromCache             bool      `json:"from_cache,omitempty"`
	Streaming             bool      `json:"streaming,omitempty"`
	RequestBodyTruncated  bool      `json:"request_body_truncated,omitempty"`
	ResponseBodyTruncated bool      `json:"response_body_truncated,omitempty"`
}

// Storage is the interface that all storage backends must implement.
type Storage interface {
	// Add inserts a new record. Returns the record ID.
	Add(ctx context.Context, record Record) (string, error)

	// Get retrieves a record by ID. Returns false if not found.
	Get(ctx context.Context, id string) (Record, bool, error)

	// List retrieves all records, ordered by timestamp descending.
	List(ctx context.Context) ([]Record, error)

	// UpdateStatus updates the response status and flags for an existing record.
	UpdateStatus(ctx context.Context, id string, status int, fromCache bool, streaming bool) error

	// AttachRequest updates the request body for an existing record.
	AttachRequest(ctx context.Context, id string, body string, truncated bool) error

	// AttachResponse updates the response body for an existing record.
	AttachResponse(ctx context.Context, id string, body string, truncated bool) error

	// Close releases resources held by the storage backend.
	Close() error
}

// Config holds common configuration options for all storage backends.
type Config struct {
	Backend      string // "memory", "redis", "postgres", "milvus"
	TTLSeconds   int    // Time-to-live for records (0 = no expiration)
	AsyncWrites  bool   // Enable asynchronous writes
	MaxBodyBytes int    // Maximum bytes to store for request/response bodies

	// Backend-specific configurations
	Redis    *RedisConfig
	Postgres *PostgresConfig
	Milvus   *MilvusConfig
}

// RedisConfig holds Redis-specific configuration.
type RedisConfig struct {
	Address  string `json:"address" yaml:"address"`
	DB       int    `json:"db" yaml:"db"`
	Password string `json:"password" yaml:"password"`
	// Optional TLS configuration
	UseTLS        bool   `json:"use_tls,omitempty" yaml:"use_tls,omitempty"`
	TLSSkipVerify bool   `json:"tls_skip_verify,omitempty" yaml:"tls_skip_verify,omitempty"`
	MaxRetries    int    `json:"max_retries,omitempty" yaml:"max_retries,omitempty"`
	PoolSize      int    `json:"pool_size,omitempty" yaml:"pool_size,omitempty"`
	KeyPrefix     string `json:"key_prefix,omitempty" yaml:"key_prefix,omitempty"`
}

// PostgresConfig holds PostgreSQL-specific configuration.
type PostgresConfig struct {
	Host     string `json:"host" yaml:"host"`
	Port     int    `json:"port" yaml:"port"`
	Database string `json:"database" yaml:"database"`
	User     string `json:"user" yaml:"user"`
	Password string `json:"password" yaml:"password"`
	SSLMode  string `json:"ssl_mode,omitempty" yaml:"ssl_mode,omitempty"` // disable, require, verify-ca, verify-full
	// Connection pool settings
	MaxOpenConns    int    `json:"max_open_conns,omitempty" yaml:"max_open_conns,omitempty"`
	MaxIdleConns    int    `json:"max_idle_conns,omitempty" yaml:"max_idle_conns,omitempty"`
	ConnMaxLifetime int    `json:"conn_max_lifetime,omitempty" yaml:"conn_max_lifetime,omitempty"` // seconds
	TableName       string `json:"table_name,omitempty" yaml:"table_name,omitempty"`
}

// MilvusConfig holds Milvus-specific configuration.
type MilvusConfig struct {
	Address        string `json:"address" yaml:"address"`
	Username       string `json:"username,omitempty" yaml:"username,omitempty"`
	Password       string `json:"password,omitempty" yaml:"password,omitempty"`
	CollectionName string `json:"collection_name,omitempty" yaml:"collection_name,omitempty"`
	// Milvus specific settings
	ConsistencyLevel string `json:"consistency_level,omitempty" yaml:"consistency_level,omitempty"` // Strong, Session, Bounded, Eventually
	ShardNum         int    `json:"shard_num,omitempty" yaml:"shard_num,omitempty"`
}
