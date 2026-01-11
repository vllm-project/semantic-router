// Package responsestore provides storage interfaces and implementations
// for the Response API. It supports pluggable backends including memory,
// Milvus, and Redis for storing responses and conversations.
package responsestore

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// ResponseStore defines the interface for storing and retrieving responses.
// Implementations must be thread-safe.
type ResponseStore interface {
	// StoreResponse stores a new response.
	// Returns error if the response ID already exists.
	StoreResponse(ctx context.Context, response *responseapi.StoredResponse) error

	// GetResponse retrieves a response by ID.
	// Returns nil and ErrNotFound if the response doesn't exist.
	GetResponse(ctx context.Context, responseID string) (*responseapi.StoredResponse, error)

	// UpdateResponse updates an existing response.
	// Returns ErrNotFound if the response doesn't exist.
	UpdateResponse(ctx context.Context, response *responseapi.StoredResponse) error

	// DeleteResponse deletes a response by ID.
	// Returns ErrNotFound if the response doesn't exist.
	DeleteResponse(ctx context.Context, responseID string) error

	// GetConversationChain retrieves all responses in a conversation chain
	// starting from the given response ID and going back via previous_response_id.
	// Returns responses in chronological order (oldest first).
	GetConversationChain(ctx context.Context, responseID string) ([]*responseapi.StoredResponse, error)

	// ListResponsesByConversation lists all responses for a conversation.
	// Returns responses in chronological order.
	ListResponsesByConversation(ctx context.Context, conversationID string, opts ListOptions) ([]*responseapi.StoredResponse, error)

	// Close releases resources held by the store.
	Close() error

	// IsEnabled returns whether the store is enabled.
	IsEnabled() bool

	// CheckConnection verifies the store connection is healthy.
	CheckConnection(ctx context.Context) error
}

// ConversationStore defines the interface for storing and retrieving conversations.
// This is optional - implementations may combine this with ResponseStore.
type ConversationStore interface {
	// CreateConversation creates a new conversation.
	CreateConversation(ctx context.Context, conversation *responseapi.StoredConversation) error

	// GetConversation retrieves a conversation by ID.
	GetConversation(ctx context.Context, conversationID string) (*responseapi.StoredConversation, error)

	// UpdateConversation updates an existing conversation.
	UpdateConversation(ctx context.Context, conversation *responseapi.StoredConversation) error

	// DeleteConversation deletes a conversation and optionally its responses.
	DeleteConversation(ctx context.Context, conversationID string, deleteResponses bool) error

	// ListConversations lists conversations with pagination.
	ListConversations(ctx context.Context, opts ListOptions) ([]*responseapi.StoredConversation, error)

	// AddResponseToConversation adds a response ID to a conversation.
	AddResponseToConversation(ctx context.Context, conversationID, responseID string) error
}

// CombinedStore combines ResponseStore and ConversationStore interfaces.
// Most implementations should implement this interface.
type CombinedStore interface {
	ResponseStore
	ConversationStore
}

// ListOptions contains pagination and filtering options.
type ListOptions struct {
	// Limit is the maximum number of items to return.
	Limit int

	// After is the cursor for forward pagination (exclusive).
	After string

	// Before is the cursor for backward pagination (exclusive).
	Before string

	// Order is the sort order: "asc" or "desc" (default: "desc").
	Order string
}

// StoreConfig contains configuration for creating a store.
type StoreConfig struct {
	// BackendType specifies which store implementation to use.
	BackendType StoreBackendType `yaml:"backend_type"`

	// Enabled controls whether storage is active.
	Enabled bool `yaml:"enabled"`

	// TTLSeconds is the default TTL for stored items (0 = no expiration).
	// Default is 30 days (2592000 seconds) to match OpenAI.
	TTLSeconds int `yaml:"ttl_seconds"`

	// Memory backend configuration
	Memory MemoryStoreConfig `yaml:"memory,omitempty"`

	// Milvus backend configuration
	Milvus MilvusStoreConfig `yaml:"milvus,omitempty"`

	// Redis backend configuration
	Redis RedisStoreConfig `yaml:"redis,omitempty"`
}

// StoreBackendType defines available store backends.
type StoreBackendType string

const (
	// MemoryStoreType is the in-memory store backend.
	MemoryStoreType StoreBackendType = "memory"

	// MilvusStoreType is the Milvus store backend.
	MilvusStoreType StoreBackendType = "milvus"

	// RedisStoreType is the Redis store backend.
	RedisStoreType StoreBackendType = "redis"
)

// MemoryStoreConfig contains configuration for the in-memory store.
type MemoryStoreConfig struct {
	// MaxResponses is the maximum number of responses to store.
	MaxResponses int `yaml:"max_responses"`

	// MaxConversations is the maximum number of conversations to store.
	MaxConversations int `yaml:"max_conversations"`
}

// MilvusStoreConfig contains configuration for the Milvus store.
type MilvusStoreConfig struct {
	// Address is the Milvus server address (e.g., "localhost:19530").
	Address string `yaml:"address"`

	// Database is the Milvus database name.
	Database string `yaml:"database,omitempty"`

	// ResponseCollection is the collection name for responses.
	ResponseCollection string `yaml:"response_collection"`

	// ConversationCollection is the collection name for conversations.
	ConversationCollection string `yaml:"conversation_collection"`

	// ConfigPath is the path to additional Milvus configuration.
	ConfigPath string `yaml:"config_path,omitempty"`
}

// RedisStoreConfig contains configuration for the Redis store.
type RedisStoreConfig struct {
	// Basic connection settings (inline configuration)
	// Address is the Redis server address (e.g., "localhost:6379").
	// Not used in cluster mode (use ClusterAddresses instead).
	Address string `yaml:"address,omitempty" json:"address,omitempty"`

	// Password for Redis authentication (leave empty if no auth required).
	Password string `yaml:"password,omitempty" json:"password,omitempty"`

	// DB is the Redis database number (0-15 for standalone Redis).
	// Must be 0 for cluster mode (Redis Cluster only supports DB 0).
	DB int `yaml:"db" json:"db"`

	// KeyPrefix is the base prefix for all Redis keys (e.g., "sr:").
	// Combined with type prefixes to create keys like: sr:response:xxx, sr:conversation:xxx
	// Default: "sr:"
	KeyPrefix string `yaml:"key_prefix,omitempty" json:"key_prefix,omitempty"`

	// Cluster support
	// ClusterMode enables Redis Cluster mode (for distributed deployments).
	// When true, use ClusterAddresses instead of Address.
	ClusterMode bool `yaml:"cluster_mode,omitempty" json:"cluster_mode,omitempty"`

	// ClusterAddresses is a list of Redis cluster node addresses.
	// Only used when ClusterMode is true.
	// Example: ["redis-node-1:6379", "redis-node-2:6379", "redis-node-3:6379"]
	ClusterAddresses []string `yaml:"cluster_addresses,omitempty" json:"cluster_addresses,omitempty"`

	// Connection pooling settings
	// PoolSize is the maximum number of socket connections.
	// Default: 10 per CPU as reported by runtime.GOMAXPROCS.
	PoolSize int `yaml:"pool_size,omitempty" json:"pool_size,omitempty"`

	// MinIdleConns is the minimum number of idle connections to maintain.
	// Default: 2
	MinIdleConns int `yaml:"min_idle_conns,omitempty" json:"min_idle_conns,omitempty"`

	// MaxRetries is the maximum number of retries before giving up.
	// Default: 3
	// Set to -1 to disable retries.
	MaxRetries int `yaml:"max_retries,omitempty" json:"max_retries,omitempty"`

	// Timeout settings (in seconds)
	// DialTimeout is the timeout for establishing new connections.
	// Default: 5 seconds
	DialTimeout int `yaml:"dial_timeout,omitempty" json:"dial_timeout,omitempty"`

	// ReadTimeout is the timeout for socket reads.
	// Default: 3 seconds
	// Set to -1 to disable timeout.
	ReadTimeout int `yaml:"read_timeout,omitempty" json:"read_timeout,omitempty"`

	// WriteTimeout is the timeout for socket writes.
	// Default: 3 seconds
	// Set to -1 to disable timeout.
	WriteTimeout int `yaml:"write_timeout,omitempty" json:"write_timeout,omitempty"`

	// TLS/SSL settings
	// TLSEnabled enables TLS/SSL for secure connections.
	// Recommended for production environments.
	TLSEnabled bool `yaml:"tls_enabled,omitempty" json:"tls_enabled,omitempty"`

	// TLSCertPath is the path to the TLS certificate file.
	// Required when TLSEnabled is true.
	TLSCertPath string `yaml:"tls_cert_path,omitempty" json:"tls_cert_path,omitempty"`

	// TLSKeyPath is the path to the TLS private key file.
	// Required when TLSEnabled is true.
	TLSKeyPath string `yaml:"tls_key_path,omitempty" json:"tls_key_path,omitempty"`

	// TLSCAPath is the path to the CA certificate file for verification.
	// Optional - if not provided, system CA pool is used.
	TLSCAPath string `yaml:"tls_ca_path,omitempty" json:"tls_ca_path,omitempty"`

	// ConfigPath is the path to an external Redis configuration file.
	// If provided, settings from the external file take precedence over inline settings.
	// This is useful for complex configurations or sharing config across environments.
	// Example: "config/response-api/redis-cluster.yaml"
	ConfigPath string `yaml:"config_path,omitempty" json:"config_path,omitempty"`
}

// DefaultTTL is the default TTL for stored responses (30 days).
const DefaultTTL = 30 * 24 * time.Hour

// DefaultListLimit is the default limit for list operations.
const DefaultListLimit = 20

// MaxListLimit is the maximum limit for list operations.
const MaxListLimit = 100
