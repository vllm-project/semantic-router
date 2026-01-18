// Package store provides storage interfaces and implementations for the
// Router Replay plugin. It supports pluggable backends including memory
// and Redis for storing routing records.
package store

import (
	"context"
	"time"
)

// RoutingRecord represents a recorded routing decision.
// This is a copy of the routerreplay.RoutingRecord to avoid import cycles.
type RoutingRecord struct {
	ID                    string    `json:"id"`
	Timestamp             time.Time `json:"timestamp"`
	RequestID             string    `json:"request_id,omitempty"`
	Decision              string    `json:"decision,omitempty"`
	Category              string    `json:"category,omitempty"`
	OriginalModel         string    `json:"original_model,omitempty"`
	SelectedModel         string    `json:"selected_model,omitempty"`
	ReasoningMode         string    `json:"reasoning_mode,omitempty"`
	Signals               Signal    `json:"signals"`
	RequestBody           string    `json:"request_body,omitempty"`
	ResponseBody          string    `json:"response_body,omitempty"`
	ResponseStatus        int       `json:"response_status,omitempty"`
	FromCache             bool      `json:"from_cache,omitempty"`
	Streaming             bool      `json:"streaming,omitempty"`
	RequestBodyTruncated  bool      `json:"request_body_truncated,omitempty"`
	ResponseBodyTruncated bool      `json:"response_body_truncated,omitempty"`
}

// Signal contains the signals that triggered the routing decision.
type Signal struct {
	Keyword      []string `json:"keyword,omitempty"`
	Embedding    []string `json:"embedding,omitempty"`
	Domain       []string `json:"domain,omitempty"`
	FactCheck    []string `json:"fact_check,omitempty"`
	UserFeedback []string `json:"user_feedback,omitempty"`
	Preference   []string `json:"preference,omitempty"`
}

// ReplayStore defines the interface for storing and retrieving routing records.
// Implementations must be thread-safe.
type ReplayStore interface {
	// StoreRecord stores a new routing record.
	// Returns error if storage fails.
	StoreRecord(ctx context.Context, record *RoutingRecord) error

	// GetRecord retrieves a routing record by ID.
	// Returns nil and ErrNotFound if the record doesn't exist.
	GetRecord(ctx context.Context, recordID string) (*RoutingRecord, error)

	// UpdateRecord updates an existing routing record.
	// Used for attaching response body and status after initial creation.
	// Returns ErrNotFound if the record doesn't exist.
	UpdateRecord(ctx context.Context, record *RoutingRecord) error

	// DeleteRecord deletes a routing record by ID.
	// Returns ErrNotFound if the record doesn't exist.
	DeleteRecord(ctx context.Context, recordID string) error

	// ListRecords lists routing records with pagination and filtering.
	ListRecords(ctx context.Context, opts ListOptions) (*ListResult, error)

	// Close releases resources held by the store.
	Close() error

	// IsEnabled returns whether the store is enabled.
	IsEnabled() bool

	// CheckConnection verifies the store connection is healthy.
	CheckConnection(ctx context.Context) error
}

// ListOptions contains pagination and filtering options.
type ListOptions struct {
	// Limit is the maximum number of items to return.
	Limit int

	// After is the cursor for forward pagination (record ID, exclusive).
	After string

	// Before is the cursor for backward pagination (record ID, exclusive).
	Before string

	// Order is the sort order: "asc" or "desc" (default: "desc" by timestamp).
	Order string

	// Filters
	DecisionName string     // Filter by decision name
	Category     string     // Filter by category
	Model        string     // Filter by selected model
	StartTime    *time.Time // Filter records after this time
	EndTime      *time.Time // Filter records before this time
	FromCache    *bool      // Filter by cache hit status
	RequestID    string     // Filter by request ID
}

// ListResult contains the results of a list operation with pagination metadata.
type ListResult struct {
	// Records is the list of routing records.
	Records []*RoutingRecord

	// HasMore indicates if there are more records beyond this page.
	HasMore bool

	// FirstID is the ID of the first record in the result set.
	FirstID string

	// LastID is the ID of the last record in the result set.
	LastID string
}

// Default pagination constants.
const (
	DefaultListLimit = 20
	MaxListLimit     = 100
)

// DefaultTTL is the default TTL for stored records (30 days).
const DefaultTTL = 30 * 24 * time.Hour

// StoreBackendType defines available store backends.
type StoreBackendType string

const (
	// MemoryStoreType is the in-memory store backend.
	MemoryStoreType StoreBackendType = "memory"

	// RedisStoreType is the Redis store backend.
	RedisStoreType StoreBackendType = "redis"
)

// StoreConfig contains configuration for creating a store.
type StoreConfig struct {
	// BackendType specifies which store implementation to use.
	BackendType StoreBackendType

	// Enabled controls whether storage is active.
	Enabled bool

	// TTLSeconds is the default TTL for stored items (0 = 30 days default).
	TTLSeconds int

	// MaxRecords is the maximum number of records to store (for memory backend).
	MaxRecords int

	// Memory backend configuration
	Memory MemoryStoreConfig

	// Redis backend configuration
	Redis RedisStoreConfig
}

// MemoryStoreConfig contains configuration for the in-memory store.
type MemoryStoreConfig struct {
	// MaxRecords is the maximum number of records to store.
	MaxRecords int
}

// RedisStoreConfig contains configuration for the Redis store.
type RedisStoreConfig struct {
	// Address is the Redis server address (e.g., "localhost:6379").
	Address string

	// Database is the Redis database number.
	Database int

	// Password is the Redis password.
	Password string

	// KeyPrefix is the prefix for all keys (default: "replay:").
	KeyPrefix string

	// TTLSeconds overrides the global TTL for Redis.
	TTLSeconds int
}
