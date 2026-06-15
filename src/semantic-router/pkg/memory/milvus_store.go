package memory

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DefaultMaxRetries is the default number of retry attempts for transient errors
// DefaultRetryBaseDelay is the base delay for exponential backoff (in milliseconds)
const (
	DefaultMaxRetries     = 3
	DefaultRetryBaseDelay = 100
)

// MilvusStore provides memory retrieval from Milvus with similarity threshold filtering
type MilvusStore struct {
	client          client.Client
	collectionName  string
	config          config.MemoryConfig
	enabled         bool
	maxRetries      int
	retryBaseDelay  time.Duration
	embeddingConfig EmbeddingConfig // Unified embedding configuration
}

// MilvusStoreOptions contains configuration for creating a MilvusStore
//
//	Client is the Milvus client instance
//	CollectionName is the name of the Milvus collection
//	Config is the memory configuration
//	Enabled controls whether the store is active
//	EmbeddingConfig is the unified embedding configuration (optional, defaults to mmbert/768)
type MilvusStoreOptions struct {
	Client          client.Client
	CollectionName  string
	Config          config.MemoryConfig
	Enabled         bool
	EmbeddingConfig *EmbeddingConfig // Optional: if nil, derived from Config.Embedding
}

// NewMilvusStore creates a new MilvusStore instance

func NewMilvusStore(options MilvusStoreOptions) (*MilvusStore, error) {
	if !options.Enabled {
		logging.ComponentDebugEvent("memory", "milvus_store_init_skipped", map[string]interface{}{
			"reason":  "disabled",
			"enabled": false,
		})
		return &MilvusStore{
			enabled: false,
		}, nil
	}

	if options.Client == nil {
		return nil, fmt.Errorf("milvus client is required")
	}

	if options.CollectionName == "" {
		return nil, fmt.Errorf("collection name is required")
	}

	// Use default config if not provided
	cfg := options.Config
	if cfg.EmbeddingModel == "" {
		cfg = DefaultMemoryConfig()
	}

	// Initialize embedding configuration
	var embeddingCfg EmbeddingConfig
	if options.EmbeddingConfig != nil {
		embeddingCfg = *options.EmbeddingConfig
	} else {
		embeddingCfg = EmbeddingConfig{Model: EmbeddingModelBERT}
	}

	store := &MilvusStore{
		client:          options.Client,
		collectionName:  options.CollectionName,
		config:          cfg,
		enabled:         options.Enabled,
		maxRetries:      DefaultMaxRetries,
		retryBaseDelay:  DefaultRetryBaseDelay * time.Millisecond,
		embeddingConfig: embeddingCfg,
	}

	// Auto-create collection if it doesn't exist
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := store.ensureCollection(ctx); err != nil {
		return nil, fmt.Errorf("failed to ensure collection exists: %w", err)
	}

	logging.ComponentEvent("memory", "milvus_store_initialized", map[string]interface{}{
		"collection_name": store.collectionName,
		"embedding_model": store.embeddingConfig.Model,
		"dimension":       store.config.Milvus.Dimension,
	})

	return store, nil
}

func (m *MilvusStore) IsEnabled() bool {
	return m.enabled
}

func (m *MilvusStore) CheckConnection(ctx context.Context) error {
	if !m.enabled {
		return nil
	}

	if m.client == nil {
		return fmt.Errorf("milvus client is not initialized")
	}

	// Check if collection exists
	hasCollection, err := m.client.HasCollection(ctx, m.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	if !hasCollection {
		return fmt.Errorf("collection '%s' does not exist", m.collectionName)
	}

	return nil
}

func (m *MilvusStore) Close() error {
	// Note: We don't close the client here as it might be shared
	// The caller is responsible for managing the client lifecycle
	return nil
}

func isTransientError(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())

	// Check for common transient error patterns
	transientPatterns := []string{
		"connection",
		"timeout",
		"deadline exceeded",
		"context deadline exceeded",
		"unavailable",
		"temporary",
		"retry",
		"rate limit",
		"too many requests",
		"server error",
		"internal error",
		"service unavailable",
		"network",
		"broken pipe",
		"connection reset",
		"no connection",
		"connection refused",
	}

	for _, pattern := range transientPatterns {
		if strings.Contains(errStr, pattern) {
			return true
		}
	}

	return false
}

func (m *MilvusStore) retryWithBackoff(ctx context.Context, operation func() error) error {
	var lastErr error

	for attempt := 0; attempt < m.maxRetries; attempt++ {
		lastErr = operation()

		// If no error or non-transient error, return immediately
		if lastErr == nil || !isTransientError(lastErr) {
			return lastErr
		}

		// If this is the last attempt, return the error
		if attempt == m.maxRetries-1 {
			logging.Warnf("MilvusStore: operation failed after %d retries: %v", m.maxRetries, lastErr)
			return lastErr
		}

		// Calculate exponential backoff delay
		// Cap the exponent to avoid overflow (max 30 for safety)
		exponent := attempt
		if exponent < 0 {
			exponent = 0
		} else if exponent > 30 {
			exponent = 30
		}
		delay := m.retryBaseDelay * time.Duration(1<<exponent) // 2^attempt * baseDelay

		logging.Debugf("MilvusStore: transient error on attempt %d/%d, retrying in %v: %v",
			attempt+1, m.maxRetries, delay, lastErr)

		// Wait with context cancellation support
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled during retry: %w", ctx.Err())
		case <-time.After(delay):
			// Continue to next retry
		}
	}

	return lastErr
}
