package memory

import (
	"context"
)

// Store defines the interface for storing and retrieving memories.
// Implementations must be thread-safe.
type Store interface {
	// Store saves a new memory.
	// Returns error if the memory ID already exists.
	Store(ctx context.Context, memory *Memory) error

	// Retrieve performs semantic search for relevant memories.
	// Uses embedding-based similarity search.
	Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error)

	// Get retrieves a memory by ID.
	// Returns nil and error if the memory doesn't exist.
	Get(ctx context.Context, id string) (*Memory, error)

	// Update modifies an existing memory.
	// Returns error if the memory doesn't exist.
	Update(ctx context.Context, id string, memory *Memory) error

	// Forget deletes a memory by ID.
	// Returns error if the memory doesn't exist.
	Forget(ctx context.Context, id string) error

	// ForgetByScope deletes all memories matching the scope.
	// Scope includes UserID (required), ProjectID (optional), Types (optional).
	ForgetByScope(ctx context.Context, scope MemoryScope) error

	// IsEnabled returns whether the store is enabled.
	IsEnabled() bool

	// CheckConnection verifies the store connection is healthy.
	CheckConnection(ctx context.Context) error

	// Close releases resources held by the store.
	Close() error
}
