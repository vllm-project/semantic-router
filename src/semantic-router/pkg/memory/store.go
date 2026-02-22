package memory

import (
	"context"
	"fmt"
	"sync"
)

// globalMemoryStore holds the global memory store instance for API access.
// Set by the ExtProc router during initialization.
var (
	globalMemoryStore Store
	globalStoreMu     sync.RWMutex
)

// SetGlobalMemoryStore sets the global memory store instance.
// Called by the ExtProc router after creating the memory store.
func SetGlobalMemoryStore(store Store) {
	globalStoreMu.Lock()
	defer globalStoreMu.Unlock()
	globalMemoryStore = store
}

// GetGlobalMemoryStore returns the global memory store instance.
// Returns nil if memory is not enabled or not yet initialized.
func GetGlobalMemoryStore() Store {
	globalStoreMu.RLock()
	defer globalStoreMu.RUnlock()
	return globalMemoryStore
}

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

	// List returns memories matching the filter criteria with pagination.
	// UserID is required. Returns memories sorted by created_at descending.
	List(ctx context.Context, opts ListOptions) (*ListResult, error)

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

// HierarchicalStore is an optional interface for stores that support
// hierarchical two-phase retrieval with group-aware scoping and relation surfacing.
type HierarchicalStore interface {
	Store

	// HierarchicalRetrieve performs a two-phase search:
	//  Phase 1: Search category nodes (is_category=true) with group-aware filters.
	//  Phase 2: Drill down into top categories, propagating parent scores.
	HierarchicalRetrieve(ctx context.Context, opts HierarchicalRetrieveOptions) ([]*RetrieveResult, error)

	// StoreRelation persists a link between two memories.
	StoreRelation(ctx context.Context, rel MemoryRelation) error

	// GetRelations returns relations originating from the given memory ID.
	GetRelations(ctx context.Context, memoryID string, limit int) ([]MemoryRelation, error)
}

// AsHierarchicalStore attempts to cast a Store to HierarchicalStore.
// Returns nil, false if the store does not implement hierarchical retrieval.
func AsHierarchicalStore(s Store) (HierarchicalStore, bool) {
	hs, ok := s.(HierarchicalStore)
	return hs, ok
}

// ErrHierarchicalNotSupported is returned when hierarchical retrieval is
// requested on a store that does not implement HierarchicalStore.
var ErrHierarchicalNotSupported = fmt.Errorf("store does not support hierarchical retrieval")
