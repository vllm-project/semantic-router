package memory

import (
	"context"
	"sync"
	"sync/atomic"
)

type storeValue struct {
	store Store
}

// StoreSlot publishes a generation-owned memory store atomically. The slot's
// identity stays stable while an unavailable backend is reconnecting, so all
// consumers of that generation observe the same store once it becomes ready.
type StoreSlot struct {
	current atomic.Pointer[storeValue]
}

// NewStoreSlot creates a slot with store as its initial value. A nil store
// leaves the slot empty until Install succeeds.
func NewStoreSlot(store Store) *StoreSlot {
	slot := &StoreSlot{}
	if store != nil {
		slot.current.Store(&storeValue{store: store})
	}
	return slot
}

// Load returns the currently published store, or nil while it is unavailable.
func (s *StoreSlot) Load() Store {
	if s == nil {
		return nil
	}
	value := s.current.Load()
	if value == nil {
		return nil
	}
	return value.store
}

// Install publishes store only when the slot is still empty. This prevents a
// late or duplicate reconnect attempt from replacing a store already in use.
func (s *StoreSlot) Install(store Store) bool {
	if s == nil || store == nil {
		return false
	}
	return s.current.CompareAndSwap(nil, &storeValue{store: store})
}

// Close removes and closes the published store. Generation shutdown calls it
// only after request and persistence work has drained.
func (s *StoreSlot) Close() error {
	if s == nil {
		return nil
	}
	value := s.current.Swap(nil)
	if value == nil || value.store == nil {
		return nil
	}
	return value.store.Close()
}

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
