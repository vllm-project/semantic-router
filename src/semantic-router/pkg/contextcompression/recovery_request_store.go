package contextcompression

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Recovery defaults shared by every context action that stores recoverable
// content. A per-request byte default belongs to the action that writes; the
// rest is one contract for the one shared store.
const (
	DefaultRecoveryTTL           = 15 * time.Minute
	DefaultRecoveryMaxTotalBytes = 256 * 1024 * 1024
	DefaultRecoveryMaxRetrievals = 8
	// DefaultCompressionRecoveryBytesPerRequest is compression's own bound.
	DefaultCompressionRecoveryBytesPerRequest = 10 * 1024 * 1024
)

// ErrRecoveryRequestLimitExceeded reports a write that would take one request
// past its recovery byte budget. It is a stable value so each action's failure
// mode can classify it without matching error text.
var ErrRecoveryRequestLimitExceeded = errors.New("recovery request byte limit exceeded")

// RequestRecoveryStore bounds one request's recovery writes across every
// context action. Each action also checks its own payload before writing; this
// store is where their combined writes meet one limit. Only a write the backing
// store accepted consumes budget. Reads and invalidation pass through, and the
// backing store, which is process-wide, is not closed here.
type RequestRecoveryStore struct {
	backing RecoveryStore
	limit   int
	mu      sync.Mutex
	used    int
}

// NewRequestRecoveryStore wraps the process-wide store for one request. A
// limit of zero or less leaves the aggregate unbounded.
func NewRequestRecoveryStore(backing RecoveryStore, limit int) *RequestRecoveryStore {
	return &RequestRecoveryStore{backing: backing, limit: limit}
}

func (s *RequestRecoveryStore) Put(ctx context.Context, entry RecoveryEntry, ttl time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	size := len(entry.Content)
	if s.limit > 0 && s.used+size > s.limit {
		return ErrRecoveryRequestLimitExceeded
	}
	if err := s.backing.Put(ctx, entry, ttl); err != nil {
		return err
	}
	s.used += size
	return nil
}

func (s *RequestRecoveryStore) Get(ctx context.Context, scope string, key string) (RecoveryEntry, error) {
	return s.backing.Get(ctx, scope, key)
}

func (s *RequestRecoveryStore) InvalidateScope(ctx context.Context, scope string) (int64, error) {
	return s.backing.InvalidateScope(ctx, scope)
}

func (s *RequestRecoveryStore) Health(ctx context.Context) error {
	return s.backing.Health(ctx)
}

func (s *RequestRecoveryStore) Close() error {
	return nil
}

// StoreRecoveryContent writes one recoverable entry and returns its key. It is
// the single entry-building protocol: the key is derived from the scope and
// content, so the same content in the same scope is stored once.
func StoreRecoveryContent(
	ctx context.Context,
	store RecoveryStore,
	scope string,
	content string,
	ttl time.Duration,
) (string, error) {
	if ttl <= 0 {
		return "", fmt.Errorf("recovery TTL must be positive, got %s", ttl)
	}
	key := recoveryKey(scope, content)
	now := time.Now().UTC()
	entry := RecoveryEntry{
		Key:       key,
		Scope:     scope,
		Content:   content,
		CreatedAt: now,
		ExpiresAt: now.Add(ttl),
	}
	if err := store.Put(ctx, entry, ttl); err != nil {
		return "", err
	}
	return key, nil
}
