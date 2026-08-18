package sessiontelemetry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// RouterSessionStateStore is an optional shared backing store for protection state.
type RouterSessionStateStore interface {
	Load(sessionID string) (RouterSessionSnapshot, bool, error)
	Save(snapshot RouterSessionSnapshot, ttl time.Duration) error
	Close() error
}

// RedisRouterSessionStoreConfig configures shared session protection state.
type RedisRouterSessionStoreConfig struct {
	Address   string
	Password  string
	Database  int
	Timeout   time.Duration
	TTL       time.Duration
	KeyPrefix string
}

type redisRouterSessionStore struct {
	client    *redis.Client
	timeout   time.Duration
	ttl       time.Duration
	keyPrefix string
}

// RouterSessionStateStoreSlot owns one store generation and leases individual
// Load/Save operations. Retirement prevents new leases, waits for in-flight
// operations, and closes the store exactly once.
type RouterSessionStateStoreSlot struct {
	store      RouterSessionStateStore
	mu         sync.Mutex
	operations sync.WaitGroup
	retired    bool
	retiredCh  chan struct{}
	closeOnce  sync.Once
	closeErr   error
}

var (
	routerSessionStoreMu sync.RWMutex
	routerSessionStore   *RouterSessionStateStoreSlot
)

// NewRedisRouterSessionStateStore creates a bounded-time Redis state store.
func NewRedisRouterSessionStateStore(
	config RedisRouterSessionStoreConfig,
) (RouterSessionStateStore, error) {
	if config.Address == "" {
		return nil, fmt.Errorf("redis router session store address is required")
	}
	timeout := config.Timeout
	if timeout <= 0 {
		timeout = 50 * time.Millisecond
	}
	prefix := config.KeyPrefix
	if prefix == "" {
		prefix = "vsr:router-session:v1:"
	}
	client := redis.NewClient(&redis.Options{
		Addr:         config.Address,
		Password:     config.Password,
		DB:           config.Database,
		DialTimeout:  timeout,
		ReadTimeout:  timeout,
		WriteTimeout: timeout,
	})
	return &redisRouterSessionStore{
		client:    client,
		timeout:   timeout,
		ttl:       config.TTL,
		keyPrefix: prefix,
	}, nil
}

// NewRouterSessionStateStoreSlot creates an unpublished, generation-owned
// store slot. A nil store has no slot.
func NewRouterSessionStateStoreSlot(store RouterSessionStateStore) *RouterSessionStateStoreSlot {
	if store == nil {
		return nil
	}
	return &RouterSessionStateStoreSlot{
		store:     store,
		retiredCh: make(chan struct{}),
	}
}

func (s *RouterSessionStateStoreSlot) acquire() (RouterSessionStateStore, func(), bool) {
	if s == nil {
		return nil, nil, false
	}
	s.mu.Lock()
	if s.retired {
		s.mu.Unlock()
		return nil, nil, false
	}
	s.operations.Add(1)
	store := s.store
	s.mu.Unlock()
	return store, s.operations.Done, true
}

// RetireAndClose prevents new operation leases, waits for existing operations,
// and closes the owned store exactly once.
func (s *RouterSessionStateStoreSlot) RetireAndClose() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	if !s.retired {
		s.retired = true
		close(s.retiredCh)
	}
	s.mu.Unlock()
	s.operations.Wait()
	s.closeOnce.Do(func() {
		s.closeErr = s.store.Close()
	})
	return s.closeErr
}

// SetRouterSessionStateStore swaps the optional shared store and closes the
// previous one. It is intended for standalone callers and tests. Router
// generations use PublishRouterSessionStateStore so retirement, rather than
// publication, owns resource cleanup.
func SetRouterSessionStateStore(store RouterSessionStateStore) {
	next := NewRouterSessionStateStoreSlot(store)
	routerSessionStoreMu.Lock()
	previous := routerSessionStore
	if previous != nil && previous.store == store {
		routerSessionStoreMu.Unlock()
		return
	}
	routerSessionStore = next
	routerSessionStoreMu.Unlock()
	if previous != nil {
		_ = previous.RetireAndClose()
	}
}

// PublishRouterSessionStateStore makes a generation-owned store current
// without closing the previous generation's store. The retiring router closes
// its store after all of its leases have drained.
func PublishRouterSessionStateStore(store *RouterSessionStateStoreSlot) {
	routerSessionStoreMu.Lock()
	routerSessionStore = store
	routerSessionStoreMu.Unlock()
}

// UnpublishRouterSessionStateStore clears store only when it is still current.
// This prevents a retiring generation from clearing a newer generation's
// published store.
func UnpublishRouterSessionStateStore(store *RouterSessionStateStoreSlot) {
	if store == nil {
		return
	}
	routerSessionStoreMu.Lock()
	if routerSessionStore == store {
		routerSessionStore = nil
	}
	routerSessionStoreMu.Unlock()
}

func acquireCurrentRouterSessionStateStore() (RouterSessionStateStore, func(), bool) {
	routerSessionStoreMu.RLock()
	store := routerSessionStore
	if store == nil {
		routerSessionStoreMu.RUnlock()
		return nil, nil, false
	}
	stateStore, release, acquired := store.acquire()
	routerSessionStoreMu.RUnlock()
	return stateStore, release, acquired
}

func persistRouterSessionState(sessionID string) {
	if sessionID == "" {
		return
	}
	store, release, acquired := acquireCurrentRouterSessionStateStore()
	if !acquired {
		return
	}
	defer release()
	snapshot, ok := GetRouterSessionSnapshot(sessionID, time.Now())
	if !ok {
		return
	}
	_ = store.Save(snapshot, routerMemoryTTL)
}

func loadSharedRouterSessionSnapshot(
	sessionID string,
	now time.Time,
) (RouterSessionSnapshot, bool) {
	store, release, acquired := acquireCurrentRouterSessionStateStore()
	if !acquired {
		return RouterSessionSnapshot{}, false
	}
	defer release()
	snapshot, found, err := store.Load(sessionID)
	if err != nil || !found {
		return RouterSessionSnapshot{}, false
	}
	if snapshot.SessionID != sessionID {
		return RouterSessionSnapshot{}, false
	}
	if now.IsZero() {
		now = time.Now()
	}
	idleFor := now.Sub(snapshot.LastSeen)
	if idleFor < 0 {
		idleFor = 0
	}
	if idleFor > routerMemoryTTL {
		return RouterSessionSnapshot{}, false
	}
	snapshot.IdleFor = idleFor
	hydrateRouterSessionSnapshot(snapshot)
	return snapshot, true
}

func hydrateRouterSessionSnapshot(snapshot RouterSessionSnapshot) {
	s := globalRouterSessionMemory
	s.mu.Lock()
	defer s.mu.Unlock()
	modelTurns := cloneIntMap(snapshot.ModelTurns)
	if modelTurns == nil {
		modelTurns = make(map[string]int)
	}
	s.sessions[snapshot.SessionID] = &routerSessionState{
		sessionID:                       snapshot.SessionID,
		userID:                          snapshot.UserID,
		currentModel:                    snapshot.CurrentModel,
		lastSeen:                        snapshot.LastSeen,
		turnCount:                       snapshot.TurnCount,
		switchCount:                     snapshot.SwitchCount,
		modelTurns:                      modelTurns,
		cumulativePrompt:                snapshot.CumulativePromptTokens,
		cumulativeCached:                snapshot.CumulativeCachedTokens,
		cumulativeCacheWrite:            snapshot.CumulativeCacheWriteTokens,
		cumulativeEstimatedCached:       snapshot.CumulativeEstimatedCachedTokens,
		cumulativeCompletion:            snapshot.CumulativeCompletionTokens,
		cumulativeCost:                  snapshot.CumulativeCost,
		cumulativeEstimatedCacheSavings: snapshot.CumulativeEstimatedCacheSavings,
		activeToolLoop:                  snapshot.ActiveToolLoop,
		lastDecisionName:                snapshot.LastDecisionName,
		lastDecisionReason:              snapshot.LastDecisionReason,
		lastCacheAccountingSource:       snapshot.LastCacheAccountingSource,
		lastPolicy:                      clonePolicyMap(snapshot.LastPolicy),
	}
}

func (s *redisRouterSessionStore) Load(sessionID string) (RouterSessionSnapshot, bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	payload, err := s.client.Get(ctx, s.keyPrefix+sessionID).Bytes()
	if errors.Is(err, redis.Nil) {
		return RouterSessionSnapshot{}, false, nil
	}
	if err != nil {
		return RouterSessionSnapshot{}, false, err
	}
	var snapshot RouterSessionSnapshot
	if err := json.Unmarshal(payload, &snapshot); err != nil {
		return RouterSessionSnapshot{}, false, err
	}
	return snapshot, true, nil
}

func (s *redisRouterSessionStore) Save(snapshot RouterSessionSnapshot, ttl time.Duration) error {
	payload, err := json.Marshal(snapshot)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	if s.ttl > 0 {
		ttl = s.ttl
	}
	return s.client.Set(ctx, s.keyPrefix+snapshot.SessionID, payload, ttl).Err()
}

func (s *redisRouterSessionStore) Close() error {
	return s.client.Close()
}
