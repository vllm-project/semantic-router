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

	SaveIfVersion(
		snapshot RouterSessionSnapshot,
		expectedVersion uint64,
		ttl time.Duration,
	) (bool, error)

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

var errStaleRouterSessionVersion = errors.New("stale router session version")

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

func persistRouterSessionDecision(p SessionDecisionParams) {
	if p.SessionID == "" {
		return
	}

	store, release, acquired := acquireCurrentRouterSessionStateStore()
	if !acquired {
		return
	}
	defer release()

	snapshot, ok := GetRouterSessionSnapshot(p.SessionID, time.Now())
	if !ok {
		return
	}

	// First attempt: persist the local mutation.
	saved, err := store.SaveIfVersion(
		snapshot,
		snapshot.Version,
		routerMemoryTTL,
	)
	if err != nil || saved {
		if saved {
			updateRouterSessionVersion(p.SessionID, snapshot.Version+1)
		}
		return
	}

	// CAS failed because another replica advanced the shared state.
	latest, found, err := store.Load(p.SessionID)
	if err != nil || !found {
		return
	}

	if latest.SessionID != p.SessionID {
		return
	}

	// Reapply this operation to the latest shared state.
	rebased := latest
	applySessionDecisionSnapshot(&rebased, p)

	saved, err = store.SaveIfVersion(
		rebased,
		latest.Version,
		routerMemoryTTL,
	)
	if err != nil || !saved {
		return
	}

	updateRouterSessionVersion(p.SessionID, rebased.Version)
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
		version:                         snapshot.Version,
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

func (s *redisRouterSessionStore) SaveIfVersion(
	snapshot RouterSessionSnapshot,
	expectedVersion uint64,
	ttl time.Duration,
) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()

	if s.ttl > 0 {
		ttl = s.ttl
	}

	key := s.keyPrefix + snapshot.SessionID

	err := s.client.Watch(ctx, func(tx *redis.Tx) error {
		payload, err := tx.Get(ctx, key).Bytes()

		if errors.Is(err, redis.Nil) {
			// A missing key represents version 0.
			if expectedVersion != 0 {
				return errStaleRouterSessionVersion
			}
		} else if err != nil {
			return err
		} else {
			var current RouterSessionSnapshot
			if err := json.Unmarshal(payload, &current); err != nil {
				return err
			}

			if current.Version != expectedVersion {
				return errStaleRouterSessionVersion
			}
		}

		snapshot.Version = expectedVersion + 1

		newPayload, err := json.Marshal(snapshot)
		if err != nil {
			return err
		}

		_, err = tx.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
			pipe.Set(ctx, key, newPayload, ttl)
			return nil
		})
		return err
	}, key)

	if errors.Is(err, redis.TxFailedErr) ||
		errors.Is(err, errStaleRouterSessionVersion) {
		return false, nil
	}

	if err != nil {
		return false, err
	}

	return true, nil
}

func (s *redisRouterSessionStore) Close() error {
	return s.client.Close()
}

func persistRouterSessionUsage(p SessionUsageParams) {
	if p.SessionID == "" {
		return
	}

	store, release, acquired := acquireCurrentRouterSessionStateStore()
	if !acquired {
		return
	}
	defer release()

	snapshot, ok := GetRouterSessionSnapshot(p.SessionID, time.Now())
	if !ok {
		return
	}

	saved, err := store.SaveIfVersion(
		snapshot,
		snapshot.Version,
		routerMemoryTTL,
	)
	if err != nil || saved {
		if saved {
			updateRouterSessionVersion(p.SessionID, snapshot.Version+1)
		}
		return
	}

	// Another replica advanced the shared state.
	latest, found, err := store.Load(p.SessionID)
	if err != nil || !found {
		return
	}

	if latest.SessionID != p.SessionID {
		return
	}

	rebased := latest
	applySessionUsageSnapshot(&rebased, p)

	saved, err = store.SaveIfVersion(
		rebased,
		latest.Version,
		routerMemoryTTL,
	)
	if err != nil || !saved {
		return
	}

	updateRouterSessionVersion(p.SessionID, rebased.Version)
}
