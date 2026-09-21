package sessiontelemetry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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

// Version 2 binds persisted snapshots to RoutingSessionKey's escaped identity
// components. Unversioned snapshots used ambiguous raw IDs and cannot be reused.
const redisRouterSessionEncodingVersion = 2

type redisRouterSessionEnvelope struct {
	Version  int                   `json:"version"`
	Snapshot RouterSessionSnapshot `json:"snapshot"`
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
	// A store that can merge keeps concurrent writers' facts; the plain Save
	// path is the fallback for stores without that capability.
	if merger, ok := store.(RouterSessionStateMerger); ok {
		if err := merger.Merge(snapshot, routerMemoryTTL); err != nil {
			// A merge that cannot read the stored payload leaves the local
			// snapshot unpersisted; without this the loss is silent.
			logging.ComponentWarnEvent("router", "session_state_merge_failed", map[string]interface{}{
				"session_id": sessionID,
				"error":      err.Error(),
			})
		}
		return
	}
	_ = store.Save(snapshot, routerMemoryTTL)
}

func loadSharedRouterSessionSnapshotMode(sessionID string, now time.Time, hydrate bool) (RouterSessionSnapshot, bool) {
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
	snapshot.CurrentCandidate = cloneSessionCandidate(snapshot.CurrentCandidate)
	snapshot.ModelTurns = cloneIntMap(snapshot.ModelTurns)
	snapshot.LastPolicy = clonePolicyMap(snapshot.LastPolicy)
	snapshot.RecentOutcomes = cloneTurnOutcomes(snapshot.RecentOutcomes)
	snapshot.SwitchTimestamps = cloneInt64Slice(snapshot.SwitchTimestamps)
	if hydrate {
		hydrateRouterSessionSnapshot(snapshot)
	}
	return snapshot, true
}

func hydrateRouterSessionSnapshot(snapshot RouterSessionSnapshot) {
	s := globalRouterSessionMemory
	s.mu.Lock()
	defer s.mu.Unlock()
	if st := s.sessions[snapshot.SessionID]; st != nil &&
		(st.outcomeWindowSize > 0 || snapshot.OutcomeWindowSize > 0) {
		return
	}
	modelTurns := cloneIntMap(snapshot.ModelTurns)
	if modelTurns == nil {
		modelTurns = make(map[string]int)
	}
	s.sessions[snapshot.SessionID] = &routerSessionState{
		sessionID:                       snapshot.SessionID,
		userID:                          snapshot.UserID,
		currentModel:                    snapshot.CurrentModel,
		currentCandidate:                cloneSessionCandidate(snapshot.CurrentCandidate),
		lastSeen:                        snapshot.LastSeen,
		lastSwitchAt:                    snapshot.LastSwitchAt,
		switchTimestamps:                cloneInt64Slice(snapshot.SwitchTimestamps),
		outcomeWindowSize:               snapshot.OutcomeWindowSize,
		outcomeWindowTTL:                time.Duration(snapshot.OutcomeWindowTTLSeconds) * time.Second,
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
		recentOutcomes:                  cloneTurnOutcomes(snapshot.RecentOutcomes),
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
	return decodeRedisRouterSessionSnapshot(payload, sessionID)
}

func decodeRedisRouterSessionSnapshot(payload []byte, sessionID string) (RouterSessionSnapshot, bool, error) {
	var envelope struct {
		Version  int             `json:"version"`
		Snapshot json.RawMessage `json:"snapshot"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return RouterSessionSnapshot{}, false, err
	}
	if envelope.Version != redisRouterSessionEncodingVersion || len(envelope.Snapshot) == 0 {
		return RouterSessionSnapshot{}, false, nil
	}
	var snapshot RouterSessionSnapshot
	if err := json.Unmarshal(envelope.Snapshot, &snapshot); err != nil {
		return RouterSessionSnapshot{}, false, err
	}
	if sessionID == "" || snapshot.SessionID != sessionID {
		return RouterSessionSnapshot{}, false, nil
	}
	return snapshot, true, nil
}

// encodeRedisRouterSessionSnapshot is the write side of the store codec. Every
// value the store persisches goes through it, so the loader and the merge read
// what Save wrote.
func encodeRedisRouterSessionSnapshot(snapshot RouterSessionSnapshot) ([]byte, error) {
	return json.Marshal(redisRouterSessionEnvelope{
		Version:  redisRouterSessionEncodingVersion,
		Snapshot: snapshot,
	})
}

func (s *redisRouterSessionStore) Save(snapshot RouterSessionSnapshot, ttl time.Duration) error {
	payload, err := encodeRedisRouterSessionSnapshot(snapshot)
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

// redisMergeAttempts bounds the optimistic-concurrency retries.
const redisMergeAttempts = 4

// Merge folds the local snapshot into the stored one under a compare-and-swap,
// so two replicas that loaded the same session cannot overwrite each other.
func (s *redisRouterSessionStore) Merge(local RouterSessionSnapshot, ttl time.Duration) error {
	if local.SessionID == "" {
		return nil
	}
	if s.ttl > 0 {
		ttl = s.ttl
	}
	key := s.keyPrefix + local.SessionID

	var lastErr error
	for attempt := 0; attempt < redisMergeAttempts; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
		err := s.client.Watch(ctx, func(tx *redis.Tx) error {
			merged := local
			stored, err := tx.Get(ctx, key).Bytes()
			switch {
			case errors.Is(err, redis.Nil):
			case err != nil:
				return err
			default:
				merged, err = mergeStoredSnapshot(stored, local)
				if err != nil {
					return err
				}
			}
			payload, err := encodeRedisRouterSessionSnapshot(merged)
			if err != nil {
				return err
			}
			_, err = tx.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
				pipe.Set(ctx, key, payload, ttl)
				return nil
			})
			return err
		}, key)
		cancel()

		if errors.Is(err, redis.TxFailedErr) {
			lastErr = err
			continue
		}
		return err
	}
	return lastErr
}

func (s *redisRouterSessionStore) Close() error {
	return s.client.Close()
}
