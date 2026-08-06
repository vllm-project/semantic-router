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

var (
	routerSessionStoreMu sync.RWMutex
	routerSessionStore   RouterSessionStateStore
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

// SetRouterSessionStateStore swaps the optional shared store.
func SetRouterSessionStateStore(store RouterSessionStateStore) {
	routerSessionStoreMu.Lock()
	previous := routerSessionStore
	routerSessionStore = store
	routerSessionStoreMu.Unlock()
	if previous != nil && previous != store {
		_ = previous.Close()
	}
}

func currentRouterSessionStateStore() RouterSessionStateStore {
	routerSessionStoreMu.RLock()
	defer routerSessionStoreMu.RUnlock()
	return routerSessionStore
}

func persistRouterSessionState(sessionID string) {
	store := currentRouterSessionStateStore()
	if store == nil || sessionID == "" {
		return
	}
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
	store := currentRouterSessionStateStore()
	if store == nil {
		return RouterSessionSnapshot{}, false
	}
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
