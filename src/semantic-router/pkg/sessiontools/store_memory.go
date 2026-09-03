package sessiontools

import (
	"context"
	"hash/fnv"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// memoryStoreShardCount is a fixed shard count for the per-key lock split.
// Sessions hash-distribute across shards, so the hot read/write path (an
// existing key's CompareAndSwap or a Load) only ever contends with other
// operations landing on the same shard, not with the whole store. Not
// configurable: this is an implementation-scale knob, not a deployment
// concern (unlike max_sessions/max_sessions_per_identity, which are).
const memoryStoreShardCount = 64

// memoryStoreEvictionSampleSize bounds how many entries a global-eviction
// scan inspects when the store is at its max_sessions cap. Matches
// pkg/sessiontelemetry/router_memory.go's evictOldestLocked: a full scan
// across up to 100000 sessions (config.ToolSessionStoreMaxMaxSessions) on
// every admission would be an unbounded hot-path cost for a fallback-tier
// local store; a bounded sample gives approximate LRU at O(1) amortized
// cost instead. Per-identity eviction stays exact — that bucket is bounded
// by max_sessions_per_identity itself, typically far smaller.
const memoryStoreEvictionSampleSize = 32

// memoryEntry is the shard-internal record. Never returned to a caller
// directly — every read clones State first (see MemoryStore.Load).
type memoryEntry struct {
	state State
	quota QuotaKey
}

type memoryShard struct {
	mu      sync.Mutex
	entries map[string]*memoryEntry
}

// MemoryStore is the local (non-shared) Store implementation for
// session-scoped sticky tool-set selection. Restart or router-generation
// reload loses all state; the next request performs normal selection and
// reseeds. That is documented, intended behavior (see
// config.ToolSessionStoreBackendLocal), not a defect.
type MemoryStore struct {
	shards [memoryStoreShardCount]*memoryShard

	clock                 func() time.Time
	ttl                   time.Duration
	maxSessions           int
	maxSessionsByIdentity int

	// admissionMu guards every field below and serializes admission
	// (new-key creation, which may trigger eviction) relative to other
	// admissions. An existing key's CompareAndSwap/Load never takes this
	// lock — only that key's shard lock — so reuse of already-admitted
	// sessions is not serialized by store-wide bookkeeping. Lock ordering
	// is always admissionMu outer, a shard's mu inner; a shard's mu is
	// never held while acquiring admissionMu.
	admissionMu  sync.Mutex
	keyQuota     map[string]QuotaKey
	quotaMembers map[QuotaKey]map[string]struct{}
	totalCount   int
	closed       bool
}

// NewMemoryStore constructs a MemoryStore from the resolved
// config.ToolSessionStoreConfig. clock defaults to time.Now when nil —
// tests inject a synthetic clock to exercise TTL expiry deterministically.
func NewMemoryStore(cfg config.ToolSessionStoreConfig, clock func() time.Time) *MemoryStore {
	if clock == nil {
		clock = time.Now
	}
	s := &MemoryStore{
		clock:                 clock,
		ttl:                   time.Duration(cfg.EffectiveTTLSeconds()) * time.Second,
		maxSessions:           cfg.EffectiveMaxSessions(),
		maxSessionsByIdentity: cfg.EffectiveMaxSessionsByIdentity(),
		keyQuota:              make(map[string]QuotaKey),
		quotaMembers:          make(map[QuotaKey]map[string]struct{}),
	}
	for i := range s.shards {
		s.shards[i] = &memoryShard{entries: make(map[string]*memoryEntry)}
	}
	return s
}

func (s *MemoryStore) shardFor(key string) *memoryShard {
	h := fnv.New64a()
	_, _ = h.Write([]byte(key)) // fnv.Write never returns an error
	return s.shards[h.Sum64()%memoryStoreShardCount]
}

func (s *MemoryStore) isClosed() bool {
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	return s.closed
}

// Load implements Store. See store.go's Store doc comment for why this
// returns (VersionedState, error) rather than the originally sketched
// (VersionedState, bool, error).
func (s *MemoryStore) Load(_ context.Context, key string) (VersionedState, error) {
	if s.isClosed() {
		return VersionedState{}, ErrStoreClosed
	}
	shard := s.shardFor(key)
	now := s.clock()

	shard.mu.Lock()
	entry, ok := shard.entries[key]
	expired := ok && !entry.state.ExpiresAt.After(now)
	if ok && !expired {
		// Sliding idle expiry: a successful Load refreshes the deadline,
		// same as a successful CompareAndSwap.
		entry.state.LastSeenAt = now
		entry.state.ExpiresAt = now.Add(s.ttl)
		result := entry.state.Clone()
		shard.mu.Unlock()
		return VersionedState{State: result, Found: true}, nil
	}
	shard.mu.Unlock()

	if expired {
		// Prune it now rather than waiting for an admission-time sweep to
		// find it. Bookkeeping removal needs admissionMu, which the
		// lock-ordering rule (admissionMu outer, a shard's mu inner)
		// requires not be acquired while holding shard.mu — safe here
		// since shard.mu was already released above.
		s.removeFromBookkeeping(key)
	}
	return VersionedState{}, nil
}

// CompareAndSwap implements Store.
func (s *MemoryStore) CompareAndSwap(
	_ context.Context,
	key string,
	expectedRevision uint64,
	next State,
	ttl time.Duration,
	quota QuotaKey,
) (bool, error) {
	if s.isClosed() {
		return false, ErrStoreClosed
	}
	if ttl <= 0 {
		ttl = s.ttl
	}
	shard := s.shardFor(key)

	shard.mu.Lock()
	entry, exists := shard.entries[key]
	now := s.clock()
	expired := exists && !entry.state.ExpiresAt.After(now)

	if exists && !expired {
		// Update path: no admission bookkeeping change, shard lock only.
		defer shard.mu.Unlock()
		if entry.state.Revision != expectedRevision {
			return false, ErrRevisionMismatch
		}
		stored := next.Clone()
		stored.LastSeenAt = now
		stored.ExpiresAt = now.Add(ttl)
		stored.Revision = entry.state.Revision + 1
		entry.state = stored
		return true, nil
	}
	shard.mu.Unlock()

	// Creation path (key absent, or present but expired and therefore
	// treated as absent): expectedRevision must be 0.
	if expectedRevision != 0 {
		return false, ErrRevisionMismatch
	}
	if expired {
		s.removeFromBookkeeping(key)
	}
	return s.admitNew(key, next, ttl, quota, now)
}

// admitNew creates a brand-new entry, evicting under quota/global caps
// first if necessary. Held under admissionMu for its whole duration,
// serializing concurrent creations against each other — reuse of existing
// keys (the hot path) never contends with this.
func (s *MemoryStore) admitNew(key string, next State, ttl time.Duration, quota QuotaKey, now time.Time) (bool, error) {
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	if s.closed {
		return false, ErrStoreClosed
	}
	// Another goroutine may have created this key between our shard-level
	// check and acquiring admissionMu; re-check under the lock that
	// actually serializes creation.
	if _, already := s.keyQuota[key]; already {
		return false, ErrRevisionMismatch
	}

	s.evictForIdentityLocked(quota)
	s.evictForCapacityLocked()

	stored := next.Clone()
	stored.LastSeenAt = now
	stored.ExpiresAt = now.Add(ttl)
	stored.Revision = 1

	shard := s.shardFor(key)
	shard.mu.Lock()
	shard.entries[key] = &memoryEntry{state: stored, quota: quota}
	shard.mu.Unlock()

	s.keyQuota[key] = quota
	if s.quotaMembers[quota] == nil {
		s.quotaMembers[quota] = make(map[string]struct{})
	}
	s.quotaMembers[quota][key] = struct{}{}
	s.totalCount++
	return true, nil
}

// evictForIdentityLocked evicts the least-recently-seen member of quota's
// bucket if admitting one more member would exceed
// max_sessions_per_identity. Exact (not sampled): a quota bucket is
// bounded by max_sessions_per_identity itself, typically far smaller than
// the global cap. Caller must hold admissionMu.
func (s *MemoryStore) evictForIdentityLocked(quota QuotaKey) {
	members := s.quotaMembers[quota]
	if len(members) < s.maxSessionsByIdentity {
		return
	}
	var oldestKey string
	var oldestSeen time.Time
	first := true
	now := s.clock()
	for key := range members {
		seen, expired := s.peekLastSeenLocked(key, now)
		if expired {
			s.deleteEntryLocked(key)
			return // freed a slot without evicting a live session
		}
		if first || seen.Before(oldestSeen) {
			oldestKey, oldestSeen, first = key, seen, false
		}
	}
	if !first {
		s.deleteEntryLocked(oldestKey)
	}
}

// evictForCapacityLocked evicts an approximately-least-recently-seen
// session store-wide if admitting one more would exceed max_sessions.
// Sampled, not exact — see memoryStoreEvictionSampleSize. Caller must hold
// admissionMu.
func (s *MemoryStore) evictForCapacityLocked() {
	if s.totalCount < s.maxSessions {
		return
	}
	var oldestKey string
	var oldestSeen time.Time
	first := true
	sampled := 0
	now := s.clock()
	for key := range s.keyQuota {
		seen, expired := s.peekLastSeenLocked(key, now)
		if expired {
			s.deleteEntryLocked(key)
			return
		}
		if first || seen.Before(oldestSeen) {
			oldestKey, oldestSeen, first = key, seen, false
		}
		sampled++
		if sampled >= memoryStoreEvictionSampleSize {
			break
		}
	}
	if !first {
		s.deleteEntryLocked(oldestKey)
	}
}

// peekLastSeenLocked reads a key's current LastSeenAt and whether it is
// expired as of now. Caller must hold admissionMu; briefly takes the
// key's shard lock to read it safely (an existing key's CompareAndSwap can
// run concurrently, holding only the shard lock).
func (s *MemoryStore) peekLastSeenLocked(key string, now time.Time) (seen time.Time, expired bool) {
	shard := s.shardFor(key)
	shard.mu.Lock()
	defer shard.mu.Unlock()
	entry, ok := shard.entries[key]
	if !ok {
		return time.Time{}, true
	}
	return entry.state.LastSeenAt, !entry.state.ExpiresAt.After(now)
}

// deleteEntryLocked removes key from both its shard and admission
// bookkeeping. Caller must hold admissionMu.
func (s *MemoryStore) deleteEntryLocked(key string) {
	shard := s.shardFor(key)
	shard.mu.Lock()
	_, existed := shard.entries[key]
	delete(shard.entries, key)
	shard.mu.Unlock()
	if !existed {
		return
	}
	if quota, ok := s.keyQuota[key]; ok {
		delete(s.keyQuota, key)
		if members := s.quotaMembers[quota]; members != nil {
			delete(members, key)
			if len(members) == 0 {
				delete(s.quotaMembers, quota)
			}
		}
		s.totalCount--
	}
}

// removeFromBookkeeping acquires admissionMu itself; used by callers
// (Load, CompareAndSwap's creation path) that must not already hold a
// shard lock when calling it, per this store's lock-ordering rule.
func (s *MemoryStore) removeFromBookkeeping(key string) {
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	s.deleteEntryLocked(key)
}

// Delete implements Store.
func (s *MemoryStore) Delete(_ context.Context, key string) error {
	if s.isClosed() {
		return ErrStoreClosed
	}
	s.removeFromBookkeeping(key)
	return nil
}

// Close implements Store. Idempotent.
func (s *MemoryStore) Close() error {
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	s.closed = true
	return nil
}
