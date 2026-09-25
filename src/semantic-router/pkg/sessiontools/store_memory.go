package sessiontools

import (
	"context"
	"hash/fnv"
	"sort"
	"sync"
	"sync/atomic"
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

// memoryEntry is the shard-internal record. Never returned to a caller
// directly — every read clones State first (see MemoryStore.Load).
type memoryEntry struct {
	state      State
	quota      QuotaKey
	generation uint64
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
	// nextRevision is shared by every successful write so a recreated key
	// cannot reuse a revision held by a stale writer from an earlier
	// incarnation. Zero remains reserved for CompareAndSwap's create case.
	nextRevision atomic.Uint64
	// lifecycleMu keeps Close linearizable with every operation: an
	// operation that acquired the read lock completes before Close returns,
	// while one arriving afterwards observes ErrStoreClosed.
	lifecycleMu sync.RWMutex
	closed      bool

	// admissionMu guards every field below and serializes admission
	// (new-key creation, which may trigger eviction) relative to other
	// admissions. An existing key's CompareAndSwap/Load never takes this
	// lock — only that key's shard lock — so reuse of already-admitted
	// sessions is not serialized by store-wide bookkeeping. Lock ordering
	// is always admissionMu outer, a shard's mu inner; a shard's mu is
	// never held while acquiring admissionMu.
	admissionMu    sync.Mutex
	keyQuota       map[string]QuotaKey
	quotaMembers   map[QuotaKey]map[string]struct{}
	totalCount     int
	nextGeneration uint64
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

// Load implements Store. See store.go's Store doc comment for why this
// returns (VersionedState, error) rather than the originally sketched
// (VersionedState, bool, error).
func (s *MemoryStore) Load(ctx context.Context, key string) (VersionedState, error) {
	result, _, err := s.LoadWithMetadata(ctx, key)
	return result, err
}

// LoadWithMetadata is the optional expiry-aware Load extension used by the
// manager to emit an explicit expiry receipt without changing Store's narrow
// compatibility contract.
func (s *MemoryStore) LoadWithMetadata(_ context.Context, key string) (VersionedState, LoadMetadata, error) {
	s.lifecycleMu.RLock()
	defer s.lifecycleMu.RUnlock()
	if s.closed {
		return VersionedState{}, LoadMetadata{}, ErrStoreClosed
	}
	shard := s.shardFor(key)
	now := s.clock()

	shard.mu.Lock()
	entry, ok := shard.entries[key]
	var observedRevision uint64
	var observedGeneration uint64
	var observedExpiry time.Time
	expired := false
	if ok {
		observedRevision = entry.state.Revision
		observedGeneration = entry.generation
		observedExpiry = entry.state.ExpiresAt
		expired = !observedExpiry.After(now)
	}
	if ok && !expired {
		// Sliding idle expiry: a successful Load refreshes the deadline,
		// same as a successful CompareAndSwap.
		entry.state.LastSeenAt = now
		entry.state.ExpiresAt = now.Add(s.ttl)
		result := entry.state.Clone()
		shard.mu.Unlock()
		return VersionedState{State: result, Found: true}, LoadMetadata{
			ObservedRevision:   observedRevision,
			ObservedGeneration: observedGeneration,
		}, nil
	}
	shard.mu.Unlock()

	if expired {
		// Prune it now rather than waiting for an admission-time sweep to
		// find it. Bookkeeping removal needs admissionMu, which the
		// lock-ordering rule (admissionMu outer, a shard's mu inner)
		// requires not be acquired while holding shard.mu — safe here
		// since shard.mu was already released above. Deletion is
		// conditional on the exact entry observed above (identity + TTL
		// re-check under admissionMu): a concurrent CompareAndSwap(0) may
		// have already deleted-and-recreated this key by the time this
		// runs, and this stale observation must not delete that fresh,
		// live entry (an ABA race — see deleteIfExpiredLocked).
		s.deleteIfExpiredLocked(key, observedRevision, observedGeneration, observedExpiry)
	}
	return VersionedState{}, LoadMetadata{
		Expired:            expired,
		ObservedRevision:   observedRevision,
		ObservedGeneration: observedGeneration,
	}, nil
}

// deleteIfExpiredLocked deletes key's entry only if it is still, right now,
// the exact same expired entry a caller (Load, or CompareAndSwap's create
// path) observed as expired: same generation, revision, and ExpiresAt, still
// expired as of the current clock reading. Without this identity check, a stale
// "this was expired" observation acted on after the fact could delete a
// brand-new, live entry that a concurrent CompareAndSwap(0) admitted at the
// same key in the meantime (classic ABA: same key, different underlying
// value). Acquires admissionMu itself — callers must not already hold a
// shard lock.
func (s *MemoryStore) deleteIfExpiredLocked(key string, observedRevision, observedGeneration uint64, observedExpiry time.Time) {
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()

	shard := s.shardFor(key)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	entry, ok := shard.entries[key]
	if !ok {
		return
	}
	if entry.state.Revision != observedRevision || entry.generation != observedGeneration {
		return
	}
	if !entry.state.ExpiresAt.Equal(observedExpiry) {
		return
	}
	if entry.state.ExpiresAt.After(s.clock()) {
		return
	}

	delete(shard.entries, key)
	s.removeBookkeepingForEntryLocked(key, entry.quota)
}

// CompareAndSwap implements Store. expectedRevision == 0 (create) and > 0
// (update an existing, live entry) are handled by different paths with
// different locking: an update only ever touches the key's own shard lock
// (the hot, contended-only-with-itself path); a create must serialize
// against every other concurrent create via admissionMu, since it may
// evict under quota/global caps and must not race the expired-readmission
// ABA hazard documented on compareAndSwapCreate/deleteIfExpiredLocked.
func (s *MemoryStore) CompareAndSwap(
	ctx context.Context,
	key string,
	expectedRevision uint64,
	next State,
	ttl time.Duration,
	quota QuotaKey,
) (bool, error) {
	_, applied, err := s.CompareAndSwapWithRevision(ctx, key, expectedRevision, next, ttl, quota)
	return applied, err
}

// CompareAndSwapWithRevision implements RevisionedCompareAndSwapStore.
func (s *MemoryStore) CompareAndSwapWithRevision(
	_ context.Context,
	key string,
	expectedRevision uint64,
	next State,
	ttl time.Duration,
	quota QuotaKey,
) (uint64, bool, error) {
	s.lifecycleMu.RLock()
	defer s.lifecycleMu.RUnlock()
	if s.closed {
		return 0, false, ErrStoreClosed
	}
	if ttl <= 0 {
		ttl = s.ttl
	}
	now := s.clock()

	if expectedRevision == 0 {
		return s.compareAndSwapCreate(key, next, ttl, quota, now)
	}

	// Update path: an existing, live entry only. A missing or expired key
	// is never eligible for an update — the caller must go through the
	// expectedRevision == 0 create path instead, which is the only path
	// that touches admission bookkeeping. No opportunistic cleanup of an
	// expired entry happens here: that stays admissionMu-gated (Load,
	// compareAndSwapCreate) to avoid the ABA hazard those functions guard
	// against.
	shard := s.shardFor(key)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	entry, exists := shard.entries[key]
	if !exists || !entry.state.ExpiresAt.After(now) {
		return 0, false, ErrRevisionMismatch
	}
	if entry.state.Revision != expectedRevision {
		return 0, false, ErrRevisionMismatch
	}

	stored := next.Clone()
	stored.LastSeenAt = now
	stored.ExpiresAt = now.Add(ttl)
	stored.Revision = s.nextRevision.Add(1)
	entry.state = stored
	return stored.Revision, true, nil
}

// compareAndSwapCreate handles CompareAndSwap's expectedRevision == 0
// (create) path. Holds admissionMu for its entire duration — including the
// expired-entry check, its removal, and admission of the fresh entry — so
// no other create for this key and no stale deleteIfExpiredLocked
// observation from a concurrent Load can interleave and delete a
// just-admitted live entry out from under this call (the ABA race this
// store's expired-cleanup paths must all guard against).
func (s *MemoryStore) compareAndSwapCreate(key string, next State, ttl time.Duration, quota QuotaKey, now time.Time) (uint64, bool, error) {
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()

	shard := s.shardFor(key)
	shard.mu.Lock()
	entry, exists := shard.entries[key]
	if exists {
		if entry.state.ExpiresAt.After(now) {
			// Still live: a create cannot replace a live entry.
			shard.mu.Unlock()
			return 0, false, ErrRevisionMismatch
		}
		// Expired: safe to reclaim here, unlike deleteIfExpiredLocked's
		// stale-observation case, because this whole function already
		// holds admissionMu — nothing else admission-gated can have
		// changed this entry since shard.mu was locked just above.
		delete(shard.entries, key)
		s.removeBookkeepingForEntryLocked(key, entry.quota)
	}
	shard.mu.Unlock()

	return s.admitNewLocked(key, next, ttl, quota, now)
}

// admitNewLocked creates a brand-new entry, evicting under quota/global
// caps first if necessary. Caller must already hold admissionMu for the
// call's whole duration (see compareAndSwapCreate) — this serializes
// concurrent creations against each other and against expired-cleanup;
// reuse of existing keys (the hot CompareAndSwap update path) never
// contends with this.
func (s *MemoryStore) admitNewLocked(key string, next State, ttl time.Duration, quota QuotaKey, now time.Time) (uint64, bool, error) {
	if _, already := s.keyQuota[key]; already {
		return 0, false, ErrRevisionMismatch
	}

	s.evictForIdentityLocked(quota)
	s.evictForCapacityLocked()

	stored := next.Clone()
	stored.LastSeenAt = now
	stored.ExpiresAt = now.Add(ttl)
	stored.Revision = s.nextRevision.Add(1)
	s.nextGeneration++
	generation := s.nextGeneration

	shard := s.shardFor(key)
	shard.mu.Lock()
	if _, already := shard.entries[key]; already {
		shard.mu.Unlock()
		return 0, false, ErrRevisionMismatch
	}
	shard.entries[key] = &memoryEntry{state: stored, quota: quota, generation: generation}
	shard.mu.Unlock()

	s.keyQuota[key] = quota
	if s.quotaMembers[quota] == nil {
		s.quotaMembers[quota] = make(map[string]struct{})
	}
	s.quotaMembers[quota][key] = struct{}{}
	s.totalCount++
	return stored.Revision, true, nil
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
	keys := make([]string, 0, len(members))
	for key := range members {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		seen, expired := s.peekLastSeenLocked(key, now)
		if expired {
			s.deleteEntryLocked(key)
			return // freed a slot without evicting a live session
		}
		if first || seen.Before(oldestSeen) || (seen.Equal(oldestSeen) && key < oldestKey) {
			oldestKey, oldestSeen, first = key, seen, false
		}
	}
	if !first {
		s.deleteEntryLocked(oldestKey)
	}
}

// evictForCapacityLocked evicts the least-recently-seen session store-wide if
// admitting one more would exceed max_sessions. The key list is sorted before
// inspection so map iteration order cannot change which tied session is
// evicted. Caller must hold admissionMu.
func (s *MemoryStore) evictForCapacityLocked() {
	if s.totalCount < s.maxSessions {
		return
	}
	var oldestKey string
	var oldestSeen time.Time
	first := true
	now := s.clock()
	keys := make([]string, 0, len(s.keyQuota))
	for key := range s.keyQuota {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		seen, expired := s.peekLastSeenLocked(key, now)
		if expired {
			s.deleteEntryLocked(key)
			return
		}
		if first || seen.Before(oldestSeen) || (seen.Equal(oldestSeen) && key < oldestKey) {
			oldestKey, oldestSeen, first = key, seen, false
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
	entry, existed := shard.entries[key]
	delete(shard.entries, key)
	shard.mu.Unlock()
	if !existed {
		return
	}
	s.removeBookkeepingForEntryLocked(key, entry.quota)
}

// removeBookkeepingForEntryLocked removes key's admission bookkeeping
// (keyQuota, quotaMembers, totalCount). fallback supplies the quota to
// clean up from quotaMembers when key is no longer present in keyQuota
// (defensive: callers pass the memoryEntry's own carried quota, so
// bookkeeping removal is correct even if keyQuota was already mutated by
// the time this runs). Caller must hold admissionMu.
func (s *MemoryStore) removeBookkeepingForEntryLocked(key string, fallback QuotaKey) {
	quota, ok := s.keyQuota[key]
	if ok {
		delete(s.keyQuota, key)
		s.totalCount--
	} else {
		quota = fallback
	}

	if members := s.quotaMembers[quota]; members != nil {
		delete(members, key)
		if len(members) == 0 {
			delete(s.quotaMembers, quota)
		}
	}
}

// removeFromBookkeeping acquires admissionMu itself; used only by Delete,
// which genuinely wants "whatever is currently at this key" gone
// regardless of what it is. Load's and CompareAndSwap's expired-cleanup
// paths must not use this: they use deleteIfExpiredLocked /
// compareAndSwapCreate instead, which validate the entry's identity (or
// hold admissionMu for the whole check-then-act sequence) before deleting,
// to avoid the ABA race an unconditional by-key delete would reintroduce.
func (s *MemoryStore) removeFromBookkeeping(key string) {
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	s.deleteEntryLocked(key)
}

// Delete implements Store.
func (s *MemoryStore) Delete(_ context.Context, key string) error {
	s.lifecycleMu.RLock()
	defer s.lifecycleMu.RUnlock()
	if s.closed {
		return ErrStoreClosed
	}
	s.removeFromBookkeeping(key)
	return nil
}

// DeleteIfToken implements ConditionalDeleteStore. A changed or missing entry
// is left untouched and reported as not applied, allowing a stale invalidation
// caller to continue with a fresh load/CAS cycle. Generation is checked in
// addition to revision so a delete-and-recreate ABA cycle cannot match.
func (s *MemoryStore) DeleteIfToken(_ context.Context, key string, token StateToken) (bool, error) {
	s.lifecycleMu.RLock()
	defer s.lifecycleMu.RUnlock()
	if s.closed {
		return false, ErrStoreClosed
	}

	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	shard := s.shardFor(key)
	shard.mu.Lock()
	entry, exists := shard.entries[key]
	if !exists || entry.state.Revision != token.Revision ||
		(token.Generation > 0 && entry.generation != token.Generation) {
		shard.mu.Unlock()
		return false, nil
	}
	delete(shard.entries, key)
	shard.mu.Unlock()
	s.removeBookkeepingForEntryLocked(key, entry.quota)
	return true, nil
}

// Close implements Store. Idempotent.
func (s *MemoryStore) Close() error {
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()
	s.closed = true
	return nil
}
