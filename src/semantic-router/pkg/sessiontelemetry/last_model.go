package sessiontelemetry

import (
	"sync"
	"time"
)

// maxLastModelSessions caps the number of sessions tracked by the last-model
// store to bound memory. It is a var (not a const) so tests can exercise the
// eviction path without inserting tens of thousands of entries.
var maxLastModelSessions = 50_000

// evictionSampleSize bounds evictOldestLocked to an approximate (sampled) LRU:
// rather than scanning the whole map for the true least-recently-seen entry
// (O(N) under the write lock — which bites exactly at the cap this guards, since
// once the map is full every new session would trigger a full scan), we sample
// this many entries (Go map iteration order is randomized) and evict the oldest
// of the sample. O(1), the same approach Redis uses for approximate LRU.
// Intentionally approximate: bounding the map is what matters, evicting the
// exact oldest does not. Shared by the evictOldestLocked of all four session
// stores in this package.
const evictionSampleSize = 10

type lastModelState struct {
	model    string
	lastSeen time.Time
}

// lastModelStore remembers the most recent model used per session so the next
// turn can populate the previous-model signal that the model-switch gate needs.
// It is in-memory and per-replica (lost on restart), matching the existing
// telemetry and transition stores; durable cross-replica persistence is tracked
// as follow-up debt.
type lastModelStore struct {
	mu       sync.Mutex
	sessions map[string]*lastModelState
	nowFn    func() time.Time
}

var globalLastModelStore = &lastModelStore{
	sessions: make(map[string]*lastModelState),
	nowFn:    time.Now,
}

// RecordLastModel stores model as the most recent model for sessionID. Empty
// sessionID or model are ignored so the store never grows on unresolvable
// sessions. Entries older than ttl are evicted, and a size cap bounds total
// growth.
func RecordLastModel(sessionID, model string) {
	if sessionID == "" || model == "" {
		return
	}
	s := globalLastModelStore
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.nowFn()
	s.evictExpiredLocked(now)
	st := s.sessions[sessionID]
	if st == nil {
		if len(s.sessions) >= maxLastModelSessions {
			s.evictOldestLocked()
		}
		st = &lastModelState{}
		s.sessions[sessionID] = st
	}
	st.model = model
	st.lastSeen = now
}

// GetLastModel returns the most recent model recorded for sessionID and whether
// a non-expired entry exists.
func GetLastModel(sessionID string) (string, bool) {
	model, _, ok := GetLastModelInfo(sessionID, time.Time{})
	return model, ok
}

// GetLastModelInfo returns the most recent model and idle age for sessionID.
// Passing a zero now uses the store clock.
func GetLastModelInfo(sessionID string, now time.Time) (string, time.Duration, bool) {
	if sessionID == "" {
		return "", 0, false
	}
	s := globalLastModelStore
	s.mu.Lock()
	defer s.mu.Unlock()
	st := s.sessions[sessionID]
	if st == nil {
		return "", 0, false
	}
	if now.IsZero() {
		now = s.nowFn()
	}
	idleFor := now.Sub(st.lastSeen)
	if idleFor < 0 {
		idleFor = 0
	}
	if idleFor > ttl {
		delete(s.sessions, sessionID)
		return "", 0, false
	}
	return st.model, idleFor, true
}

func (s *lastModelStore) evictExpiredLocked(t time.Time) {
	for k, v := range s.sessions {
		if t.Sub(v.lastSeen) > ttl {
			delete(s.sessions, k)
		}
	}
}

// evictOldestLocked evicts the oldest entry among a bounded random sample
// (approximate LRU — see evictionSampleSize). It is a best-effort safety valve
// for the size cap when TTL eviction did not free room. Callers must hold s.mu.
func (s *lastModelStore) evictOldestLocked() {
	var oldestKey string
	var oldestSeen time.Time
	sampled := 0
	for k, v := range s.sessions {
		if sampled == 0 || v.lastSeen.Before(oldestSeen) {
			oldestKey, oldestSeen = k, v.lastSeen
		}
		sampled++
		if sampled >= evictionSampleSize {
			break
		}
	}
	if sampled > 0 {
		delete(s.sessions, oldestKey)
	}
}

// ResetLastModelForTesting clears the in-memory last-model store (tests only).
func ResetLastModelForTesting() {
	s := globalLastModelStore
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions = make(map[string]*lastModelState)
}

// setLastModelNowForTesting overrides the store clock for deterministic TTL
// tests. Passing nil restores time.Now.
func setLastModelNowForTesting(fn func() time.Time) {
	s := globalLastModelStore
	s.mu.Lock()
	defer s.mu.Unlock()
	if fn == nil {
		fn = time.Now
	}
	s.nowFn = fn
}

// lastModelSessionCount returns the number of tracked sessions (tests only).
func lastModelSessionCount() int {
	s := globalLastModelStore
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.sessions)
}
