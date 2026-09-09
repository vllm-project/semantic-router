package sessiontelemetry

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
)

const routerMemoryTTL = 24 * time.Hour

// maxRouterSessions caps the number of sessions tracked by the router-owned
// memory store so memory stays bounded under high session cardinality (session
// IDs are derived from message content, so distinct conversations create
// distinct entries). Mirrors last_model.go's maxLastModelSessions. It is a var
// (not a const) so tests can exercise the eviction path without inserting tens
// of thousands of entries.
var maxRouterSessions = 50_000

// RouterSessionSnapshot is the router-owned, model-independent memory for a
// session. It is intentionally about routing state, not prompt-visible user
// memory.
type RouterSessionSnapshot struct {
	SessionID string
	UserID    string

	CurrentModel string
	LastSeen     time.Time
	IdleFor      time.Duration
	// LastSwitchAt is the time of the most recent model change; zero when the
	// session has never switched.
	LastSwitchAt time.Time `json:"last_switch_at,omitempty"`
	// SwitchTimestamps are the recent model-change times in unix millis,
	// pruned by the session's evidence-window TTL. The gate counts switches
	// inside its configured window from these.
	SwitchTimestamps []int64 `json:"switch_timestamps,omitempty"`

	TurnCount   int
	SwitchCount int
	ModelTurns  map[string]int

	CumulativePromptTokens          int64
	CumulativeCachedTokens          int64
	CumulativeCacheWriteTokens      int64
	CumulativeEstimatedCachedTokens int64
	CumulativeCompletionTokens      int64
	CumulativeCost                  float64
	CumulativeEstimatedCacheSavings float64

	ActiveToolLoop            bool
	LastDecisionName          string
	LastDecisionReason        string
	LastCacheAccountingSource string
	LastPolicy                map[string]interface{}

	RecentOutcomes []TurnOutcome `json:"recent_outcomes,omitempty"`
}

// SessionDecisionParams records the pre-dispatch policy result for one session
// turn. Usage and response-side costs are attached later by RecordSessionUsage.
type SessionDecisionParams struct {
	SessionID      string
	UserID         string
	PreviousModel  string
	SelectedModel  string
	DecisionName   string
	TurnIndex      int
	ActiveToolLoop bool
	Policy         map[string]interface{}
	Timestamp      time.Time
}

// SessionUsageParams records response-side usage into router-owned session
// memory. Costs are in the pricing currency selected by the router config.
type SessionUsageParams struct {
	SessionID                   string
	Model                       string
	PromptTokens                int
	CachedPromptTokens          int
	CacheWriteTokens            int
	EstimatedCachedPromptTokens int
	CompletionTokens            int
	Cost                        float64
	EstimatedCacheSavings       float64
	CacheAccountingSource       string
	Timestamp                   time.Time
}

type routerSessionState struct {
	sessionID string
	userID    string

	currentModel string
	lastSeen     time.Time
	lastSwitchAt time.Time
	// switchTimestamps mirrors LastSwitchAt as a bounded series for the
	// oscillation guard's window-scoped count.
	switchTimestamps []int64

	turnCount   int
	switchCount int
	modelTurns  map[string]int

	cumulativePrompt                int64
	cumulativeCached                int64
	cumulativeCacheWrite            int64
	cumulativeEstimatedCached       int64
	cumulativeCompletion            int64
	cumulativeCost                  float64
	cumulativeEstimatedCacheSavings float64

	activeToolLoop            bool
	lastDecisionName          string
	lastDecisionReason        string
	lastCacheAccountingSource string
	lastPolicy                map[string]interface{}

	recentOutcomes []TurnOutcome
	// Evidence window policy for this session, set from the active progress
	// gate config. Process-local: the gate re-applies it on the request path.
	outcomeWindowSize int
	outcomeWindowTTL  time.Duration
}

type routerSessionMemoryStore struct {
	mu       sync.Mutex
	sessions map[string]*routerSessionState
	nowFn    func() time.Time
}

var globalRouterSessionMemory = &routerSessionMemoryStore{
	sessions: make(map[string]*routerSessionState),
	nowFn:    time.Now,
}

// RecordSessionDecision updates router-owned session memory from the policy
// decision made before dispatching the request upstream.
func RecordSessionDecision(p SessionDecisionParams) {
	if p.SessionID == "" || p.SelectedModel == "" {
		return
	}
	RecordLastModel(p.SessionID, p.SelectedModel)

	s := globalRouterSessionMemory
	defer persistRouterSessionState(p.SessionID)
	s.mu.Lock()
	defer s.mu.Unlock()

	now := p.Timestamp
	if now.IsZero() {
		now = s.nowFn()
	}
	s.evictExpiredLocked(now)
	st := s.sessionLocked(p.SessionID)
	if p.UserID != "" {
		st.userID = p.UserID
	}
	previous := st.currentModel
	if previous == "" {
		previous = p.PreviousModel
	}
	if previous != "" && previous != p.SelectedModel {
		st.switchCount++
		st.lastSwitchAt = now
		_, ttl := st.windowPolicy()
		st.switchTimestamps = pruneSwitchTimestamps(append(st.switchTimestamps, now.UnixMilli()), ttl, now)
	}
	st.currentModel = p.SelectedModel
	st.lastSeen = now
	if p.TurnIndex+1 > st.turnCount {
		st.turnCount = p.TurnIndex + 1
	} else {
		st.turnCount++
	}
	st.modelTurns[p.SelectedModel]++
	st.activeToolLoop = p.ActiveToolLoop
	st.lastDecisionName = p.DecisionName
	if p.Policy != nil {
		st.lastPolicy = clonePolicyMap(p.Policy)
		st.lastDecisionReason = policyDecisionReason(p.Policy)
	}
}

// RecordSessionUsage attaches response usage and cost to router-owned session
// memory. It does not create a model checkout when no prior decision exists.
func RecordSessionUsage(p SessionUsageParams) {
	if p.SessionID == "" || p.Model == "" {
		return
	}
	s := globalRouterSessionMemory
	defer persistRouterSessionState(p.SessionID)
	s.mu.Lock()
	defer s.mu.Unlock()

	now := p.Timestamp
	if now.IsZero() {
		now = s.nowFn()
	}
	s.evictExpiredLocked(now)
	st := s.sessionLocked(p.SessionID)
	st.currentModel = p.Model
	st.lastSeen = now
	usage := modelpricing.Normalize(modelpricing.Usage{
		PromptTokens:      p.PromptTokens,
		CachedInputTokens: p.CachedPromptTokens,
		CacheWriteTokens:  p.CacheWriteTokens,
		CompletionTokens:  p.CompletionTokens,
	})
	st.cumulativePrompt += int64(usage.PromptTokens)
	st.cumulativeCached += int64(usage.CachedInputTokens)
	st.cumulativeCacheWrite += int64(usage.CacheWriteTokens)
	st.cumulativeEstimatedCached += int64(clampCachedPromptTokens(p.PromptTokens, p.EstimatedCachedPromptTokens))
	st.cumulativeCompletion += int64(usage.CompletionTokens)
	st.cumulativeCost += p.Cost
	if p.EstimatedCacheSavings > 0 {
		st.cumulativeEstimatedCacheSavings += p.EstimatedCacheSavings
	}
	if p.CacheAccountingSource != "" {
		st.lastCacheAccountingSource = p.CacheAccountingSource
	}
}

// GetRouterSessionSnapshot returns a clone of the router-owned session memory.
func GetRouterSessionSnapshot(sessionID string, now time.Time) (RouterSessionSnapshot, bool) {
	if sessionID == "" {
		return RouterSessionSnapshot{}, false
	}
	s := globalRouterSessionMemory
	s.mu.Lock()
	st := s.sessions[sessionID]
	if st == nil {
		s.mu.Unlock()
		return loadSharedRouterSessionSnapshot(sessionID, now)
	}
	if now.IsZero() {
		now = s.nowFn()
	}
	idleFor := now.Sub(st.lastSeen)
	if idleFor < 0 {
		idleFor = 0
	}
	if idleFor > routerMemoryTTL {
		delete(s.sessions, sessionID)
		s.mu.Unlock()
		return RouterSessionSnapshot{}, false
	}
	snapshot := RouterSessionSnapshot{
		SessionID:                       st.sessionID,
		UserID:                          st.userID,
		CurrentModel:                    st.currentModel,
		LastSeen:                        st.lastSeen,
		LastSwitchAt:                    st.lastSwitchAt,
		SwitchTimestamps:                cloneInt64Slice(st.switchTimestamps),
		IdleFor:                         idleFor,
		TurnCount:                       st.turnCount,
		SwitchCount:                     st.switchCount,
		ModelTurns:                      cloneIntMap(st.modelTurns),
		CumulativePromptTokens:          st.cumulativePrompt,
		CumulativeCachedTokens:          st.cumulativeCached,
		CumulativeCacheWriteTokens:      st.cumulativeCacheWrite,
		CumulativeEstimatedCachedTokens: st.cumulativeEstimatedCached,
		CumulativeCompletionTokens:      st.cumulativeCompletion,
		CumulativeCost:                  st.cumulativeCost,
		CumulativeEstimatedCacheSavings: st.cumulativeEstimatedCacheSavings,
		ActiveToolLoop:                  st.activeToolLoop,
		LastDecisionName:                st.lastDecisionName,
		LastDecisionReason:              st.lastDecisionReason,
		LastCacheAccountingSource:       st.lastCacheAccountingSource,
		LastPolicy:                      clonePolicyMap(st.lastPolicy),
		RecentOutcomes:                  cloneTurnOutcomes(st.recentOutcomes),
	}
	s.mu.Unlock()
	return snapshot, true
}

func (s *routerSessionMemoryStore) sessionLocked(sessionID string) *routerSessionState {
	st := s.sessions[sessionID]
	if st != nil {
		return st
	}
	if len(s.sessions) >= maxRouterSessions {
		s.evictOldestLocked()
	}
	st = &routerSessionState{
		sessionID:  sessionID,
		modelTurns: make(map[string]int),
	}
	s.sessions[sessionID] = st
	return st
}

func (s *routerSessionMemoryStore) evictExpiredLocked(now time.Time) {
	for k, v := range s.sessions {
		if now.Sub(v.lastSeen) > routerMemoryTTL {
			delete(s.sessions, k)
		}
	}
}

// evictOldestLocked evicts the oldest session among a bounded random sample
// (approximate LRU — see evictionSampleSize). It is a best-effort safety valve
// for the size cap when TTL eviction did not free room. Callers must hold s.mu.
func (s *routerSessionMemoryStore) evictOldestLocked() {
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

func cloneIntMap(in map[string]int) map[string]int {
	if in == nil {
		return nil
	}
	out := make(map[string]int, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneInt64Slice(in []int64) []int64 {
	if len(in) == 0 {
		return nil
	}
	out := make([]int64, len(in))
	copy(out, in)
	return out
}

// pruneSwitchTimestamps keeps the model-change times newer than ttl. Callers
// pass the append time so the series stays bounded without reads mutating it.
func pruneSwitchTimestamps(timestamps []int64, ttl time.Duration, now time.Time) []int64 {
	if len(timestamps) == 0 || ttl <= 0 || now.IsZero() {
		return timestamps
	}
	cutoff := now.Add(-ttl).UnixMilli()
	kept := timestamps[:0]
	for _, ts := range timestamps {
		if ts >= cutoff {
			kept = append(kept, ts)
		}
	}
	return kept
}

// CountRecentSwitches returns how many recorded model changes fall inside the
// window ending at now. The gate feeds its configured window TTL.
func CountRecentSwitches(timestamps []int64, window time.Duration, now time.Time) int {
	if len(timestamps) == 0 || window <= 0 || now.IsZero() {
		return 0
	}
	cutoff := now.Add(-window).UnixMilli()
	count := 0
	for _, ts := range timestamps {
		if ts >= cutoff {
			count++
		}
	}
	return count
}

func clonePolicyMap(in map[string]interface{}) map[string]interface{} {
	if in == nil {
		return nil
	}
	b, err := json.Marshal(in)
	if err != nil {
		return nil
	}
	var out map[string]interface{}
	if err := json.Unmarshal(b, &out); err != nil {
		return nil
	}
	return out
}

func policyDecisionReason(policy map[string]interface{}) string {
	if policy == nil {
		return ""
	}
	if reason, ok := policy["decision_reason"].(string); ok {
		return reason
	}
	return ""
}

// ResetRouterSessionMemoryForTesting clears router-owned session memory.
func ResetRouterSessionMemoryForTesting() {
	s := globalRouterSessionMemory
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions = make(map[string]*routerSessionState)
}

// routerSessionCount returns the number of tracked sessions (tests only).
func routerSessionCount() int {
	s := globalRouterSessionMemory
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.sessions)
}

// setRouterSessionMemoryNowForTesting overrides the memory store clock.
func setRouterSessionMemoryNowForTesting(fn func() time.Time) {
	s := globalRouterSessionMemory
	s.mu.Lock()
	defer s.mu.Unlock()
	if fn == nil {
		fn = time.Now
	}
	s.nowFn = fn
}
