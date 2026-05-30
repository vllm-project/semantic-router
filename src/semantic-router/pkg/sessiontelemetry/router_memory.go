package sessiontelemetry

import (
	"encoding/json"
	"sync"
	"time"
)

const routerMemoryTTL = 24 * time.Hour

// RouterSessionSnapshot is the router-owned, model-independent memory for a
// session. It is intentionally about routing state, not prompt-visible user
// memory.
type RouterSessionSnapshot struct {
	SessionID string
	UserID    string

	CurrentModel string
	LastSeen     time.Time
	IdleFor      time.Duration

	TurnCount   int
	SwitchCount int
	ModelTurns  map[string]int

	CumulativePromptTokens     int64
	CumulativeCachedTokens     int64
	CumulativeCompletionTokens int64
	CumulativeCost             float64

	ActiveToolLoop     bool
	LastDecisionReason string
	LastPolicy         map[string]interface{}
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
	SessionID          string
	Model              string
	PromptTokens       int
	CachedPromptTokens int
	CompletionTokens   int
	Cost               float64
	Timestamp          time.Time
}

type routerSessionState struct {
	sessionID string
	userID    string

	currentModel string
	lastSeen     time.Time

	turnCount   int
	switchCount int
	modelTurns  map[string]int

	cumulativePrompt     int64
	cumulativeCached     int64
	cumulativeCompletion int64
	cumulativeCost       float64

	activeToolLoop     bool
	lastDecisionReason string
	lastPolicy         map[string]interface{}
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
	st.cumulativePrompt += int64(p.PromptTokens)
	st.cumulativeCached += int64(clampCachedPromptTokens(p.PromptTokens, p.CachedPromptTokens))
	st.cumulativeCompletion += int64(p.CompletionTokens)
	st.cumulativeCost += p.Cost
}

// GetRouterSessionSnapshot returns a clone of the router-owned session memory.
func GetRouterSessionSnapshot(sessionID string, now time.Time) (RouterSessionSnapshot, bool) {
	if sessionID == "" {
		return RouterSessionSnapshot{}, false
	}
	s := globalRouterSessionMemory
	s.mu.Lock()
	defer s.mu.Unlock()

	st := s.sessions[sessionID]
	if st == nil {
		return RouterSessionSnapshot{}, false
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
		return RouterSessionSnapshot{}, false
	}
	return RouterSessionSnapshot{
		SessionID:                  st.sessionID,
		UserID:                     st.userID,
		CurrentModel:               st.currentModel,
		LastSeen:                   st.lastSeen,
		IdleFor:                    idleFor,
		TurnCount:                  st.turnCount,
		SwitchCount:                st.switchCount,
		ModelTurns:                 cloneIntMap(st.modelTurns),
		CumulativePromptTokens:     st.cumulativePrompt,
		CumulativeCachedTokens:     st.cumulativeCached,
		CumulativeCompletionTokens: st.cumulativeCompletion,
		CumulativeCost:             st.cumulativeCost,
		ActiveToolLoop:             st.activeToolLoop,
		LastDecisionReason:         st.lastDecisionReason,
		LastPolicy:                 clonePolicyMap(st.lastPolicy),
	}, true
}

func (s *routerSessionMemoryStore) sessionLocked(sessionID string) *routerSessionState {
	st := s.sessions[sessionID]
	if st != nil {
		return st
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
