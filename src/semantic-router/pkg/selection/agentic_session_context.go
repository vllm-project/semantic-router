package selection

import "time"

// AgenticPhase describes where the request sits in a tool-using conversation.
type AgenticPhase string

const (
	AgenticPhaseUnknown       AgenticPhase = ""
	AgenticPhaseUserTurn      AgenticPhase = "user_turn"
	AgenticPhaseToolLoop      AgenticPhase = "tool_loop"
	AgenticPhaseProviderState AgenticPhase = "provider_state"
)

// AgenticSessionContext is the selector-owned view of a multi-turn session.
// It is derived at request time by extproc and intentionally contains facts,
// not policy decisions.
type AgenticSessionContext struct {
	ID                 string
	UserID             string
	TurnIndex          int
	PreviousModel      string
	PreviousResponseID string

	MemoryPresent               bool
	MemoryTurnCount             int
	MemorySwitchCount           int
	MemoryModelTurnCnts         map[string]int
	MemoryPromptTokens          int64
	MemoryCachedTokens          int64
	MemoryEstimatedCachedTokens int64
	MemoryOutputTokens          int64
	MemoryCost                  float64
	MemoryEstimatedCacheSavings float64
	MemoryCacheAccountingSource string
	LastDecisionName            string
	LastDecisionReason          string

	HistoryTokens int
	ContextTokens int

	IdleFor   time.Duration
	IdleKnown bool

	CacheWarmth   float64
	CacheWarmthOK bool

	Phase          AgenticPhase
	ActiveToolLoop bool

	HasNonPortableContext    bool
	NonPortableContextReason string

	ToolCallCount     int
	ToolResultCount   int
	ToolDefinitionCnt int

	ModelContextWindows map[string]int
}

func (s *AgenticSessionContext) idleExpired(timeout time.Duration) bool {
	if s == nil || timeout <= 0 || !s.IdleKnown {
		return false
	}
	return s.IdleFor >= timeout
}
